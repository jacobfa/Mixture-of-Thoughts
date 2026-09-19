import copy
import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mot.config import OptimConfig, TrainConfig
from mot.data import LengthGroupedBatchSampler
from mot.losses import linear_cross_entropy, supervised_positions
from mot.optim import MuonAdamW, build_optimizer
from mot.training import Trainer


def _make(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(24, 40), nn.Linear(40, 3)


def _step(opt, mats, biases):
    for p in [*mats, *biases]:
        p.grad = torch.randn_like(p)
    opt.step()


def test_muon_matches_torch_reference():
    torch_muon = getattr(torch.optim, "Muon", None)
    if torch_muon is None:
        pytest.skip("torch.optim.Muon unavailable")
    a, b = _make()
    a_ref, b_ref = copy.deepcopy(a), copy.deepcopy(b)
    ours = MuonAdamW([{"params": [a.weight, b.weight], "use_muon": True}], lr=0.02, weight_decay=0.1, momentum=0.9)
    ref = torch_muon([a_ref.weight, b_ref.weight], lr=0.02, weight_decay=0.1, momentum=0.9, adjust_lr_fn="match_rms_adamw")
    for seed in range(3):
        torch.manual_seed(seed)
        grads = [torch.randn_like(a.weight), torch.randn_like(b.weight)]
        for (p, p_ref), g in zip(((a.weight, a_ref.weight), (b.weight, b_ref.weight)), grads):
            p.grad, p_ref.grad = g.clone(), g.clone()
        ours.step()
        ref.step()
    assert torch.allclose(a.weight, a_ref.weight, atol=1e-6)
    assert torch.allclose(b.weight, b_ref.weight, atol=1e-6)


def test_adamw_branch_matches_torch_adamw():
    a, b = _make()
    a_ref, b_ref = copy.deepcopy(a), copy.deepcopy(b)
    ours = MuonAdamW(
        [{"params": [a.bias, b.bias, b.weight]}], lr=1e-2, weight_decay=0.05, betas=(0.9, 0.95), eps=1e-8
    )
    ref = torch.optim.AdamW([a_ref.bias, b_ref.bias, b_ref.weight], lr=1e-2, weight_decay=0.05, betas=(0.9, 0.95), eps=1e-8)
    for seed in range(4):
        torch.manual_seed(seed)
        for p, p_ref in ((a.bias, a_ref.bias), (b.bias, b_ref.bias), (b.weight, b_ref.weight)):
            g = torch.randn_like(p)
            p.grad, p_ref.grad = g.clone(), g.clone()
        ours.step()
        ref.step()
    for p, p_ref in ((a.bias, a_ref.bias), (b.bias, b_ref.bias), (b.weight, b_ref.weight)):
        assert torch.allclose(p, p_ref, atol=1e-6)


def test_build_optimizer_groups(tiny_model):
    opt = build_optimizer(tiny_model, OptimConfig(name="muon"), lr=1e-3, weight_decay=0.01)
    assert isinstance(opt, MuonAdamW)
    muon = [p for g in opt.param_groups if g.get("use_muon") for p in g["params"]]
    others = [p for g in opt.param_groups if not g.get("use_muon") for p in g["params"]]
    head = tiny_model.router.mlp[-1].weight
    assert all(p.ndim == 2 for p in muon)
    assert not any(p is head for p in muon) and any(p is head for p in others)
    assert len(muon) + len(others) == len(list(tiny_model.trainable_parameters()))
    no_decay = [g for g in opt.param_groups if g["weight_decay"] == 0.0]
    assert no_decay and all(p.ndim == 1 for p in no_decay[0]["params"])
    adamw = build_optimizer(tiny_model, OptimConfig(name="adamw"), lr=1e-3, weight_decay=0.01)
    assert isinstance(adamw, torch.optim.AdamW)
    with pytest.raises(ValueError):
        build_optimizer(tiny_model, OptimConfig(name="lion"), lr=1e-3, weight_decay=0.01)


def test_chunked_linear_ce_matches_dense():
    torch.manual_seed(0)
    head = nn.Linear(16, 50, bias=False)
    hidden = torch.randn(37, 16, requires_grad=True)
    targets = torch.randint(0, 50, (37,))
    dense = F.cross_entropy(head(hidden), targets, reduction="sum")
    (g_dense,) = torch.autograd.grad(dense, hidden)
    chunked = linear_cross_entropy(head, hidden, targets, chunk_size=8, fused=False)
    (g_chunked,) = torch.autograd.grad(chunked, hidden)
    assert torch.allclose(dense, chunked, atol=1e-5)
    assert torch.allclose(g_dense, g_chunked, atol=1e-6)
    empty = linear_cross_entropy(head, hidden[:0], targets[:0], chunk_size=8, fused=False)
    assert empty == 0


def test_supervised_positions_shift():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float).view(2, 5, 3)
    labels = torch.tensor([[-100, -100, 7, 8, 2], [-100, 4, 5, -100, -100]])
    h, t = supervised_positions(hidden, labels)
    assert t.tolist() == [7, 8, 2, 4, 5]
    assert torch.equal(h, torch.stack([hidden[0, 1], hidden[0, 2], hidden[0, 3], hidden[1, 0], hidden[1, 1]]))


def test_length_grouped_sampler_shards_and_groups():
    lengths = list(range(100, 0, -1))
    world = 3
    seen = []
    for rank in range(world):
        s = LengthGroupedBatchSampler(lengths, batch_size=4, seed=1, rank=rank, world_size=world, mega=2)
        batches = list(s)
        assert len(batches) == len(s) == (100 // 4) // world
        for b in batches:
            ls = [lengths[i] for i in b]
            assert ls == sorted(ls, reverse=True)
        seen += [i for b in batches for i in b]
    assert len(seen) == len(set(seen))
    s = LengthGroupedBatchSampler(lengths, batch_size=4, seed=1)
    first = list(s)
    s.set_epoch(1)
    assert list(s) != first


def test_trainer_grad_accum_and_resume(tiny_model, jsonl_data, tmp_path):
    cfg = TrainConfig(
        data=str(jsonl_data),
        output_dir=str(tmp_path / "run"),
        epochs=2,
        batch_size=1,
        grad_accum=2,
        lr=1e-3,
        warmup_steps=1,
        scheduler="linear",
        max_length=64,
        num_workers=0,
        log_every=1,
        save_every=0,
    )
    trainer = Trainer(tiny_model, cfg)
    assert trainer.total_steps == (2 * 5) // 2
    trainer.fit()
    assert trainer.step == trainer.total_steps
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == trainer.step
    rec = json.loads(lines[-1])
    assert set(rec) >= {"loss", "lm", "routing", "step", "lr", "grad_norm", "usage", "primary_is_best"}
    assert sum(rec["usage"]) == cfg.grad_accum * cfg.batch_size * tiny_model.cfg.top_k

    resumed = Trainer(tiny_model, TrainConfig(**{**cfg.__dict__, "resume": str(tmp_path / "run" / "final"), "epochs": 3}))
    assert resumed.step == trainer.step and resumed.epoch == 2
    assert resumed.scheduler.last_epoch == trainer.step
