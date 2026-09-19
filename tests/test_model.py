import pytest
import torch

from mot.config import GenerationConfig, InteractionConfig, LossConfig, OptimConfig, TrainConfig
from mot.experts.streams import PAD_TIME, ExpertBatch
from mot.model import MixtureOfThoughts, StepInputs
from mot.training import Trainer
from tests.conftest import CORPUS, tiny_config

PROMPTS = [text[: len(text) // 2] for text in CORPUS[:3]]
COMPLETIONS = [text[len(text) // 2 :] for text in CORPUS[:3]]


def test_forward_backward_trains_only_adapters(tiny_model):
    tiny_model.train()
    targets = torch.eye(len(tiny_model.pool))[: len(PROMPTS)]
    out = tiny_model(PROMPTS, COMPLETIONS, max_length=64, expert_targets=targets, loss_weights=LossConfig())
    assert torch.isfinite(out["loss"]) and out["lm"] > 0 and out["routing"] > 0
    assert out["active"].shape == (len(PROMPTS), tiny_model.cfg.top_k)
    out["loss"].backward()

    params = dict(tiny_model.trainable_modules().named_parameters())
    assert params["router.mlp.0.weight"].grad.abs().sum() > 0
    used = set(out["active"].flatten().tolist())
    for m in used:  # reverse projectors are zero-initialised, so they are the first adapters to move
        assert params[f"interactions.0.reverse_proj.{m}.weight"].grad.abs().sum() > 0
    assert all(p.grad is None for e in tiny_model.pool for p in e.model.parameters())

    opt = torch.optim.SGD(params.values(), lr=1e-2)
    opt.step()
    opt.zero_grad(set_to_none=True)
    out = tiny_model(PROMPTS, COMPLETIONS, max_length=64, expert_targets=targets, loss_weights=LossConfig())
    out["loss"].backward()
    for m in set(out["active"].flatten().tolist()):
        assert params[f"interactions.0.forward_proj.{m}.weight"].grad.abs().sum() > 0
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert params[f"interactions.0.attn.{name}.weight"].grad.abs().sum() > 0
    for name in ("attn.registers", "attn.kv_bias.weight", "ffn.down_proj.weight", "attn_norm.weight", "attn.q_norm.weight"):
        assert params[f"interactions.0.{name}"].grad is not None
    tiny_model.zero_grad(set_to_none=True)


def test_forward_without_targets(tiny_model):
    tiny_model.train()
    out = tiny_model(PROMPTS, COMPLETIONS, max_length=64)
    assert "routing" not in out and torch.isfinite(out["loss"])


def test_peer_loss_and_tail_skip_agree(tiny_model):
    """Skipping the peers' tail layers must not change the primary LM loss; the peer loss must be finite."""
    tiny_model.eval()
    with torch.no_grad():
        fast = tiny_model(PROMPTS, COMPLETIONS, max_length=64, loss_weights=LossConfig(peer_lm_weight=0.0))
        full = tiny_model(PROMPTS, COMPLETIONS, max_length=64, loss_weights=LossConfig(peer_lm_weight=0.5))
    assert "peer_lm" not in fast and "peer_lm" in full
    assert torch.allclose(fast["lm"], full["lm"], atol=1e-5)
    assert torch.isfinite(full["peer_lm"]) and full["peer_lm"] > 0
    assert torch.allclose(full["loss"], full["lm"] + 0.5 * full["peer_lm"])
    tiny_model.train()


def test_looped_interaction_trains_and_generates(tiny_repo):
    from mot.optim import build_optimizer

    torch.manual_seed(0)
    ia = InteractionConfig(num_layers=2, shared_dim=16, num_heads=2, dropout=0.0, loops=3)
    model = MixtureOfThoughts(tiny_config(tiny_repo, interaction=ia))
    assert all(layer.loops == 3 and layer.loop_embed.weight.shape == (3, 16) for layer in model.interactions)
    opt = build_optimizer(model, OptimConfig(name="muon"), lr=1e-3, weight_decay=0.01)
    muon_ids = {id(p) for g in opt.param_groups if g.get("use_muon") for p in g["params"]}
    assert all(id(layer.loop_embed.weight) not in muon_ids for layer in model.interactions)
    model.train()
    out = model(PROMPTS, COMPLETIONS, max_length=64)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert model.interactions[0].loop_embed.weight.grad is not None
    gen = GenerationConfig(max_new_tokens=4, do_sample=False, max_prompt_length=64)
    a, b = model.generate(PROMPTS, gen), model.generate(PROMPTS, gen)
    assert a.token_ids == b.token_ids and len(a.texts) == len(PROMPTS)


def test_paper_block_ablation(tiny_repo):
    torch.manual_seed(0)
    cfg = tiny_config(
        tiny_repo,
        interaction=InteractionConfig(
            num_layers=2,
            shared_dim=16,
            num_heads=2,
            dropout=0.0,
            norm=False,
            rope=False,
            expert_bias=False,
            ffn=False,
            symmetric=False,
            num_registers=0,
        ),
    )
    model = MixtureOfThoughts(cfg, pool=None)
    layer = model.interactions[0]
    assert layer.ffn is None and layer.attn.registers is None and layer.attn.kv_bias is None and layer.attn.rope is None
    assert isinstance(layer.attn_norm, torch.nn.Identity) and isinstance(layer.in_norm[0], torch.nn.Identity)
    model.train()
    out = model(PROMPTS, COMPLETIONS, max_length=64)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    gen = model.generate(PROMPTS, GenerationConfig(max_new_tokens=3, do_sample=False, max_prompt_length=64))
    assert len(gen.texts) == len(PROMPTS)


def test_generate_shapes_and_determinism(tiny_model):
    gen = GenerationConfig(max_new_tokens=6, do_sample=False, max_prompt_length=64)
    a = tiny_model.generate(PROMPTS, gen)
    b = tiny_model.generate(PROMPTS, gen)
    assert len(a.texts) == len(PROMPTS)
    assert a.token_ids == b.token_ids
    assert all(0 < len(ids) <= gen.max_new_tokens for ids in a.token_ids)
    assert a.routing.active.shape == (len(PROMPTS), tiny_model.cfg.top_k)


def test_generate_sampling_runs(tiny_model):
    gen = GenerationConfig(max_new_tokens=4, do_sample=True, temperature=0.7, top_p=0.9, max_prompt_length=64)
    out = tiny_model.generate(PROMPTS, gen)
    assert len(out.texts) == len(PROMPTS)


@pytest.mark.parametrize("loops", [1, 2])
def test_cached_decode_matches_full_recompute(tiny_repo, loops):
    """Greedy decoding with backbone + interaction KV caches must equal a single uncached forward
    over the same token stream (checks cache bookkeeping, positions, the alignment mask and per-pass caches)."""
    torch.manual_seed(1)
    ia = InteractionConfig(num_layers=2, shared_dim=16, num_heads=2, dropout=0.0, loops=loops)
    model = MixtureOfThoughts(tiny_config(tiny_repo, top_k=1, interaction=ia)).eval()
    for layer in model.interactions:  # make the interaction path non-trivial
        for proj in layer.reverse_proj:
            torch.nn.init.normal_(proj.weight, std=0.1)
        if layer.loop_embed is not None:
            torch.nn.init.normal_(layer.loop_embed.weight, std=0.1)
    prompt = PROMPTS[0]
    gen = GenerationConfig(max_new_tokens=5, do_sample=False, max_prompt_length=64)
    out = model.generate([prompt], gen)
    m = out.routing.primary.item()
    expert = model.pool[m]

    prompt_ids = expert.tokenizer(prompt, return_offsets_mapping=True)
    ids = torch.tensor([*prompt_ids["input_ids"], *out.token_ids[0]])
    time = torch.tensor(
        [end for _, end in prompt_ids["offset_mapping"]] + [len(prompt) + 1 + t for t in range(len(out.token_ids[0]))]
    )
    batch = ExpertBatch(
        expert=m,
        rows=torch.tensor([0]),
        slot=torch.tensor([0]),
        input_ids=ids[None],
        attention_mask=torch.ones_like(ids)[None],
        time=time[None],
    )
    step = {m: StepInputs(batch.input_ids, batch.attention_mask, batch.position_ids, batch.time)}
    with torch.no_grad():
        hidden = model._run_stacks(step, {m: batch}, out.routing.gate)
        logits = expert.logits(hidden[m])[0]
    n_prompt = len(prompt_ids["input_ids"])
    recomputed = logits[n_prompt - 1 : -1].argmax(-1).tolist()
    assert recomputed == out.token_ids[0]
    assert PAD_TIME not in time.tolist()


def test_save_load_roundtrip(tiny_model, tmp_path):
    tiny_model.save(tmp_path / "ckpt")
    reloaded = MixtureOfThoughts.load(tmp_path / "ckpt", tiny_model.cfg, pool=tiny_model.pool)
    for (ka, a), (kb, b) in zip(
        tiny_model.trainable_modules().state_dict().items(),
        reloaded.trainable_modules().state_dict().items(),
        strict=True,
    ):
        assert ka == kb and torch.equal(a, b)


def test_trainer_smoke(tiny_model, jsonl_data, tmp_path):
    cfg = TrainConfig(
        data=str(jsonl_data),
        output_dir=str(tmp_path / "run"),
        epochs=1,
        batch_size=2,
        lr=1e-3,
        warmup_steps=1,
        max_length=64,
        num_workers=0,
        log_every=1,
        save_every=0,
    )
    trainer = Trainer(tiny_model, cfg)
    trainer.fit()
    assert (tmp_path / "run" / "final" / "mot.safetensors").exists()
    assert (tmp_path / "run" / "metrics.jsonl").read_text().count("\n") == len(trainer.loader)
