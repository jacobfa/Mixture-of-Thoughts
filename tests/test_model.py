import torch

from mot.config import GenerationConfig, LossConfig, TrainConfig
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
    assert all(p.grad is not None for p in params.values())
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
    tiny_model.zero_grad(set_to_none=True)


def test_forward_without_targets(tiny_model):
    tiny_model.train()
    out = tiny_model(PROMPTS, COMPLETIONS, max_length=64)
    assert "routing" not in out and torch.isfinite(out["loss"])


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


def test_cached_decode_matches_full_recompute(tiny_repo):
    """Greedy decoding with backbone + interaction KV caches must equal a single uncached forward
    over the same token stream (checks cache bookkeeping, positions and the alignment mask)."""
    torch.manual_seed(1)
    model = MixtureOfThoughts(tiny_config(tiny_repo, top_k=1)).eval()
    for layer in model.interactions:  # make the interaction path non-trivial
        for proj in layer.reverse_proj:
            torch.nn.init.normal_(proj.weight, std=0.1)
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
