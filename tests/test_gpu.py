import pytest
import torch

from mot.config import GenerationConfig, LossConfig
from mot.model import MixtureOfThoughts
from tests.conftest import tiny_config
from tests.test_model import COMPLETIONS, PROMPTS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

ATTN = ["sdpa", "kernels-community/flash-attn3"]


@pytest.fixture(scope="module", params=ATTN)
def gpu_model(tiny_repo, request):
    torch.manual_seed(0)
    cfg = tiny_config(
        tiny_repo,
        dtype="bfloat16",
        attn_implementation=request.param,
        devices=None,
        main_device="cuda:0",
        gradient_checkpointing=True,
    )
    model = MixtureOfThoughts(cfg)
    used = model.pool[0].model.config._attn_implementation
    if request.param != "sdpa" and used != request.param:
        pytest.skip(f"{request.param} unavailable on this node (loaded {used})")
    return model


def test_train_step_bf16(gpu_model):
    gpu_model.train()
    targets = torch.eye(len(gpu_model.pool))[: len(PROMPTS)]
    out = gpu_model(PROMPTS, COMPLETIONS, max_length=64, expert_targets=targets, loss_weights=LossConfig())
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    params = dict(gpu_model.trainable_modules().named_parameters())
    assert all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None)
    assert params["router.mlp.0.weight"].grad is not None
    for m in set(out["active"].flatten().tolist()):
        assert params[f"interactions.0.reverse_proj.{m}.weight"].grad is not None
    gpu_model.zero_grad(set_to_none=True)


def test_muon_step_bf16(gpu_model):
    from mot.config import OptimConfig
    from mot.optim import build_optimizer

    gpu_model.train()
    opt = build_optimizer(gpu_model, OptimConfig(name="muon"), lr=1e-3, weight_decay=0.01)
    before = [p.detach().clone() for p in gpu_model.trainable_parameters()]
    out = gpu_model(PROMPTS, COMPLETIONS, max_length=64)
    out["loss"].backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    after = list(gpu_model.trainable_parameters())
    assert all(torch.isfinite(p).all() for p in after)
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_generate_bf16(gpu_model):
    gen = GenerationConfig(max_new_tokens=8, do_sample=False, max_prompt_length=64)
    a = gpu_model.generate(PROMPTS, gen)
    b = gpu_model.generate(PROMPTS, gen)
    assert len(a.texts) == len(PROMPTS) and a.token_ids == b.token_ids
    assert all(0 < len(ids) <= gen.max_new_tokens for ids in a.token_ids)


def test_experts_spread_over_gpus(gpu_model):
    n = torch.cuda.device_count()
    devices = {e.device.index for e in gpu_model.pool}
    assert len(devices) == min(n, len(gpu_model.pool))
