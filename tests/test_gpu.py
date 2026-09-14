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
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu_model.trainable_parameters())
    gpu_model.zero_grad(set_to_none=True)


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
