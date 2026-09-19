from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from mot.config import ExpertSpec, ModelConfig
from mot.experts.backbone import StackedExpert

log = logging.getLogger(__name__)


def default_devices() -> list[str]:
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def load_causal_lm(name: str, device: str, dtype: torch.dtype, attn_implementation: str) -> PreTrainedModel:
    kwargs = {"dtype": dtype, "device_map": device, "low_cpu_mem_usage": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation=attn_implementation, **kwargs)
    except Exception as err:  # kernel hub unavailable, CPU-only, unsupported arch, ...
        if attn_implementation == "sdpa":
            raise
        log.warning("attn_implementation=%r failed for %s (%s); falling back to sdpa", attn_implementation, name, err)
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa", **kwargs)
    return model.eval()


def load_tokenizer(spec: ExpertSpec):
    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer or spec.name)
    if not tokenizer.is_fast:
        raise ValueError(f"{spec.name}: a fast tokenizer is required (offset mappings are used for alignment)")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class ExpertPool:
    """Owns the frozen experts and their device placement."""

    def __init__(self, experts: list[StackedExpert]):
        if not experts:
            raise ValueError("ExpertPool needs at least one expert")
        self.experts = experts

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> ExpertPool:
        devices = cfg.devices or default_devices()
        dtype = getattr(torch, cfg.dtype)
        experts = []
        for i, spec in enumerate(cfg.experts):
            device = spec.device or devices[i % len(devices)]
            log.info("loading expert %d: %s -> %s", i, spec.name, device)
            model = load_causal_lm(spec.name, device, dtype, cfg.attn_implementation)
            tokenizer = load_tokenizer(spec)
            expert = StackedExpert(model, tokenizer, cfg.interaction.num_layers, cfg.interaction.placement)
            if cfg.compile:
                expert.compile()
            experts.append(expert)
        return cls(experts)

    def __len__(self) -> int:
        return len(self.experts)

    def __getitem__(self, i: int) -> StackedExpert:
        return self.experts[i]

    def __iter__(self) -> Iterator[StackedExpert]:
        return iter(self.experts)

    @property
    def hidden_sizes(self) -> list[int]:
        return [e.hidden_size for e in self.experts]

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        for e in self.experts:
            e.set_gradient_checkpointing(enabled)

    def train(self, mode: bool) -> None:
        for e in self.experts:
            e.train(mode)
