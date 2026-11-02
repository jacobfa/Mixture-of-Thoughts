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
    kwargs = {"dtype": dtype, "device_map": device}
    try:
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation=attn_implementation, **kwargs)
    except Exception as err:  # kernel hub unavailable, CPU-only, unsupported arch, ...
        if attn_implementation == "sdpa":
            raise
        log.warning("attn_implementation=%r failed for %s (%s); falling back to sdpa", attn_implementation, name, err)
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa", **kwargs)
    drop_non_text_modules(model)
    return model.eval()


def drop_non_text_modules(model: PreTrainedModel) -> None:
    """Free the vision/audio towers of a multimodal checkpoint; only the text decoder and lm_head are used."""
    decoder = model.get_decoder()
    parents = [model, *(c for _, c in model.named_children() if c is not decoder and any(m is decoder for m in c.modules()))]
    for parent in parents:
        for name, child in list(parent.named_children()):
            if child is decoder or child is model.lm_head or any(m is decoder for m in child.modules()):
                continue
            if next(child.parameters(), None) is not None:
                log.info("dropping %s.%s (%.2fB params)", type(parent).__name__, name, sum(p.numel() for p in child.parameters()) / 1e9)
                delattr(parent, name)


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
            model = load_causal_lm(spec.name, device, dtype, spec.attn_implementation or cfg.attn_implementation)
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
