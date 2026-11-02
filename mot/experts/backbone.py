from __future__ import annotations

import inspect
from collections import UserDict
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)

PLACEMENTS = ("uniform", "shallow", "intermediate", "deep")
MASK_BUILDERS = {
    "full_attention": create_causal_mask,
    "sliding_attention": create_sliding_window_causal_mask,
    "linear_attention": create_recurrent_attention_mask,
}


def interaction_boundaries(num_layers: int, num_interactions: int, placement: str) -> list[int]:
    """Layer indices (exclusive) after which each of the Q interaction layers is applied. The placement range is
    split into Q+1 pieces so the final decoder segment (the tail) always runs after the last interaction."""
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    lo, hi = {
        "uniform": (0, num_layers),
        "shallow": (0, num_layers // 2),
        "intermediate": (num_layers // 4, 3 * num_layers // 4),
        "deep": (num_layers // 2, num_layers),
    }[placement]
    bounds = [lo + round((i + 1) * (hi - lo) / (num_interactions + 1)) for i in range(num_interactions)]
    if len(set(bounds)) != len(bounds) or bounds[0] <= 0 or bounds[-1] >= num_layers:
        raise ValueError(f"{num_interactions} interaction layers do not fit in layers [{lo}, {hi})")
    return bounds


def layer_segments(num_layers: int, num_interactions: int, placement: str) -> list[tuple[int, int]]:
    """Q+1 contiguous layer ranges; interaction layer q follows segment q, the last segment is the tail."""
    bounds = interaction_boundaries(num_layers, num_interactions, placement)
    starts = [0, *bounds]
    ends = [*bounds, num_layers]
    return list(zip(starts, ends, strict=True))


@dataclass
class ExpertState:
    hidden: Tensor
    layer_types: list[str]
    masks: dict[str, Tensor | None]
    position_embeddings: dict[str, tuple[Tensor, Tensor]]
    position_ids: Tensor
    cache: Cache | None
    per_layer_inputs: Tensor | None = None
    shared_kv: UserDict | None = None

    def layer_kwargs(self, i: int) -> dict:
        kind = self.layer_types[i]
        kwargs = {
            "attention_mask": self.masks[kind],
            "position_embeddings": self.position_embeddings[kind],
            "position_ids": self.position_ids,
            "past_key_values": self.cache,
        }
        if self.per_layer_inputs is not None:
            kwargs["per_layer_input"] = self.per_layer_inputs[:, :, i]
        if self.shared_kv is not None:
            kwargs["shared_kv_states"] = self.shared_kv
        return kwargs


class SoftcappedHead(nn.Module):
    def __init__(self, head: nn.Linear, cap: float):
        super().__init__()
        self.head = head
        self.cap = cap

    @property
    def weight(self) -> Tensor:
        return self.head.weight

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.head(x) / self.cap) * self.cap


class StackedExpert:
    """A frozen decoder-only HF model whose layers are partitioned into segments. Supports Llama-style stacks
    and their hybrids: per-layer-type masks and rotary embeddings (sliding / linear attention), multimodal
    3-D rotary positions, Gemma per-layer inputs and shared KV states."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        num_interactions: int,
        placement: str = "uniform",
    ):
        decoder = model.get_decoder()
        for attr in ("embed_tokens", "layers", "norm", "rotary_emb"):
            if not hasattr(decoder, attr):
                raise TypeError(f"{type(model).__name__} is not a supported Llama-style decoder (missing {attr})")
        self.model = model.requires_grad_(False)
        self.decoder = decoder
        self.config = decoder.config
        self.tokenizer = tokenizer
        num_layers = len(decoder.layers)
        self.segments = layer_segments(num_layers, num_interactions, placement)
        self.hidden_size: int = self.config.hidden_size
        self.vocab_size: int = self.config.vocab_size
        self.pad_token_id: int = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id: int = tokenizer.eos_token_id
        gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        gen_eos = [gen_eos] if isinstance(gen_eos, int) else list(gen_eos or [])
        self.eos_token_ids: list[int] = sorted({self.eos_token_id, *gen_eos})

        default_type = "sliding_attention" if getattr(self.config, "sliding_window", None) else "full_attention"
        self.layer_types = list(getattr(self.config, "layer_types", None) or [default_type] * num_layers)
        unknown = set(self.layer_types) - set(MASK_BUILDERS)
        if unknown:
            raise TypeError(f"{type(model).__name__}: unsupported layer types {sorted(unknown)}")
        self.rope_per_type = "layer_type" in inspect.signature(decoder.rotary_emb.forward).parameters
        rope_params = getattr(self.config, "rope_parameters", None) or {}
        self.mrope = bool(rope_params.get("mrope_section"))
        layer_params = inspect.signature(decoder.layers[0].forward).parameters
        self.per_layer_inputs = bool(getattr(decoder, "hidden_size_per_layer_input", 0)) and "per_layer_input" in layer_params
        self.shared_kv = "shared_kv_states" in layer_params
        cap = getattr(self.config, "final_logit_softcapping", None)
        self._lm_head = SoftcappedHead(model.lm_head, cap) if cap else model.lm_head

    @property
    def device(self) -> torch.device:
        return self.decoder.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder.embed_tokens.weight.dtype

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        if enabled:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
        else:
            self.model.gradient_checkpointing_disable()

    def compile(self, **kwargs) -> None:
        """Regional compilation: one compiled graph per decoder layer, reused across all layers."""
        kwargs.setdefault("dynamic", True)
        for layer in self.decoder.layers:
            layer.compile(**kwargs)

    def train(self, mode: bool) -> None:
        self.model.train(mode)

    def new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.model.config)

    def begin(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        cache: Cache | None = None,
    ) -> ExpertState:
        decoder = self.decoder
        embeds = decoder.embed_tokens(input_ids)
        kinds = set(self.layer_types)
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": embeds,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "position_ids": position_ids,
        }
        masks = {kind: MASK_BUILDERS[kind](**mask_kwargs) for kind in kinds}
        if self.rope_per_type:
            rope = {kind: decoder.rotary_emb(embeds, position_ids, kind) for kind in kinds}
        else:
            rope_positions = position_ids[None].expand(3, -1, -1) if self.mrope else position_ids
            shared = decoder.rotary_emb(embeds, rope_positions)
            rope = dict.fromkeys(kinds, shared)
        per_layer = None
        if self.per_layer_inputs:
            per_layer = decoder.project_per_layer_inputs(embeds, decoder.get_per_layer_inputs(input_ids, embeds))
        return ExpertState(
            hidden=embeds,
            layer_types=self.layer_types,
            masks=masks,
            position_embeddings=rope,
            position_ids=position_ids,
            cache=cache,
            per_layer_inputs=per_layer,
            shared_kv=UserDict() if self.shared_kv else None,
        )

    def run_segment(self, state: ExpertState, index: int) -> Tensor:
        start, end = self.segments[index]
        hidden = state.hidden
        for i in range(start, end):
            hidden = self.decoder.layers[i](hidden, **state.layer_kwargs(i))
        state.hidden = hidden
        return hidden

    def norm(self, hidden: Tensor) -> Tensor:
        return self.decoder.norm(hidden)

    @property
    def lm_head(self):
        return self._lm_head

    def logits(self, hidden: Tensor) -> Tensor:
        return self.lm_head(self.norm(hidden))
