from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

PLACEMENTS = ("uniform", "shallow", "intermediate", "deep")


def interaction_boundaries(num_layers: int, num_interactions: int, placement: str) -> list[int]:
    """Layer indices (exclusive) after which each of the Q interaction layers is applied."""
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    lo, hi = {
        "uniform": (0, num_layers),
        "shallow": (0, num_layers // 2),
        "intermediate": (num_layers // 4, 3 * num_layers // 4),
        "deep": (num_layers // 2, num_layers),
    }[placement]
    bounds = [lo + round((i + 1) * (hi - lo) / num_interactions) for i in range(num_interactions)]
    if len(set(bounds)) != len(bounds) or bounds[0] <= 0:
        raise ValueError(f"{num_interactions} interaction layers do not fit in layers [{lo}, {hi})")
    return bounds


def layer_segments(num_layers: int, num_interactions: int, placement: str) -> list[tuple[int, int]]:
    """Q+1 contiguous layer ranges; interaction layer q follows segment q, the last segment is the tail."""
    bounds = interaction_boundaries(num_layers, num_interactions, placement)
    starts = [0, *bounds]
    ends = [*bounds, num_layers]
    return list(zip(starts, ends))


@dataclass
class ExpertState:
    hidden: Tensor
    attention_mask: Tensor | None
    position_ids: Tensor
    cache_position: Tensor
    position_embeddings: tuple[Tensor, Tensor]
    cache: Cache | None


class StackedExpert:
    """A frozen decoder-only HF model whose layers are partitioned into segments."""

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
        self.tokenizer = tokenizer
        self.segments = layer_segments(len(decoder.layers), num_interactions, placement)
        self.hidden_size: int = model.config.hidden_size
        self.vocab_size: int = model.config.vocab_size
        self.pad_token_id: int = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id: int = tokenizer.eos_token_id

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
        embeds = self.decoder.embed_tokens(input_ids)
        past = cache.get_seq_length() if cache is not None else 0
        cache_position = torch.arange(past, past + input_ids.shape[1], device=embeds.device)
        mask_fn = (
            create_sliding_window_causal_mask
            if getattr(self.model.config, "sliding_window", None)
            else create_causal_mask
        )
        causal_mask = mask_fn(
            config=self.model.config,
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=cache,
            position_ids=position_ids,
        )
        position_embeddings = self.decoder.rotary_emb(embeds, position_ids=position_ids)
        return ExpertState(embeds, causal_mask, position_ids, cache_position, position_embeddings, cache)

    def select_rows(self, state: ExpertState, rows: Tensor) -> ExpertState | None:
        """Restrict a (cache-free) state to a subset of batch rows; None when the mask cannot be sliced."""
        mask = state.attention_mask
        if state.cache is not None or (mask is not None and not isinstance(mask, Tensor)):
            return None
        if mask is not None and mask.shape[0] != 1:
            mask = mask[rows]
        cos, sin = state.position_embeddings
        if cos.shape[0] != 1:
            cos, sin = cos[rows], sin[rows]
        pos = state.position_ids if state.position_ids.shape[0] == 1 else state.position_ids[rows]
        return ExpertState(state.hidden[rows], mask, pos, state.cache_position, (cos, sin), None)

    def run_segment(self, state: ExpertState, index: int) -> Tensor:
        start, end = self.segments[index]
        hidden = state.hidden
        for layer in self.decoder.layers[start:end]:
            hidden = layer(
                hidden,
                attention_mask=state.attention_mask,
                position_ids=state.position_ids,
                past_key_values=state.cache,
                use_cache=state.cache is not None,
                cache_position=state.cache_position,
                position_embeddings=state.position_embeddings,
            )
        state.hidden = hidden
        return hidden

    def norm(self, hidden: Tensor) -> Tensor:
        return self.decoder.norm(hidden)

    @property
    def lm_head(self):
        return self.model.lm_head

    def logits(self, hidden: Tensor) -> Tensor:
        return self.lm_head(self.norm(hidden))
