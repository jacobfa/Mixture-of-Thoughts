from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


POOLINGS = ("mean", "cls", "last")


class PromptEncoder(nn.Module):
    """Frozen text encoder producing the router input. `pooling` should match how the checkpoint was trained:
    mean (DeBERTa, nomic/modernbert-embed), cls (gte-modernbert, UltraBERT-Embed), last (Qwen3-Embedding)."""

    def __init__(
        self,
        name: str,
        max_length: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"router.pooling must be one of {POOLINGS}, got {pooling!r}")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.padding_side = "right"
        self.model = AutoModel.from_pretrained(name, device_map=str(device), dtype=dtype).eval().requires_grad_(False)
        self.max_length = max_length
        self.pooling = pooling
        self.output_dim: int = self.model.config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, texts: list[str]) -> Tensor:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        hidden = self.model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"]
        if self.pooling == "cls":
            return hidden[:, 0]
        if self.pooling == "last":
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), mask.sum(1) - 1]
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
