from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


class PromptEncoder(nn.Module):
    """Frozen sentence encoder (mean-pooled token states) producing the router input."""

    def __init__(self, name: str, max_length: int, device: torch.device | str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, device_map=str(device)).eval().requires_grad_(False)
        self.max_length = max_length
        self.output_dim: int = self.model.config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, texts: list[str]) -> Tensor:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        hidden = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
