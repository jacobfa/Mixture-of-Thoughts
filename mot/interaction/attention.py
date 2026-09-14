from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def alignment_mask(query_time: Tensor, key_time: Tensor) -> Tensor:
    """Block-causal mask over concatenated expert streams: a query may attend to any key whose
    alignment coordinate is not in its future. Shape [B, 1, Tq, Tk], True = attend."""
    return key_time[:, None, None, :] <= query_time[:, None, :, None]


@dataclass
class InteractionCache:
    """Per-interaction-layer KV cache over the K active expert streams."""

    keys: Tensor | None = None  # [B, K, T, H, dh]
    values: Tensor | None = None
    time: Tensor | None = None  # [B, K, T]

    def append(self, keys: Tensor, values: Tensor, time: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.keys is None:
            self.keys, self.values, self.time = keys, values, time
        else:
            self.keys = torch.cat([self.keys, keys], dim=2)
            self.values = torch.cat([self.values, values], dim=2)
            self.time = torch.cat([self.time, time], dim=2)
        return self.keys, self.values, self.time


class SharedCrossAttention(nn.Module):
    """Multi-head cross-attention in the shared latent space. Queries come from the primary expert's
    stream; keys/values from all K active streams (slot-major layout [B, K, T, ...])."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"shared_dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        query: Tensor,
        query_time: Tensor,
        board: Tensor,
        board_time: Tensor,
        cache: InteractionCache | None = None,
    ) -> Tensor:
        B, K, T, _ = board.shape
        H, dh = self.num_heads, self.head_dim
        k = self.k_proj(board).view(B, K, T, H, dh)
        v = self.v_proj(board).view(B, K, T, H, dh)
        if cache is not None:
            k, v, board_time = cache.append(k, v, board_time)
        k = k.flatten(1, 2).transpose(1, 2)
        v = v.flatten(1, 2).transpose(1, 2)
        q = self.q_proj(query).view(B, -1, H, dh).transpose(1, 2)
        mask = alignment_mask(query_time, board_time.flatten(1))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.o_proj(out.transpose(1, 2).reshape(B, -1, H * dh))
