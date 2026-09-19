from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Alignment coordinate of register slots: below every real coordinate, so every query may attend to them.
REGISTER_TIME = -1
# Clamp for RoPE positions; padding uses a huge sentinel that must not reach the trig functions.
MAX_ROPE_TIME = 2**24


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


def rotate_half(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


class TimeRotary(nn.Module):
    """RoPE whose position is the alignment coordinate (character offset) divided by `time_scale`.
    Works for any layout [..., H, dh] with time [...]."""

    def __init__(self, head_dim: int, theta: float, time_scale: float):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim={head_dim} must be even for RoPE")
        self.time_scale = time_scale
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        pos = time.to(torch.float32).clamp(min=-self.time_scale, max=MAX_ROPE_TIME) / self.time_scale
        freqs = pos.unsqueeze(-1) * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(-2)
        xf = x.float()
        return (xf * emb.cos() + rotate_half(xf) * emb.sin()).to(x.dtype)


class SharedCrossAttention(nn.Module):
    """Multi-head attention in the shared latent space over the K active expert streams laid out
    slot-major as [B, K, T, ...]. Queries are either the primary stream or (symmetric) every stream."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        num_experts: int,
        qk_norm: bool = True,
        rope: bool = True,
        rope_theta: float = 10000.0,
        time_scale: float = 4.0,
        expert_bias: bool = True,
        num_registers: int = 0,
    ):
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
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.rope = TimeRotary(self.head_dim, rope_theta, time_scale) if rope else None
        self.kv_bias = nn.Embedding(num_experts, 2 * dim) if expert_bias else None
        if self.kv_bias is not None:
            nn.init.zeros_(self.kv_bias.weight)
        self.registers = nn.Parameter(torch.randn(num_registers, 2, dim) * 0.02) if num_registers else None

    def forward(
        self,
        query: Tensor,  # [B, Tq, D]
        query_time: Tensor,  # [B, Tq]
        board: Tensor,  # [B, K, T, D]
        board_time: Tensor,  # [B, K, T]
        board_expert: Tensor,  # [B, K] expert id occupying each slot
        cache: InteractionCache | None = None,
    ) -> Tensor:
        B, K, T, _ = board.shape
        H, dh = self.num_heads, self.head_dim
        k = self.k_proj(board).view(B, K, T, H, dh)
        v = self.v_proj(board).view(B, K, T, H, dh)
        dtype = v.dtype
        if self.kv_bias is not None:
            kb, vb = self.kv_bias(board_expert).view(B, K, 1, H, 2 * dh).chunk(2, dim=-1)
            k = k + kb.to(dtype)
            v = v + vb.to(dtype)
        k = self.k_norm(k).to(dtype)
        if self.rope is not None:
            k = self.rope(k, board_time)
        if cache is not None:
            k, v, board_time = cache.append(k, v, board_time)
        k = k.flatten(1, 2).transpose(1, 2)  # [B, H, N, dh]
        v = v.flatten(1, 2).transpose(1, 2)
        key_time = board_time.flatten(1)

        q = self.q_norm(self.q_proj(query).view(B, -1, H, dh)).to(dtype)
        if self.rope is not None:
            q = self.rope(q, query_time)
        q = q.transpose(1, 2)

        if self.registers is not None:
            R = self.registers.shape[0]
            rk, rv = self.registers.to(dtype).unbind(1)
            rk = self.k_norm(rk.view(R, H, dh)).to(dtype).view(1, R, H, dh).expand(B, R, H, dh).transpose(1, 2)
            rv = rv.view(1, R, H, dh).expand(B, R, H, dh).transpose(1, 2)
            k = torch.cat([k, rk], dim=2)
            v = torch.cat([v, rv], dim=2)
            key_time = torch.cat([key_time, key_time.new_full((B, R), REGISTER_TIME)], dim=1)

        mask = alignment_mask(query_time, key_time)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(out.transpose(1, 2).reshape(B, -1, H * dh))
