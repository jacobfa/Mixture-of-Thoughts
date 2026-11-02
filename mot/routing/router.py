from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class RoutingDecision:
    scores: Tensor  # [B, M] raw router scores
    active: Tensor  # [B, K] expert ids, descending score; column 0 is the primary expert
    gate: Tensor  # [B, M] equals 1 on active experts in the forward pass; straight-through gradients

    @property
    def primary(self) -> Tensor:
        return self.active[:, 0]

    @property
    def top_k(self) -> int:
        return self.active.shape[1]


def gumbel_like(x: Tensor, eps: float = 1e-10) -> Tensor:
    u = torch.rand_like(x).clamp(eps, 1 - eps)
    return -torch.log(-torch.log(u))


class Router(nn.Module):
    """2-layer MLP scoring experts from a prompt embedding, with Gumbel top-K straight-through selection."""

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int, top_k: int, temperature: float, dropout: float):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k={top_k} exceeds number of experts {num_experts}")
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

    def forward(self, prompt_embedding: Tensor) -> Tensor:
        return self.mlp(prompt_embedding.to(self.mlp[0].weight.dtype))

    def soft(self, scores: Tensor) -> Tensor:
        return F.softmax(scores / self.temperature, dim=-1)

    def select(self, scores: Tensor, noisy: bool, straight_through: bool) -> RoutingDecision:
        """Hard top-K in the forward pass; with `straight_through` the gate carries softmax gradients."""
        logits = scores / self.temperature
        if noisy:
            logits = logits + gumbel_like(logits)
        active = logits.topk(self.top_k, dim=-1).indices
        gate = F.one_hot(active, self.num_experts).sum(1).to(scores.dtype)
        if straight_through:
            soft = F.softmax(logits, dim=-1)
            gate = gate + soft - soft.detach()
        return RoutingDecision(scores=scores, active=active, gate=gate)
