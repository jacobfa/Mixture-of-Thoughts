from __future__ import annotations

import torch
from torch import Tensor, nn

from mot.experts.streams import PAD_TIME, ExpertBatch
from mot.interaction.attention import InteractionCache, SharedCrossAttention


class InteractionLayer(nn.Module):
    """One latent-space interaction: per-expert forward/reverse projectors around a shared
    cross-attention that only the primary stream executes (peers pass through the projector pair)."""

    def __init__(
        self,
        expert_dims: list[int],
        shared_dim: int,
        num_heads: int,
        top_k: int,
        dropout: float,
        cross_attention: bool = True,
    ):
        super().__init__()
        self.top_k = top_k
        self.shared_dim = shared_dim
        self.forward_proj = nn.ModuleList(nn.Linear(d, shared_dim) for d in expert_dims)
        self.reverse_proj = nn.ModuleList(nn.Linear(shared_dim, d) for d in expert_dims)
        for proj in self.reverse_proj:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        self.attn = SharedCrossAttention(shared_dim, num_heads, dropout) if cross_attention else None
        self.dropout = nn.Dropout(dropout)

    @property
    def device(self) -> torch.device:
        return self.forward_proj[0].weight.device

    def forward(
        self,
        hiddens: dict[int, Tensor],
        batches: dict[int, ExpertBatch],
        times: dict[int, Tensor],
        gate: Tensor,
        cache: InteractionCache | None = None,
    ) -> dict[int, Tensor]:
        device = self.device
        B = gate.shape[0]
        latents = {
            m: self.forward_proj[m](h.to(device)) * gate[batches[m].rows, m][:, None, None].to(h.dtype)
            for m, h in hiddens.items()
        }
        if self.attn is not None:
            ref = next(iter(latents.values()))
            T = ref.shape[1]
            board = ref.new_zeros(B, self.top_k, T, self.shared_dim)
            board_time = torch.full((B, self.top_k, T), PAD_TIME, dtype=torch.long, device=device)
            for m, z in latents.items():
                b = batches[m]
                board = board.index_put((b.rows, b.slot), z)
                board_time[b.rows, b.slot] = times[m]
            attended = self.attn(board[:, 0], board_time[:, 0], board, board_time, cache)

        updated = {}
        for m, h in hiddens.items():
            b = batches[m]
            z = latents[m]
            if self.attn is not None:
                z = torch.where(b.primary_rows[:, None, None], attended[b.rows], z)
            updated[m] = h + self.reverse_proj[m](self.dropout(z)).to(h.device, h.dtype)
        return updated
