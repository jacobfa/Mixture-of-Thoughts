from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mot.config import InteractionConfig
from mot.experts.streams import PAD_TIME, ExpertBatch
from mot.interaction.attention import InteractionCache, SharedCrossAttention


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class InteractionLayer(nn.Module):
    """One latent-space interaction: per-expert (norm ->) forward projector into the shared space, a
    pre-norm attention (+ SwiGLU) block over the board of K active streams, and a zero-initialised
    per-expert reverse projector added back onto each expert's residual stream."""

    def __init__(self, expert_dims: list[int], cfg: InteractionConfig, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.shared_dim = cfg.shared_dim
        self.symmetric = cfg.symmetric
        norm = (lambda d: nn.RMSNorm(d, eps=1e-6)) if cfg.norm else (lambda d: nn.Identity())
        self.in_norm = nn.ModuleList(norm(d) for d in expert_dims)
        self.forward_proj = nn.ModuleList(nn.Linear(d, cfg.shared_dim) for d in expert_dims)
        self.reverse_proj = nn.ModuleList(nn.Linear(cfg.shared_dim, d) for d in expert_dims)
        for proj in self.reverse_proj:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        self.attn = None
        self.ffn = None
        if cfg.cross_attention:
            self.attn_norm = norm(cfg.shared_dim)
            self.attn = SharedCrossAttention(
                cfg.shared_dim,
                cfg.num_heads,
                cfg.dropout,
                num_experts=len(expert_dims),
                qk_norm=cfg.norm,
                rope=cfg.rope,
                rope_theta=cfg.rope_theta,
                time_scale=cfg.time_scale,
                expert_bias=cfg.expert_bias,
                num_registers=cfg.num_registers,
            )
            if cfg.ffn:
                self.ffn_norm = norm(cfg.shared_dim)
                self.ffn = SwiGLU(cfg.shared_dim, int(cfg.ffn_mult * cfg.shared_dim))
        if cfg.loops < 1:
            raise ValueError(f"interaction.loops must be >= 1, got {cfg.loops}")
        self.loops = cfg.loops
        self.loop_embed = nn.Embedding(cfg.loops, cfg.shared_dim) if cfg.loops > 1 else None
        if self.loop_embed is not None:
            nn.init.zeros_(self.loop_embed.weight)
        self.dropout = nn.Dropout(cfg.dropout)

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
        loop: int = 0,
    ) -> dict[int, Tensor]:
        device = self.device
        B, K, D = gate.shape[0], self.top_k, self.shared_dim
        latents = {}
        for m, h in hiddens.items():
            z = self.forward_proj[m](self.in_norm[m](h.to(device)))
            if self.loop_embed is not None:
                z = z + self.loop_embed.weight[loop].to(z.dtype)
            latents[m] = z * gate[batches[m].rows, m][:, None, None].to(z.dtype)

        if self.attn is not None:
            ref = next(iter(latents.values()))
            T = ref.shape[1]
            board = ref.new_zeros(B, K, T, D)
            board_time = torch.full((B, K, T), PAD_TIME, dtype=torch.long, device=device)
            board_expert = torch.zeros((B, K), dtype=torch.long, device=device)
            for m, z in latents.items():
                b = batches[m]
                board = board.index_put((b.rows, b.slot), z)
                board_time[b.rows, b.slot] = times[m]
                board_expert[b.rows, b.slot] = m
            normed = self.attn_norm(board).to(board.dtype)
            if self.symmetric:
                attended = self.attn(normed.flatten(1, 2), board_time.flatten(1), normed, board_time, board_expert, cache)
                board = board + attended.view(B, K, T, D)
            else:
                attended = self.attn(normed[:, 0], board_time[:, 0], normed, board_time, board_expert, cache)
                board = torch.cat([board[:, :1] + attended[:, None], board[:, 1:]], dim=1)
            if self.ffn is not None:
                board = board + self.ffn(self.ffn_norm(board).to(board.dtype))
            latents = {m: board[batches[m].rows, batches[m].slot] for m in latents}

        return {
            m: h + self.reverse_proj[m](self.dropout(latents[m])).to(h.device, h.dtype) for m, h in hiddens.items()
        }
