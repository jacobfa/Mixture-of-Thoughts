from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from mot.experts.streams import IGNORE_INDEX

ROUTING_LOSSES = ("ce", "bce", "none")

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    _liger_ce = LigerFusedLinearCrossEntropyLoss(reduction="sum")
except ImportError:  # pragma: no cover - optional dependency
    _liger_ce = None


def supervised_positions(hidden: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Next-token alignment restricted to supervised positions: ([N, H] states, [N] targets)."""
    targets = labels[:, 1:]
    keep = targets != IGNORE_INDEX
    return hidden[:, :-1][keep.to(hidden.device)], targets[keep].to(hidden.device)


def _ce_sum(lm_head: nn.Module, hidden: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(lm_head(hidden).float(), targets, reduction="sum")


def linear_cross_entropy(lm_head: nn.Module, hidden: Tensor, targets: Tensor, chunk_size: int, fused: bool) -> Tensor:
    """Sum of token NLLs for lm_head(hidden) without materialising the full [N, V] logits."""
    if hidden.shape[0] == 0:
        return hidden.new_zeros((), dtype=torch.float32)
    if fused and _liger_ce is not None and hidden.is_cuda and isinstance(lm_head, nn.Linear) and lm_head.bias is None:
        return _liger_ce(lm_head.weight, hidden.to(lm_head.weight.dtype), targets).float()
    recompute = torch.is_grad_enabled() and hidden.requires_grad and hidden.shape[0] > chunk_size
    total = None
    for h, t in zip(hidden.split(chunk_size), targets.split(chunk_size)):
        term = checkpoint(_ce_sum, lm_head, h, t, use_reentrant=False) if recompute else _ce_sum(lm_head, h, t)
        total = term if total is None else total + term
    return total


def lm_loss(
    heads: dict[int, tuple[nn.Module, Tensor, Tensor]],
    chunk_size: int = 2048,
    fused: bool = True,
    device: torch.device | str | None = None,
) -> Tensor:
    """Token-averaged NLL over the primary streams. `heads[m] = (lm_head, normed_hidden [N, H], targets [N])`."""
    total, count = None, 0
    for lm_head, hidden, targets in heads.values():
        term = linear_cross_entropy(lm_head, hidden, targets, chunk_size, fused)
        if device is not None:
            term = term.to(device)
        total = term if total is None else total + term.to(total.device)
        count += targets.numel()
    if total is None:
        raise ValueError("lm_loss needs at least one supervised stream")
    return total / max(count, 1)


def routing_loss(scores: Tensor, targets: Tensor, kind: str) -> Tensor:
    """Supervised router loss from per-expert quality labels `targets` [B, M] (e.g. 0/1 correctness)."""
    if kind not in ROUTING_LOSSES:
        raise ValueError(f"routing loss must be one of {ROUTING_LOSSES}, got {kind!r}")
    targets = targets.to(scores.device, scores.dtype)
    if kind == "none":
        return scores.new_zeros(())
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(scores, targets)
    mass = targets.sum(-1, keepdim=True)
    dist = targets / mass.clamp(min=1e-8)
    valid = (mass.squeeze(-1) > 0).to(scores.dtype)
    per_sample = -(dist * F.log_softmax(scores, dim=-1)).sum(-1)
    return (per_sample * valid).sum() / valid.sum().clamp(min=1)
