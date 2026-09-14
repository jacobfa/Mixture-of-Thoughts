from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from mot.experts.streams import IGNORE_INDEX

ROUTING_LOSSES = ("ce", "bce", "none")


def _shift(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Next-token alignment restricted to supervised positions: returns ([N, V] logits, [N] targets)."""
    targets = labels[:, 1:]
    keep = targets != IGNORE_INDEX
    return logits[:, :-1][keep], targets[keep]


def lm_loss(logits: dict[int, Tensor], labels: dict[int, Tensor]) -> Tensor:
    """Token-averaged NLL of the primary expert over completion tokens, pooled across primaries."""
    total, count = None, 0
    for m, lg in logits.items():
        lg, tgt = _shift(lg, labels[m])
        term = F.cross_entropy(lg.float(), tgt, reduction="sum")
        total = term if total is None else total + term.to(total.device)
        count += tgt.numel()
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
