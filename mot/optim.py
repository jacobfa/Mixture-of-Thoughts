from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from mot.config import OptimConfig

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
OPTIMIZERS = ("muon", "adamw")


def newton_schulz(g: Tensor, steps: int, eps: float, coefficients: tuple[float, float, float] = NS_COEFFICIENTS) -> Tensor:
    """Quintic Newton-Schulz iteration approximating UV^T for G = USV^T, run in bf16."""
    a, b, c = coefficients
    x = g.bfloat16()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / x.norm().clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    return x.T if transposed else x


def muon_lr_scale(shape: torch.Size, rule: str) -> float:
    rows, cols = shape[:2]
    if rule == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, cols))
    if rule == "original":
        return math.sqrt(max(1.0, rows / cols))
    raise ValueError(f"unknown Muon lr rule {rule!r}")


class MuonAdamW(Optimizer):
    """One optimizer, one scheduler: Muon on param groups flagged `use_muon` (2D hidden weights),
    AdamW elsewhere. Muon steps are RMS-matched to AdamW so lr / weight decay are shared."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        adjust_lr: str = "match_rms_adamw",
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "ns_eps": ns_eps,
            "betas": betas,
            "eps": eps,
            "adjust_lr": adjust_lr,
            "use_muon": False,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["use_muon"] and any(p.ndim != 2 for p in group["params"]):
                raise ValueError("Muon param groups must contain only 2D parameters")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            if group["use_muon"]:
                self._muon(group, params)
            else:
                self._adamw(group, params)
        return loss

    def _muon(self, group: dict, params: list[Tensor]) -> None:
        lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
        for p in params:
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - mu)
            update = p.grad.lerp(buf, mu) if group["nesterov"] else buf
            update = newton_schulz(update, group["ns_steps"], group["ns_eps"])
            p.mul_(1 - lr * wd)
            p.add_(update.to(p.dtype), alpha=-lr * muon_lr_scale(p.shape, group["adjust_lr"]))

    def _adamw(self, group: dict, params: list[Tensor]) -> None:
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        b1, b2 = group["betas"]
        for p in params:
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.lerp_(p.grad, 1 - b1)
            exp_avg_sq.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(1 - b2**t)).add_(eps)
            p.mul_(1 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-lr / (1 - b1**t))


def param_groups(named_params: Iterable[tuple[str, Tensor]], muon: bool, exclude: set[int]) -> list[dict]:
    """2D hidden weights -> Muon; embeddings, output head, registers -> AdamW with decay; 1D -> AdamW without decay."""
    hidden, decay, no_decay = [], [], []
    for _, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            no_decay.append(p)
        elif muon and p.ndim == 2 and id(p) not in exclude:
            hidden.append(p)
        else:
            decay.append(p)
    groups = []
    if hidden:
        groups.append({"params": hidden, "use_muon": True})
    if decay:
        groups.append({"params": decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(model, cfg: OptimConfig, lr: float, weight_decay: float) -> Optimizer:
    if cfg.name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {cfg.name!r}")
    use_muon = cfg.name == "muon"
    modules = model.trainable_modules()
    exclude = {id(model.router.mlp[-1].weight)}
    exclude |= {id(mod.weight) for mod in modules.modules() if isinstance(mod, nn.Embedding)}
    groups = param_groups(modules.named_parameters(), use_muon, exclude)
    if use_muon:
        for g in groups:
            if g.get("use_muon") and cfg.muon_lr is not None:
                g["lr"] = cfg.muon_lr
        return MuonAdamW(
            groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            ns_steps=cfg.ns_steps,
            betas=tuple(cfg.betas),
            eps=cfg.eps,
            adjust_lr=cfg.adjust_lr,
        )
    device = groups[0]["params"][0].device
    return AdamW(groups, lr=lr, weight_decay=weight_decay, betas=tuple(cfg.betas), eps=cfg.eps, fused=device.type == "cuda")
