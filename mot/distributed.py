from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import Tensor


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("LOCAL_SIZE", "1")))


def is_distributed() -> bool:
    return world_size() > 1


def is_main() -> bool:
    return rank() == 0


def local_devices() -> list[str]:
    """Visible GPUs split evenly across the ranks on this node; experts round-robin inside a rank's share."""
    if not torch.cuda.is_available():
        return ["cpu"]
    n, procs = torch.cuda.device_count(), local_world_size()
    if n < procs:
        raise ValueError(f"{procs} ranks on this node but only {n} visible GPUs")
    per_rank = n // procs
    first = local_rank() * per_rank
    return [f"cuda:{i}" for i in range(first, first + per_rank)]


def init(use_deepspeed: bool = False) -> None:
    if not is_distributed():
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device(local_devices()[0]))
    if use_deepspeed:
        import deepspeed

        deepspeed.init_distributed()
    elif not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_params(params: list[Tensor], src: int = 0) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in params:
        dist.broadcast(p.data, src=src)


def all_reduce_grads(params: list[Tensor]) -> None:
    """Average gradients across ranks. Missing grads (e.g. an expert not routed on this rank) are zero-filled
    so every rank participates in the same collectives."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    grads = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    handles = [dist.all_reduce(g, op=dist.ReduceOp.SUM, async_op=True) for g in grads]
    for h in handles:
        h.wait()
    torch._foreach_div_(grads, dist.get_world_size())


def all_reduce_mean(x: Tensor) -> Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    x = x.clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x / dist.get_world_size()
