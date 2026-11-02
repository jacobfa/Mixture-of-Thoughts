from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import get_scheduler, set_seed

from mot import distributed
from mot.config import TrainConfig
from mot.data import Batch, make_loader
from mot.model import WEIGHTS_FILE, MixtureOfThoughts
from mot.optim import MuonAdamW, build_optimizer
from mot.tracking import WandbRun, config_dict

log = logging.getLogger(__name__)
TRAINER_STATE = "trainer.pt"
DS_TAG = "deepspeed"


def fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


class Trainer:
    """Trains router + interaction layers. Single process, `torchrun` data parallel (manual grad all-reduce)
    or DeepSpeed (`train.deepspeed=<json>`); every rank holds a full copy of the frozen experts."""

    def __init__(self, model: MixtureOfThoughts, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg
        self.rank, self.world = distributed.rank(), distributed.world_size()
        self.is_main = self.rank == 0
        self.out_dir = Path(cfg.output_dir)
        if self.is_main:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed)
        self.loader = make_loader(
            cfg.data,
            cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            num_experts=len(model.pool),
            group_by_length=cfg.group_by_length,
            seed=cfg.seed,
            rank=self.rank,
            world_size=self.world,
        )
        if cfg.loss.routing != "none" and not self.loader.dataset.has_targets:
            log.warning(
                "loss.routing=%r but %s has no 'expert_scores'; the router will only receive the LM gradient",
                cfg.loss.routing,
                cfg.data,
            )
        self.params = list(model.trainable_parameters())
        self.total_steps = cfg.epochs * len(self.loader) // cfg.grad_accum
        optimizer = build_optimizer(model, cfg.optim, cfg.lr, cfg.weight_decay)

        def make_scheduler(opt):
            extra = {"min_lr_rate": cfg.min_lr_ratio} if cfg.scheduler == "cosine_with_min_lr" else None
            return get_scheduler(cfg.scheduler, opt, cfg.warmup_steps, self.total_steps, scheduler_specific_kwargs=extra)

        self.engine = None
        if cfg.deepspeed:
            import deepspeed

            ds_cfg = self._deepspeed_config()
            native = "optimizer" in ds_cfg
            if not native and isinstance(optimizer, MuonAdamW) and ds_cfg["zero_optimization"].get("stage", 0) > 0:
                raise ValueError(
                    "ZeRO stage > 0 flattens parameters; use DeepSpeed's native Muon "
                    '("optimizer": {"type": "Muon", ...} in the DeepSpeed json) or zero stage 0'
                )
            self.engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=model.trainable_modules(),
                optimizer=None if native else optimizer,
                model_parameters=self.params if native else None,
                lr_scheduler=make_scheduler,
                config=ds_cfg,
            )
        else:
            self.optimizer, self.scheduler = optimizer, make_scheduler(optimizer)
            distributed.broadcast_params(self.params)

        self.step = 0
        self.epoch = 0
        self.micro = 0
        self._sums: dict[str, torch.Tensor] = {}
        self._usage = torch.zeros(len(model.pool), dtype=torch.long, device=model.main_device)
        self._tokens = torch.zeros((), dtype=torch.long, device=model.main_device)
        self._n = 0
        self._t_last = time.perf_counter()
        self._t_start = self._t_last
        self._step_start = 0
        self.metrics = open(self.out_dir / "metrics.jsonl", "a") if self.is_main else None
        if cfg.resume:
            self.load_checkpoint(cfg.resume)
        self.wandb = WandbRun(cfg.wandb, {"model": config_dict(model.cfg), "train": config_dict(cfg)})
        if self.is_main:
            log.info("config:\n%s", OmegaConf.to_yaml({"model": config_dict(model.cfg), "train": config_dict(cfg)}))
            log.info("trainable parameters: %.1fM", model.num_trainable_parameters() / 1e6)
            log.info(
                "%d ranks, %d micro-batches/epoch/rank, grad_accum=%d, %d optimizer steps",
                self.world,
                len(self.loader),
                cfg.grad_accum,
                self.total_steps,
            )
            self.wandb.summary({"trainable_parameters": model.num_trainable_parameters(), "total_steps": self.total_steps})

    def _deepspeed_config(self) -> dict:
        cfg = self.cfg
        ds = json.loads(Path(cfg.deepspeed).read_text())
        ds.setdefault("train_micro_batch_size_per_gpu", cfg.batch_size)
        ds.setdefault("gradient_accumulation_steps", cfg.grad_accum)
        ds.setdefault("gradient_clipping", cfg.grad_clip)
        ds.setdefault("steps_per_print", 10**9)
        ds.setdefault("zero_optimization", {"stage": 0})
        params = ds.get("optimizer", {}).get("params", {})
        for key in ("lr", "muon_lr", "adam_lr"):
            if params.get(key) == "auto":
                params[key] = cfg.lr
        if params.get("weight_decay") == "auto":
            params["weight_decay"] = cfg.weight_decay
        return ds

    # ------------------------------------------------------------------ loop

    def calibrate(self) -> None:
        ia = self.model.cfg.interaction
        if not ia.calibrate or self.model.is_calibrated:
            return
        dataset = self.loader.dataset
        lengths = dataset.lengths
        order = sorted(range(len(dataset)), key=lambda i: lengths[i])
        mid = len(order) // 2
        rows = [dataset[i] for i in order[mid : mid + ia.calibration_samples]]
        t = time.perf_counter()
        gains = self.model.calibrate([r["prompt"] for r in rows], [r["completion"] for r in rows], self.cfg.max_length)
        distributed.broadcast_params([layer.gain for layer in self.model.interactions])
        if self.is_main:
            log.info("reverse-projection gains [layer, expert] (%.0fs):\n%s", time.perf_counter() - t, gains)
            self.wandb.summary({"gain": gains.tolist()})

    def fit(self) -> None:
        self.calibrate()
        self.model.train()
        self._t_start = self._t_last = time.perf_counter()
        self._step_start = self.step
        show_bar = self.is_main and sys.stderr.isatty()
        for epoch in range(self.epoch, self.cfg.epochs):
            self.epoch = epoch
            t_epoch = time.perf_counter()
            self.loader.batch_sampler.set_epoch(epoch)
            skip = self.micro - epoch * len(self.loader)
            for i, batch in enumerate(tqdm(self.loader, desc=f"epoch {epoch}", dynamic_ncols=True, disable=not show_bar)):
                if i >= skip:
                    self.train_step(batch)
            if self.is_main:
                log.info("epoch %d/%d done in %s", epoch + 1, self.cfg.epochs, fmt_duration(time.perf_counter() - t_epoch))
            self.save_checkpoint(self.out_dir / "latest")
        self.save_checkpoint(self.out_dir / "final")
        if self.is_main:
            log.info("training finished: %d steps in %s", self.step, fmt_duration(time.perf_counter() - self._t_start))
        self.wandb.finish()

    def train_step(self, batch: Batch) -> dict[str, float] | None:
        cfg = self.cfg
        out = self.model(batch.prompts, batch.completions, cfg.max_length, batch.expert_targets, cfg.loss)
        if self.engine is not None:
            self.engine.backward(out["loss"])
        else:
            (out["loss"] / cfg.grad_accum).backward()
        self._accumulate(out, batch)
        self.micro += 1
        boundary = self.micro % cfg.grad_accum == 0

        grad_norm = None
        if self.engine is not None:
            self.engine.step()
            if boundary:
                grad_norm = self.engine.get_global_grad_norm()
        elif boundary:
            distributed.all_reduce_grads(self.params)
            grad_norm = clip_grad_norm_(self.params, cfg.grad_clip, foreach=True)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        if not boundary:
            return None

        self.step += 1
        record = None
        if self.step % cfg.log_every == 0:
            record = self._flush(grad_norm)
        if cfg.save_every and self.step % cfg.save_every == 0:
            self.save_checkpoint(self.out_dir / "latest")
        return record

    # ------------------------------------------------------------------ metrics (no host sync until flush)

    @torch.no_grad()
    def _accumulate(self, out: dict, batch: Batch) -> None:
        for k, v in out.items():
            if k != "tokens" and torch.is_tensor(v) and v.ndim == 0:
                v = v.detach().float()
                self._sums[k] = v if k not in self._sums else self._sums[k] + v
        active = out["active"]
        self._usage += torch.bincount(active.flatten(), minlength=len(self.model.pool))
        if "tokens" in out:
            self._tokens += out["tokens"].to(self._tokens.device)
        if batch.expert_targets is not None:
            targets = batch.expert_targets.to(active.device)
            primary_score = targets.gather(1, active[:, :1]).squeeze(1)
            hit = (primary_score >= targets.max(-1).values).float().mean()
            self._sums["primary_is_best"] = hit if "primary_is_best" not in self._sums else self._sums["primary_is_best"] + hit
        if batch.expert_nll is not None:
            nll = batch.expert_nll.to(active.device)
            for k, v in (("ref_primary", nll.gather(1, active[:, :1]).mean()), ("ref_best", nll.min(-1).values.mean())):
                self._sums[k] = v if k not in self._sums else self._sums[k] + v
        self._n += 1

    def _flush(self, grad_norm) -> dict[str, float]:
        keys = sorted(self._sums)
        stacked = torch.stack([self._sums[k] for k in keys]) / max(self._n, 1)
        stacked = distributed.all_reduce_mean(stacked)
        values = stacked.tolist()
        usage = self._usage.tolist()
        tokens = int(distributed.all_reduce_mean(self._tokens.float()).item() * self.world)
        now = time.perf_counter()
        interval = now - self._t_last
        elapsed = now - self._t_start
        done = self.step - self._step_start
        sec_per_step = elapsed / max(done, 1)
        remaining = max(self.total_steps - self.step, 0)
        eta = remaining * sec_per_step
        record = dict(zip(keys, values, strict=True))
        record.update(
            step=self.step,
            epoch=self.epoch,
            lr=self.scheduler.get_last_lr()[0],
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            sec_per_step=interval / max(self.cfg.log_every, 1),
            tokens_per_sec=tokens / max(interval, 1e-9),
            elapsed=elapsed,
            eta=eta,
            usage=usage,
        )
        if torch.cuda.is_available():
            record["max_mem_gb"] = [round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())]
        self._sums.clear()
        self._usage.zero_()
        self._tokens.zero_()
        self._n = 0
        self._t_last = now
        if self.is_main:
            self.metrics.write(json.dumps(record) + "\n")
            self.metrics.flush()
            log.info(self._format(record, eta))
            self.wandb.log(record, step=self.step)
        return record

    def _format(self, r: dict, eta: float) -> str:
        losses = " ".join(f"{k} {r[k]:.4f}" for k in ("loss", "lm", "peer_lm", "routing") if k in r)
        parts = [
            f"step {r['step']:>{len(str(self.total_steps))}}/{self.total_steps} ({100 * r['step'] / max(self.total_steps, 1):5.1f}%) ep {r['epoch'] + 1}/{self.cfg.epochs}",
            losses,
        ]
        if "ref_primary" in r:
            parts.append(f"lm-ref {r['lm'] - r['ref_primary']:+.4f} (frozen primary {r['ref_primary']:.4f}, best {r['ref_best']:.4f})")
        if "primary_is_best" in r:
            parts.append(f"route-acc {r['primary_is_best']:.2f}")
        parts.append(f"usage {r['usage']}")
        parts.append(f"lr {r['lr']:.2e} gnorm {r['grad_norm']:.3g}")
        parts.append(f"{r['sec_per_step']:.1f}s/step {r['tokens_per_sec'] / 1e3:.1f}k tok/s")
        if "max_mem_gb" in r:
            parts.append("mem " + "/".join(f"{m:.0f}" for m in r["max_mem_gb"]) + "G")
        finish = time.strftime("%H:%M", time.localtime(time.time() + eta))
        parts.append(f"elapsed {fmt_duration(r['elapsed'])} eta {fmt_duration(eta)} (~{finish})")
        return " | ".join(parts)

    # ------------------------------------------------------------------ checkpoints

    def save_checkpoint(self, path: Path) -> None:
        state = {"step": self.step, "epoch": self.epoch, "micro": self.micro}
        if self.engine is not None:
            self.engine.save_checkpoint(str(path), tag=DS_TAG, client_state=state)
        if self.is_main:
            self.model.save(path)
            if self.engine is None:
                state.update(optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict())
                torch.save(state, path / TRAINER_STATE)
            log.info("saved checkpoint to %s (step %d)", path, self.step)
            self.wandb.summary({"last_checkpoint": str(path), "checkpoint_step": self.step})
        distributed.barrier()

    def load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if self.engine is not None:
            _, state = self.engine.load_checkpoint(str(path), tag=DS_TAG)
        else:
            self.model.trainable_modules().load_state_dict(load_file(path / WEIGHTS_FILE, device=str(self.model.main_device)))
            state = torch.load(path / TRAINER_STATE, map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]
        self.micro = state.get("micro", self.step * self.cfg.grad_accum)
        self.epoch = self.micro // len(self.loader)
        log.info("resumed from %s at step %d (epoch %d, micro-batch %d)", path, self.step, self.epoch, self.micro)
