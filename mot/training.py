from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, set_seed

from mot.config import TrainConfig
from mot.data import Batch, make_loader
from mot.model import WEIGHTS_FILE, MixtureOfThoughts

log = logging.getLogger(__name__)
TRAINER_STATE = "trainer.pt"


class Trainer:
    def __init__(self, model: MixtureOfThoughts, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed)
        self.loader = make_loader(
            cfg.data, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, num_experts=len(model.pool)
        )
        if cfg.loss.routing != "none" and not self.loader.dataset.has_targets:
            log.warning(
                "loss.routing=%r but %s has no 'expert_scores'; the router will only receive the LM gradient",
                cfg.loss.routing,
                cfg.data,
            )
        self.optimizer = torch.optim.AdamW(
            model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=model.main_device.type == "cuda"
        )
        total_steps = cfg.epochs * len(self.loader)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, cfg.warmup_steps, total_steps)
        self.step = 0
        self.epoch = 0
        self.metrics = open(self.out_dir / "metrics.jsonl", "a")
        if cfg.resume:
            self.load_checkpoint(cfg.resume)
        log.info("trainable parameters: %.1fM", model.num_trainable_parameters() / 1e6)
        log.info("%d steps/epoch, %d total optimizer steps", len(self.loader), total_steps)

    def fit(self) -> None:
        self.model.train()
        for epoch in range(self.epoch, self.cfg.epochs):
            self.epoch = epoch
            for batch in tqdm(self.loader, desc=f"epoch {epoch}", dynamic_ncols=True):
                self.train_step(batch)
            self.save_checkpoint(self.out_dir / "latest")
        self.save_checkpoint(self.out_dir / "final")

    def train_step(self, batch: Batch) -> dict[str, float]:
        t0 = time.perf_counter()
        out = self.model(batch.prompts, batch.completions, self.cfg.max_length, batch.expert_targets, self.cfg.loss)
        out["loss"].backward()
        grad_norm = clip_grad_norm_(self.model.trainable_parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1

        record = {k: v.item() for k, v in out.items() if torch.is_tensor(v) and v.ndim == 0}
        record.update(
            step=self.step,
            epoch=self.epoch,
            lr=self.scheduler.get_last_lr()[0],
            grad_norm=float(grad_norm),
            step_time=time.perf_counter() - t0,
            usage=torch.bincount(out["active"].flatten(), minlength=len(self.model.pool)).tolist(),
        )
        if batch.expert_targets is not None:
            targets = batch.expert_targets.to(out["active"].device)
            primary_score = targets.gather(1, out["active"][:, :1]).squeeze(1)
            record["primary_is_best"] = (primary_score >= targets.max(-1).values).float().mean().item()
        if self.step % self.cfg.log_every == 0:
            self.metrics.write(json.dumps(record) + "\n")
            self.metrics.flush()
            log.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
        if self.cfg.save_every and self.step % self.cfg.save_every == 0:
            self.save_checkpoint(self.out_dir / "latest")
        return record

    def save_checkpoint(self, path: Path) -> None:
        self.model.save(path)
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            },
            path / TRAINER_STATE,
        )
        log.info("saved checkpoint to %s (step %d)", path, self.step)

    def load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        self.model.trainable_modules().load_state_dict(load_file(path / WEIGHTS_FILE, device=str(self.model.main_device)))
        state = torch.load(path / TRAINER_STATE, map_location="cpu")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]
        self.epoch = state["epoch"] + 1
        log.info("resumed from %s at step %d", path, self.step)
