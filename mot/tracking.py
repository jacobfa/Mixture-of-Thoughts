from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mot import distributed
from mot.config import WandbConfig

log = logging.getLogger(__name__)
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(log_file: str | Path | None, level: int | None = None) -> Path | None:
    """Console + file logging on the main rank (other ranks: console warnings only)."""
    main = distributed.is_main()
    level = level if level is not None else (logging.INFO if main else logging.WARNING)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    path = None
    if log_file and main:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(level=level, format=FORMAT, handlers=handlers, force=True)
    for name in ("transformers", "kernels", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if path is not None:
        log.info("logging to %s", path.resolve())
    return path


def resolve_wandb_mode(mode: str) -> str:
    mode = os.environ.get("WANDB_MODE", mode)
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            if wandb.api.api_key:
                return mode
        except Exception:
            pass
        log.warning("no wandb API key found; logging offline (sync later with `wandb sync`)")
        return "offline"
    return mode


class WandbRun:
    """Thin wrapper: no-op off the main rank or when mode == 'disabled'."""

    def __init__(self, cfg: WandbConfig, config: dict[str, Any], job_type: str = "train", name: str | None = None):
        self.run = None
        mode = resolve_wandb_mode(cfg.mode)
        if not distributed.is_main() or mode == "disabled":
            return
        import wandb

        self.run = wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            name=name or cfg.name,
            mode=mode,
            job_type=job_type,
            config=config,
        )
        log.info("wandb run %s (%s) -> %s", self.run.name, self.run.settings.mode, self.run.url or self.run.dir)

    def log(self, record: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        flat = {}
        for k, v in record.items():
            if isinstance(v, (list, tuple)):
                flat.update({f"{k}/{i}": x for i, x in enumerate(v)})
            else:
                flat[k] = v
        self.run.log(flat, step=step)

    def summary(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.summary.update(values)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


def config_dict(cfg: Any) -> dict:
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
