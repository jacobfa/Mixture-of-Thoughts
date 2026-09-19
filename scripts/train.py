import argparse
import logging
from pathlib import Path

import torch

from mot import MixtureOfThoughts, distributed, load_config
from mot.tracking import configure_logging
from mot.training import Trainer

log = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MoT router + interaction layers (experts stay frozen).")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.lr=1e-4")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    distributed.init(use_deepspeed=cfg.train.deepspeed is not None)
    run_name = cfg.train.wandb.name or Path(cfg.train.output_dir).name
    configure_logging(cfg.train.log_file or f"logs/{run_name}.log")
    torch.set_float32_matmul_precision("high")
    if distributed.is_distributed():
        devices = distributed.local_devices()
        if cfg.model.devices is None:
            cfg.model.devices = devices
        cfg.model.main_device = cfg.model.devices[0]

    torch.manual_seed(cfg.train.seed)
    model = MixtureOfThoughts(cfg.model)
    trainer = Trainer(model, cfg.train)
    trainer.fit()
    log.info("done: %d optimizer steps, checkpoints in %s", trainer.step, cfg.train.output_dir)


if __name__ == "__main__":
    main()
