import argparse
import logging

import torch

from mot import MixtureOfThoughts, distributed, load_config
from mot.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MoT router + interaction layers (experts stay frozen).")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.lr=1e-4")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    distributed.init(use_deepspeed=cfg.train.deepspeed is not None)
    logging.basicConfig(
        level=logging.INFO if distributed.is_main() else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    torch.set_float32_matmul_precision("high")
    if distributed.is_distributed():
        devices = distributed.local_devices()
        if cfg.model.devices is None:
            cfg.model.devices = devices
        cfg.model.main_device = cfg.model.devices[0]

    torch.manual_seed(cfg.train.seed)
    model = MixtureOfThoughts(cfg.model)
    Trainer(model, cfg.train).fit()


if __name__ == "__main__":
    main()
