import argparse
import logging

from mot import MixtureOfThoughts, load_config
from mot.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MoT router + interaction layers (experts stay frozen).")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.lr=1e-4")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.overrides)
    model = MixtureOfThoughts(cfg.model)
    Trainer(model, cfg.train).fit()


if __name__ == "__main__":
    main()
