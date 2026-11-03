"""Train RouterDC (baselines/routerdc/train_router_mdeberta.py) on the bank and log its evaluation curve to wandb.

Writes results/<pool>/routerdc/best_model.pth (selected on bank/val.json) and training_log.json.
"""

import argparse
import json
import subprocess
import sys

from common import RESULTS, ROUTERDC, Tracker, expert_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--cluster-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bank = RESULTS / args.pool / "bank"
    out = RESULTS / args.pool / "routerdc"
    out.mkdir(parents=True, exist_ok=True)
    train, val = bank / "train.json", bank / "val.json"
    with open(train) as f:
        n_train = len(json.load(f))
    n_experts = len(expert_names(args.pool))

    cmd = [
        sys.executable, "train_router_mdeberta.py",
        "--data_paths", str(train), "--test_data_paths", str(val), "--test_data_type", "probability",
        "--training_samples_per_dataset", str(n_train), "--top_k", "1", "--last_k", str(n_experts - 1),
        "--training_steps", str(args.steps), "--eval_steps", str(args.eval_every), "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.lr), "--cluster_loss_weight", str(args.cluster_loss_weight), "--seed", str(args.seed),
        "--save_path", str(out),
    ]  # fmt: skip
    tracker = Tracker(
        args.pool, "router", "router/routerdc", config={**vars(args), "train_rows": n_train}, tags=["routerdc"]
    )
    code = subprocess.run(cmd, cwd=ROUTERDC, check=False).returncode

    log_path = out / "training_log.json"
    best = -1.0
    if log_path.exists():
        with open(log_path) as f:
            entries = json.load(f)
        for i, entry in enumerate(entries):
            record = {"train/routed_accuracy": entry[str(train)][1], "val/routed_accuracy": entry[str(val)][1]}
            tracker.log(record, step=(i + 1) * args.eval_every)
            best = max(best, record["val/routed_accuracy"])
    tracker.summary({"best_val_routed_accuracy": best, "exit_code": code})
    tracker.finish()
    if code != 0:
        sys.exit(code)
    print(f"best val routed acc {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
