"""Zooter (Lu et al., 2023): distil per-expert rewards into an mdeberta routing function.

Rewards are the bank's 0/1 scores. Tag-based label enhancement replaces each reward vector r by
beta * r + (1 - beta) * mean reward of the query's tag (K-means cluster by default), and the router is trained with
KL(softmax(r*) || Z(q)). Model selection is routed accuracy on bank/val.json.
Writes results/<pool>/zooter/{best_model.pth,meta.json}.
"""

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from common import RESULTS, Tracker, expert_names, load_bank

ENCODER = "microsoft/mdeberta-v3-base"
MAX_LENGTH = 512


class ZooterModel(nn.Module):
    def __init__(self, encoder: str, num_experts: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(encoder)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_experts)

    def forward(self, **inputs) -> torch.Tensor:
        return self.head(self.backbone(**inputs).last_hidden_state[:, 0])


def tokenizer_for(encoder: str):
    return AutoTokenizer.from_pretrained(encoder, truncation_side="left")


def enhance(rewards: np.ndarray, tags: list, beta: float) -> np.ndarray:
    groups = defaultdict(list)
    for i, tag in enumerate(tags):
        groups[tag].append(i)
    out = np.empty_like(rewards)
    for idx in groups.values():
        out[idx] = beta * rewards[idx] + (1 - beta) * rewards[idx].mean(0)
    return out


@torch.no_grad()
def routed_accuracy(model, tokenizer, questions, rewards, device, batch_size) -> float:
    model.eval()
    hits = 0.0
    for i in range(0, len(questions), batch_size):
        inputs = tokenizer(
            questions[i : i + batch_size], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        choice = model(**inputs.to(device)).argmax(-1).cpu().numpy()
        hits += rewards[i : i + batch_size][np.arange(len(choice)), choice].sum()
    model.train()
    return hits / len(questions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--beta", type=float, default=0.3, help="weight of sample-level rewards (paper's best)")
    parser.add_argument("--tag", choices=["cluster_id", "dataset"], default="cluster_id")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experts = expert_names(args.pool)
    out = RESULTS / args.pool / "zooter"
    out.mkdir(parents=True, exist_ok=True)

    questions, rewards, rows = load_bank(args.pool, "train", experts)
    rewards = np.asarray(rewards, dtype=np.float32)
    targets = torch.softmax(torch.from_numpy(enhance(rewards, [r[args.tag] for r in rows], args.beta)), dim=-1)
    val_questions, val_rewards, _ = load_bank(args.pool, "val", experts)
    val_rewards = np.asarray(val_rewards, dtype=np.float32)

    tokenizer = tokenizer_for(args.encoder)
    model = ZooterModel(args.encoder, len(experts)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tracker = Tracker(
        args.pool,
        "router",
        "router/zooter",
        config={**vars(args), "train_rows": len(questions), "val_rows": len(val_questions)},
        tags=["zooter"],
    )

    best, step, order = -1.0, 0, []
    while step < args.steps:
        if not order:
            order = list(range(len(questions)))
            random.shuffle(order)
        batch, order = order[: args.batch_size], order[args.batch_size :]
        inputs = tokenizer(
            [questions[i] for i in batch], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        logits = model(**inputs.to(device))
        loss = F.kl_div(F.log_softmax(logits, dim=-1), targets[batch].to(device), reduction="batchmean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        record = {"train/loss": loss.item()}
        if step % args.eval_every == 0 or step == args.steps:
            acc = routed_accuracy(model, tokenizer, val_questions, val_rewards, device, args.batch_size)
            record["val/routed_accuracy"] = acc
            print(f"step {step}: loss {loss.item():.4f}  val routed acc {acc:.4f}", flush=True)
            if acc > best:
                best = acc
                torch.save(model.state_dict(), out / "best_model.pth")
        tracker.log(record, step=step)
    tracker.summary({"best_val_routed_accuracy": best})
    tracker.finish()

    meta = {
        "encoder": args.encoder,
        "experts": experts,
        "max_length": MAX_LENGTH,
        "beta": args.beta,
        "tag": args.tag,
        "val_accuracy": best,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"best val routed acc {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
