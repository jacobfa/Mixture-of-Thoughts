"""LoraRetriever (Zhao et al., 2024) retrieval ("Selection") over the expert pool.

Each expert is embedded as the mean instructed embedding of the calibration queries it solved; a query is routed to
the expert with the highest cosine similarity. The sentence encoder is first fine-tuned with the paper's contrastive
loss: for an anchor solved by expert e, the positive is another query e solved and the negatives are queries e failed.
Instructor-xl is replaced by a transformers-native instruction-tuned embedder. Model selection is routed accuracy on
bank/val.json. Writes results/<pool>/loraretriever/{encoder/,centroids.npy,meta.json}.
"""

import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from common import RESULTS, expert_names, load_bank

ENCODER = "BAAI/bge-base-en-v1.5"
INSTRUCTION = "Represent the sentence for similar task retrieval: "


class Encoder(nn.Module):
    def __init__(self, path: str, pooling: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path)
        self.pooling = pooling

    def forward(self, **inputs) -> torch.Tensor:
        hidden = self.backbone(**inputs).last_hidden_state
        if self.pooling == "cls":
            emb = hidden[:, 0]
        else:
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(1) / mask.sum(1)
        return F.normalize(emb, dim=-1)


def encode(model, tokenizer, texts, instruction, max_length, device, batch_size=128, grad=False) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), batch_size):
        inputs = tokenizer(
            [instruction + t for t in texts[i : i + batch_size]],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.set_grad_enabled(grad):
            out.append(model(**inputs))
    return torch.cat(out)


def centroids(embeddings: torch.Tensor, rewards: np.ndarray, per_expert: int, rng: random.Random) -> torch.Tensor:
    rows = []
    for e in range(rewards.shape[1]):
        solved = np.flatnonzero(rewards[:, e] > 0).tolist()
        if 0 < per_expert < len(solved):
            solved = rng.sample(solved, per_expert)
        rows.append(embeddings[solved].mean(0))
    return torch.stack(rows)


def route(queries: torch.Tensor, cents: torch.Tensor) -> torch.Tensor:
    return (queries @ F.normalize(cents, dim=-1).T).argmax(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--instruction", default=INSTRUCTION)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--negatives", type=int, default=7)
    parser.add_argument(
        "--samples-per-expert", type=int, default=0, help="queries averaged per expert embedding; 0 = all solved"
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experts = expert_names(args.pool)
    out = RESULTS / args.pool / "loraretriever"
    out.mkdir(parents=True, exist_ok=True)

    questions, rewards, _ = load_bank(args.pool, "train", experts)
    rewards = np.asarray(rewards, dtype=np.float32)
    val_questions, val_rewards, _ = load_bank(args.pool, "val", experts)
    val_rewards = np.asarray(val_rewards, dtype=np.float32)
    solved_by = [np.flatnonzero(rewards[:, e] > 0).tolist() for e in range(len(experts))]
    failed_by = [np.flatnonzero(rewards[:, e] == 0).tolist() for e in range(len(experts))]
    usable = [e for e in range(len(experts)) if len(solved_by[e]) > 1 and len(failed_by[e]) >= args.negatives]
    anchors = [i for i in range(len(questions)) if any(rewards[i, e] > 0 for e in usable)]

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = Encoder(args.encoder, args.pooling).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def enc(texts: list[str], grad: bool = False) -> torch.Tensor:
        return encode(model, tokenizer, texts, args.instruction, args.max_length, device, grad=grad)

    def evaluate() -> tuple[float, torch.Tensor]:
        model.eval()
        cents = centroids(enc(questions), rewards, args.samples_per_expert, random.Random(args.seed))
        choice = route(enc(val_questions), cents).cpu().numpy()
        model.train()
        return float(val_rewards[np.arange(len(choice)), choice].mean()), cents

    best, step = -1.0, 0
    while step < args.steps:
        batch = rng.sample(anchors, min(args.batch_size, len(anchors)))
        pos, neg = [], []
        for i in batch:
            e = rng.choice([e for e in usable if rewards[i, e] > 0])
            pos.append(rng.choice([j for j in solved_by[e] if j != i]))
            neg.extend(rng.sample(failed_by[e], args.negatives))
        emb = enc([questions[i] for i in [*batch, *pos, *neg]], grad=True)
        a, p, n = (
            emb[: len(batch)],
            emb[len(batch) : 2 * len(batch)],
            emb[2 * len(batch) :].view(len(batch), args.negatives, -1),
        )
        logits = torch.cat([(a * p).sum(-1, keepdim=True), torch.einsum("bd,bkd->bk", a, n)], dim=1) / args.temperature
        loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            acc, cents = evaluate()
            print(f"step {step}: loss {loss.item():.4f}  val routed acc {acc:.4f}", flush=True)
            if acc > best:
                best = acc
                model.backbone.save_pretrained(out / "encoder")
                tokenizer.save_pretrained(out / "encoder")
                np.save(out / "centroids.npy", cents.cpu().numpy())

    meta = {
        "encoder": args.encoder,
        "pooling": args.pooling,
        "instruction": args.instruction,
        "max_length": args.max_length,
        "experts": experts,
        "samples_per_expert": args.samples_per_expert,
        "val_accuracy": best,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"best val routed acc {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
