"""Add routing labels to a prompt/completion JSONL.

For every row and every expert, the completion's mean token NLL is computed with that expert alone (no interaction
layers). `expert_nll` stores the values and `expert_scores` is one-hot on the lowest-NLL expert, which feeds the
supervised routing loss (loss.routing=ce).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm

from mot import load_config
from mot.experts import ExpertPool, build_batches
from mot.losses import linear_cross_entropy, supervised_positions
from mot.tracking import configure_logging

log = logging.getLogger("arc_expert_scores")


@torch.no_grad()
def completion_nll(pool: ExpertPool, m: int, prompts: list[str], completions: list[str], max_length: int) -> list[float]:
    expert = pool[m]
    active = torch.full((len(prompts), 1), m, dtype=torch.long)
    batch = build_batches(pool, active, prompts, completions, max_length, "right", add_eos=True)[m].to(expert.device)
    state = expert.begin(batch.input_ids, batch.attention_mask, batch.position_ids)
    for q in range(len(expert.segments)):
        expert.run_segment(state, q)
    hidden = expert.norm(state.hidden)
    out = []
    for b in range(len(prompts)):
        h, targets = supervised_positions(hidden[b : b + 1], batch.labels[b : b + 1])
        nll = linear_cross_entropy(expert.lm_head, h, targets, chunk_size=2048, fused=False)
        out.append(nll.item() / max(targets.numel(), 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/arc/train.jsonl")
    parser.add_argument("--out", default=None, help="default: overwrite --data")
    parser.add_argument("--tokens-per-batch", type=int, default=24576)
    parser.add_argument("--max-length", type=int, default=16384)
    args = parser.parse_args()
    configure_logging("logs/arc_expert_scores.log")
    cfg = load_config(args.config)

    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    order = sorted(range(len(rows)), key=lambda i: rows[i].get("tokens", len(rows[i]["prompt"])), reverse=True)
    batches, cur, cur_tokens = [], [], 0
    for i in order:
        n = rows[i].get("tokens", len(rows[i]["prompt"]) // 2)
        if cur and cur_tokens + n > args.tokens_per_batch:
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += n
    if cur:
        batches.append(cur)

    pool = ExpertPool.from_config(cfg.model)
    pool.train(False)
    nll = torch.zeros(len(rows), len(pool))
    for m in range(len(pool)):
        t = time.time()
        for idx in tqdm(batches, desc=f"expert {m}", dynamic_ncols=True):
            vals = completion_nll(pool, m, [rows[i]["prompt"] for i in idx], [rows[i]["completion"] for i in idx], args.max_length)
            nll[idx, m] = torch.tensor(vals)
        log.info("expert %d (%s): mean nll %.4f in %.0fs", m, cfg.model.experts[m].name, nll[:, m].mean().item(), time.time() - t)

    best = nll.argmin(-1)
    log.info("best-expert counts: %s", torch.bincount(best, minlength=len(pool)).tolist())
    log.info("mean nll per expert: %s", [round(x, 4) for x in nll.mean(0).tolist()])
    out = Path(args.out or args.data)
    with open(out, "w") as f:
        for i, row in enumerate(rows):
            row["expert_nll"] = [round(x, 5) for x in nll[i].tolist()]
            row["expert_scores"] = [float(j == best[i].item()) for j in range(len(pool))]
            f.write(json.dumps(row) + "\n")
    log.info("wrote %s (%d rows)", out, len(rows))


if __name__ == "__main__":
    main()
