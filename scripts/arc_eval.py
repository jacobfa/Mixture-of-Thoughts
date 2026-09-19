"""Evaluate on the ARC-AGI-1 public evaluation tasks (data/arc/eval.jsonl from scripts/arc_data.py).

Protocol follows the ARC Prize scheme: up to two attempts per test input, a prediction counts when the parsed output
grid equals the target exactly. Attempt 1 decodes the task as given; attempt 2 decodes a dihedral-transformed copy
of the task (--attempt-transform) and maps the prediction back. Reported metrics:

  pass1_pair / pass2_pair   fraction of test pairs solved with 1 / 2 attempts
  pass1_task / pass2_task   fraction of tasks with every test pair solved
  arc_score_pass1 / pass2   official-style score: mean over tasks of the fraction of test pairs solved
  *_all                     the same over all 400 tasks / 419 pairs (pairs not evaluated count as failures)

`--expert m` evaluates expert m alone with plain HF generation (baseline); otherwise the MoT model is used, from
`--checkpoint` when given.
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from mot import MixtureOfThoughts, load_config
from mot.config import GenerationConfig
from mot.experts import ExpertPool
from mot.tracking import WandbRun, config_dict, configure_logging
from mot.training import fmt_duration

sys.path.insert(0, str(Path(__file__).parent))
from arc_data import build_prompt, inverse_transform, parse_grid, parse_prompt, transform, transform_pairs  # noqa: E402

log = logging.getLogger("arc_eval")
ARC_EVAL_TASKS, ARC_EVAL_PAIRS = 400, 419
STOP = "\n\n"


def output_budget(prompt: str) -> int:
    """Token budget for the answer: the largest grid in the prompt, two tokens per cell plus newlines."""
    cells = rows = 1
    for block in re.findall(r"(?:Input|Output):\n((?:[0-9 ]+\n?)+)", prompt):
        lines = [ln for ln in block.strip().split("\n") if ln]
        cells = max(cells, sum(len(ln.split()) for ln in lines))
        rows = max(rows, len(lines))
    return 2 * cells + 2 * rows + 16


@torch.no_grad()
def generate_mot(model: MixtureOfThoughts, prompts: list[str], gen: GenerationConfig) -> tuple[list[str], list[list[int]], int]:
    out = model.generate(prompts, gen)
    return out.texts, out.routing.active.tolist(), out.steps


@torch.no_grad()
def generate_expert(pool: ExpertPool, prompts: list[str], max_new_tokens: int, max_prompt_length: int) -> list[str]:
    expert = pool[0]
    tok = expert.tokenizer
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length).to(expert.device)
    out = expert.model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=expert.pad_token_id, stop_strings=[STOP], tokenizer=tok
    )
    return [t.split(STOP)[0] for t in tok.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)]


def make_batches(order: list[int], tokens: list[int], bs: int, token_budget: int) -> list[list[int]]:
    batches: list[list[int]] = []
    for i in order:
        if batches and len(batches[-1]) < bs and sum(tokens[j] for j in batches[-1]) + tokens[i] <= token_budget:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/arc/eval.jsonl")
    parser.add_argument("--checkpoint", help="MixtureOfThoughts.save directory; omit for untrained MoT")
    parser.add_argument("--expert", type=int, help="evaluate this expert alone (plain HF generate)")
    parser.add_argument("--device", help="device for the single expert (default from config)")
    parser.add_argument("--name", help="run name (log file, wandb, output file)")
    parser.add_argument("--attempts", type=int, default=2, choices=(1, 2))
    parser.add_argument("--attempt-transform", type=int, default=1, help="dihedral transform id (1-7) used for attempt 2")
    parser.add_argument("--batch-size", type=int, help="max prompts per batch (default generation.batch_size)")
    parser.add_argument("--tokens-per-batch", type=int, default=30000, help="prompt-token budget per batch")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--wandb-mode", help="override train.wandb.mode")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    if args.expert is not None:
        default_name = f"eval_expert{args.expert}_{Path(cfg.model.experts[args.expert].name).name}"
    else:
        default_name = "eval_mot_" + (Path(args.checkpoint).parent.name + "_" + Path(args.checkpoint).name if args.checkpoint else "untrained")
    name = args.name or default_name
    configure_logging(f"logs/{name}.log")
    torch.set_float32_matmul_precision("high")
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    log.info("%s: %d eval pairs from %s (%d tasks), %d attempt(s)", name, len(rows), args.data, len({r["task"] for r in rows}), args.attempts)

    gen = cfg.generation
    bs = args.batch_size or gen.batch_size
    wandb_cfg = cfg.train.wandb
    if args.wandb_mode:
        wandb_cfg.mode = args.wandb_mode
    run = WandbRun(
        wandb_cfg,
        {"model": config_dict(cfg.model), "generation": config_dict(gen), "checkpoint": args.checkpoint, "expert": args.expert, "attempts": args.attempts},
        job_type="eval",
        name=name,
    )

    if args.expert is not None:
        model_cfg = cfg.model
        model_cfg.experts = [model_cfg.experts[args.expert]]
        if args.device:
            model_cfg.experts[0].device = args.device
        model_cfg.top_k = 1
        pool = ExpertPool.from_config(model_cfg)
        pool.train(False)
        generate = lambda prompts, budget: (generate_expert(pool, prompts, budget, gen.max_prompt_length), None, 0)
    else:
        model = MixtureOfThoughts.load(args.checkpoint, cfg.model) if args.checkpoint else MixtureOfThoughts(cfg.model)
        model.eval()
        generate = lambda prompts, budget: generate_mot(
            model, prompts, GenerationConfig(**{**config_dict(gen), "max_new_tokens": budget, "do_sample": False, "stop": STOP})
        )

    # attempt a: prompts of the (transformed) task; predictions are mapped back before scoring
    transforms = [0, args.attempt_transform][: args.attempts]
    prompts_by_attempt = []
    for k in transforms:
        if k == 0:
            prompts_by_attempt.append([r["prompt"] for r in rows])
            continue
        attempt_prompts = []
        for r in rows:
            demos, query = parse_prompt(r["prompt"])
            attempt_prompts.append(build_prompt(transform_pairs(demos, k, None), transform(query, k, None)))
        prompts_by_attempt.append(attempt_prompts)
    targets = [parse_grid(r["completion"]) for r in rows]
    tokens = [r.get("tokens", len(r["prompt"]) // 2) for r in rows]
    budgets = [min(gen.max_new_tokens, output_budget(r["prompt"])) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: tokens[i], reverse=True)
    batches = make_batches(order, tokens, bs, args.tokens_per_batch)
    work = [max(budgets[i] for i in idx) for idx in batches]
    total_work = sum(work) * len(transforms)
    log.info("%d batches per attempt (<= %d prompts, <= %d prompt tokens), <= %d decode steps in total", len(batches), bs, args.tokens_per_batch, total_work)

    results = [{"task": r["task"], "pair": r["pair"], "attempts": []} for r in rows]
    routing = Counter()
    t0 = time.time()
    done_work = 0
    for a, k in enumerate(transforms):
        for bi, idx in enumerate(tqdm(batches, desc=f"attempt {a + 1}", dynamic_ncols=True, disable=not sys.stderr.isatty())):
            t_batch = time.time()
            texts, active, steps = generate([prompts_by_attempt[a][i] for i in idx], work[bi])
            for j, i in enumerate(idx):
                pred = parse_grid(texts[j])
                if pred is not None and k:
                    pred = inverse_transform(pred, k)
                results[i]["attempts"].append({"correct": pred is not None and pred == targets[i], "parsed": pred is not None, "prediction": texts[j].strip()[:4000], "experts": active[j] if active else None})
                if active and a == 0:
                    routing[active[j][0]] += 1
            done_work += work[bi]
            scored = [r for r in results if r["attempts"]]
            solved = sum(any(x["correct"] for x in r["attempts"]) for r in scored)
            elapsed = time.time() - t0
            eta = elapsed * (total_work - done_work) / max(done_work, 1)
            log.info(
                "attempt %d batch %3d/%d: %d prompts, <=%d prompt tok, %d/%d steps, %d/%d correct, %.0fs | solved so far %d/%d pairs (%.3f) | elapsed %s eta %s (~%s)",
                a + 1, bi + 1, len(batches), len(idx), max(tokens[i] for i in idx), steps or work[bi], work[bi],
                sum(results[i]["attempts"][-1]["correct"] for i in idx), len(idx), time.time() - t_batch,
                solved, len(scored), solved / max(len(scored), 1), fmt_duration(elapsed), fmt_duration(eta),
                time.strftime("%H:%M", time.localtime(time.time() + eta)),
            )

    out_path = Path("runs/arc_eval") / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    per_task: dict[str, list[list[bool]]] = defaultdict(list)
    for r in results:
        per_task[r["task"]].append([x["correct"] for x in r["attempts"]])
    n_pairs, n_tasks = len(results), len(per_task)

    def metrics(n_attempts: int) -> dict:
        solved = lambda attempts: any(attempts[:n_attempts])
        pairs_correct = sum(solved(r) for rs in per_task.values() for r in rs)
        tasks_correct = sum(all(solved(r) for r in rs) for rs in per_task.values())
        task_fraction = sum(sum(solved(r) for r in rs) / len(rs) for rs in per_task.values())
        return {
            f"pass{n_attempts}_pair": pairs_correct / n_pairs,
            f"pass{n_attempts}_task": tasks_correct / n_tasks,
            f"arc_score_pass{n_attempts}": task_fraction / n_tasks,
            f"pass{n_attempts}_pair_all": pairs_correct / ARC_EVAL_PAIRS,
            f"pass{n_attempts}_task_all": tasks_correct / ARC_EVAL_TASKS,
            f"arc_score_pass{n_attempts}_all": task_fraction / ARC_EVAL_TASKS,
            f"pairs_correct_pass{n_attempts}": pairs_correct,
            f"tasks_correct_pass{n_attempts}": tasks_correct,
        }

    summary = {"pairs_evaluated": n_pairs, "tasks_evaluated": n_tasks, "seconds": round(time.time() - t0)}
    for n in range(1, len(transforms) + 1):
        summary.update(metrics(n))
    summary["parse_rate"] = sum(x["parsed"] for r in results for x in r["attempts"]) / (n_pairs * len(transforms))
    if len(transforms) > 1:
        summary["attempt2_only_pair"] = sum((not r["attempts"][0]["correct"]) and r["attempts"][1]["correct"] for r in results) / n_pairs
        summary["attempt2_pair"] = sum(r["attempts"][1]["correct"] for r in results) / n_pairs
    if routing:
        summary["primary_routing"] = {str(k): v for k, v in sorted(routing.items())}
    log.info("summary %s:\n%s", name, json.dumps(summary, indent=2))
    log.info("predictions written to %s", out_path)
    run.summary(summary)
    run.finish()


if __name__ == "__main__":
    main()
