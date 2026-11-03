"""Detailed comparison over results/<pool>/test -> results/<pool>/summary.md and summary.json.

Accuracy per task with standard errors, ID/OOD/overall means, wins and mean task rank; best single expert and oracle
rows; token and latency cost; expert-selection distributions and oracle recovery of the routers; query-level paired
comparison of every method against MoT (McNemar); calibration-bank statistics. The accuracy table is also printed.
"""

import argparse
import json
import math
import time
from collections import Counter, defaultdict

from common import ID_TASKS, MOT, OOD_TASKS, RESULTS, expert_names

ORDER = ["random", "rank", "avengers", "routerdc", "zooter", "loraretriever", "moa", MOT]
ROUTED = ["random", "rank", "avengers", "routerdc", "zooter", "loraretriever"]
LABELS = {
    "random": "Random",
    "rank": "Avengers (rank, 1 sample)",
    "avengers": "Avengers (rank + SC@10)",
    "routerdc": "RouterDC",
    "zooter": "Zooter",
    "loraretriever": "LoraRetriever",
    "moa": "MoA",
    MOT: "MoT",
    "best_expert": "Best single expert (per task)",
    "oracle": "Oracle (any expert)",
}
FAILED = "failed to generate"


def label(method: str) -> str:
    return LABELS.get(method, method)


def pct(x, digits: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}"


def num(x, digits: int = 1) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def table(header: list, rows: list[list]) -> str:
    lines = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def method_of(path) -> str:
    slug = path.parts[-3]
    return slug[7:].replace("__", "/") if slug.startswith("expert-") else slug


def load_runs(pool: str) -> dict[tuple[str, str], dict]:
    runs = {}
    for path in sorted((RESULTS / pool / "test").glob("*/*/*.json"), key=lambda p: p.stat().st_mtime):
        with open(path) as f:
            run = json.load(f)
        runs[(method_of(path), run["config"]["experiments"]["task"])] = run
    return runs


def generations_per_query(cfg: dict) -> int:
    gen = cfg["generator"]
    if gen["type"] == "self_consistency":
        return gen["self_consistency"]["samples"]
    if gen["type"] == "moa":
        moa = gen["moa"]
        return moa["number_of_proposers"] * moa["layers"] + moa["number_of_aggregator"]
    return 1


def first_model(name) -> str:
    return str(name[0] if isinstance(name, list) and name else name)


def run_stats(run: dict) -> dict:
    records = run["records"]
    n = len(records)
    p = run["performance"]
    return {
        "n": n,
        "accuracy": p,
        "stderr": math.sqrt(p * (1 - p) / n) if n else None,
        "prompt_tokens_per_query": run["prompt_tokens"] / n,
        "completion_tokens_per_query": run["completion_tokens"] / n,
        "seconds_per_query": run["time_taken"] / n,
        "generations_per_query": generations_per_query(run["config"]),
        "failed": sum(any(o == FAILED for o in r["raw_output"]) for r in records),
        "routing": dict(Counter(first_model(r["model_name"]) for r in records)),
        "solved": {r["query"]: bool(r["is_correct"]) for r in records},
    }


def mean_over(values: dict[str, float], subset: list[str]) -> tuple[float | None, bool]:
    present = [values[t] for t in subset if t in values]
    return (sum(present) / len(present) if present else None), len(present) == len(subset)


def mcnemar(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def rank_rows(scores: dict[str, float]) -> dict[str, float]:
    return {
        m: 1 + sum(v > s for v in scores.values()) + (sum(v == s for v in scores.values()) - 1) / 2
        for m, s in scores.items()
    }


def bank_stats(pool: str, experts: list[str]) -> dict | None:
    path = RESULTS / pool / "bank.jsonl"
    if not path.exists():
        return None
    per_dataset: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            per_dataset[row["dataset"]].append(row["records"])
    out = {}
    for dataset, rows in sorted(per_dataset.items()):
        out[dataset] = {
            "n": len(rows),
            **{m: sum(r[m] for r in rows) / len(rows) for m in experts},
            "oracle": sum(any(r.values()) for r in rows) / len(rows),
            "all_solve": sum(all(r.values()) for r in rows) / len(rows),
        }
    rows = [r for group in per_dataset.values() for r in group]
    out["all"] = {
        "n": len(rows),
        **{m: sum(r[m] for r in rows) / len(rows) for m in experts},
        "oracle": sum(any(r.values()) for r in rows) / len(rows),
        "all_solve": sum(all(r.values()) for r in rows) / len(rows),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    args = parser.parse_args()
    experts = expert_names(args.pool)
    runs = load_runs(args.pool)
    stats = {key: run_stats(run) for key, run in runs.items()}

    all_tasks = [*ID_TASKS, *sorted(OOD_TASKS)]
    tasks = [t for t in all_tasks if any(k[1] == t for k in runs)]
    id_tasks = [t for t in ID_TASKS if t in tasks]
    ood_tasks = [t for t in sorted(OOD_TASKS) if t in tasks]
    methods = [m for m in [*experts, *ORDER] if any(k[0] == m for k in runs)]
    systems = [m for m in ORDER if m in methods]

    acc: dict[str, dict[str, float]] = {
        m: {t: stats[(m, t)]["accuracy"] for t in tasks if (m, t) in stats} for m in methods
    }
    n_per_task = {t: next(stats[(m, t)]["n"] for m in methods if (m, t) in stats) for t in tasks}
    for t in tasks:
        present = [acc[m][t] for m in experts if t in acc.get(m, {})]
        if present:
            acc.setdefault("best_expert", {})[t] = max(present)
        solved = defaultdict(bool)
        for m in experts:
            for q, ok in stats.get((m, t), {}).get("solved", {}).items():
                solved[q] |= ok
        if solved:
            acc.setdefault("oracle", {})[t] = sum(solved.values()) / len(solved)
    rows = [*methods, *(r for r in ["best_expert", "oracle"] if r in acc)]

    wins = Counter()
    ranks: dict[str, list[float]] = defaultdict(list)
    for t in tasks:
        scores = {m: acc[m][t] for m in systems if t in acc[m]}
        if not scores:
            continue
        for m in [m for m, s in scores.items() if s == max(scores.values())]:
            wins[m] += 1
        for m, r in rank_rows(scores).items():
            ranks[m].append(r)

    summary = {
        "pool": args.pool,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": tasks,
        "n_per_task": n_per_task,
        "experts": experts,
        "methods": {},
        "pairwise_vs_mot": {},
        "bank": bank_stats(args.pool, experts),
    }
    for m in rows:
        id_mean, id_ok = mean_over(acc[m], id_tasks)
        ood_mean, ood_ok = mean_over(acc[m], ood_tasks)
        all_mean, all_ok = mean_over(acc[m], tasks)
        entry = {
            "label": label(m),
            "accuracy": acc[m],
            "means": {"id": id_mean, "ood": ood_mean, "all": all_mean},
            "complete": {"id": id_ok, "ood": ood_ok, "all": all_ok},
        }
        if m in methods:
            per_task = {t: stats[(m, t)] for t in tasks if (m, t) in stats}
            total_n = sum(s["n"] for s in per_task.values())
            entry["stderr"] = {t: s["stderr"] for t, s in per_task.items()}
            entry["cost"] = {
                "tasks": len(per_task),
                "queries": total_n,
                "prompt_tokens_per_query": sum(s["prompt_tokens_per_query"] * s["n"] for s in per_task.values())
                / total_n,
                "completion_tokens_per_query": sum(s["completion_tokens_per_query"] * s["n"] for s in per_task.values())
                / total_n,
                "seconds_per_query": sum(s["seconds_per_query"] * s["n"] for s in per_task.values()) / total_n,
                "generations_per_query": max(s["generations_per_query"] for s in per_task.values()),
                "failed_generations": sum(s["failed"] for s in per_task.values()),
            }
            entry["completion_tokens_per_query"] = {t: s["completion_tokens_per_query"] for t, s in per_task.items()}
            if m in ROUTED:
                entry["routing"] = {
                    t: {e: s["routing"].get(e, 0) / s["n"] for e in experts} for t, s in per_task.items()
                }
                entry["oracle_recovery"] = {
                    t: acc[m][t] / acc["oracle"][t] for t in per_task if acc.get("oracle", {}).get(t)
                }
        if m in systems:
            entry["wins"] = wins[m]
            entry["mean_rank"] = sum(ranks[m]) / len(ranks[m]) if ranks[m] else None
        summary["methods"][m] = entry

    if MOT in methods:
        for m in [m for m in methods if m != MOT]:
            shared = [t for t in tasks if (m, t) in stats and (MOT, t) in stats]
            if not shared:
                continue
            deltas = {t: acc[MOT][t] - acc[m][t] for t in shared}
            b = c = 0
            for t in shared:
                mine, theirs = stats[(MOT, t)]["solved"], stats[(m, t)]["solved"]
                for q in mine.keys() & theirs.keys():
                    b += mine[q] and not theirs[q]
                    c += theirs[q] and not mine[q]
            summary["pairwise_vs_mot"][m] = {
                "tasks": len(shared),
                "mot_better": sum(d > 0 for d in deltas.values()),
                "mot_worse": sum(d < 0 for d in deltas.values()),
                "tie": sum(d == 0 for d in deltas.values()),
                "mean_delta": sum(deltas.values()) / len(deltas),
                "delta": deltas,
                "only_mot_solves": b,
                "only_other_solves": c,
                "mcnemar_p": mcnemar(b, c),
            }

    md = [
        f"# {args.pool}: Avengers-protocol comparison",
        "",
        f"Generated {summary['generated']} from {len(runs)} result files in `results/{args.pool}/test`.",
    ]
    md += [
        "Accuracy in %, with binomial standard error in parentheses. `*` marks a mean over an incomplete task set.",
        "",
    ]

    md += ["## Accuracy", ""]
    header = ["method", *(f"{t} (n={n_per_task[t]})" for t in tasks), "ID", "OOD", "All", "wins", "mean rank"]
    body = []
    for m in rows:
        e = summary["methods"][m]
        cells = [
            f"{pct(acc[m][t])} ({pct(e['stderr'][t])})" if t in acc[m] and e.get("stderr") else pct(acc[m].get(t))
            for t in tasks
        ]
        means = [
            pct(e["means"][k]) + ("*" if e["means"][k] is not None and not e["complete"][k] else "")
            for k in ("id", "ood", "all")
        ]
        extra = [e.get("wins", "-"), num(e.get("mean_rank"), 2) if m in systems else "-"]
        body.append([label(m), *cells, *means, *extra])
    md += [table(header, body), ""]

    md += ["## Cost (query-weighted means over the tasks each method ran)", ""]
    header = [
        "method",
        "tasks",
        "queries",
        "generations/query",
        "prompt tokens/query",
        "completion tokens/query",
        "seconds/query",
        "failed generations",
    ]
    body = []
    for m in methods:
        c = summary["methods"][m]["cost"]
        body.append(
            [
                label(m),
                c["tasks"],
                c["queries"],
                c["generations_per_query"],
                num(c["prompt_tokens_per_query"], 0),
                num(c["completion_tokens_per_query"], 0),
                num(c["seconds_per_query"], 2),
                c["failed_generations"],
            ]
        )
    md += [table(header, body), ""]
    md += ["### Completion tokens per query by task", ""]
    md += [
        table(
            ["method", *tasks],
            [
                [label(m), *(num(summary["methods"][m]["completion_tokens_per_query"].get(t), 0) for t in tasks)]
                for m in methods
            ],
        ),
        "",
    ]

    md += ["## Routing", "", "Share of queries sent to each expert; oracle recovery = accuracy / oracle accuracy.", ""]
    for m in [m for m in ROUTED if m in methods]:
        e = summary["methods"][m]
        md += [f"### {label(m)}", ""]
        header = ["task", *experts, "accuracy", "oracle recovery"]
        body = [
            [t, *(pct(e["routing"][t][x]) for x in experts), pct(acc[m][t]), pct(e["oracle_recovery"].get(t))]
            for t in e["routing"]
        ]
        md += [table(header, body), ""]

    if summary["pairwise_vs_mot"]:
        md += [
            "## MoT vs. each method",
            "",
            "Task counts by sign of (MoT - method); query-level counts over shared queries; two-sided exact McNemar p.",
            "",
        ]
        header = [
            "method",
            "tasks",
            "MoT better",
            "MoT worse",
            "tie",
            "mean delta (pts)",
            "only MoT solves",
            "only method solves",
            "McNemar p",
        ]
        body = []
        for m, p in summary["pairwise_vs_mot"].items():
            body.append(
                [
                    label(m),
                    p["tasks"],
                    p["mot_better"],
                    p["mot_worse"],
                    p["tie"],
                    num(100 * p["mean_delta"], 1),
                    p["only_mot_solves"],
                    p["only_other_solves"],
                    "-" if p["mcnemar_p"] is None else f"{p['mcnemar_p']:.3g}",
                ]
            )
        md += [table(header, body), ""]
        md += ["### Per-task delta (MoT - method, pts)", ""]
        md += [
            table(
                ["method", *tasks],
                [
                    [label(m), *(num(100 * p["delta"][t]) if t in p["delta"] else "-" for t in tasks)]
                    for m, p in summary["pairwise_vs_mot"].items()
                ],
            ),
            "",
        ]

    if summary["bank"]:
        md += ["## Calibration bank (single-expert runs on the train split)", ""]
        header = ["dataset", "n", *experts, "oracle", "all experts solve"]
        body = [
            [d, s["n"], *(pct(s[x]) for x in experts), pct(s["oracle"]), pct(s["all_solve"])]
            for d, s in summary["bank"].items()
        ]
        md += [table(header, body), ""]

    out = RESULTS / args.pool
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(md[md.index("## Accuracy") + 2])
    print(f"\nwrote {out / 'summary.md'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
