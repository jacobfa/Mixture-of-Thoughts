"""Accuracy table over results/<pool>/test: one row per method, one column per task, ID and OOD means, plus an
oracle row (a query counts as solved when any single expert solved it)."""

import argparse
import json
from collections import defaultdict

from common import ID_TASKS, MOT, OOD_TASKS, RESULTS, expert_names

ORDER = ["random", "rank", "avengers", "routerdc", "moa", MOT, "oracle"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    args = parser.parse_args()
    experts = expert_names(args.pool)

    acc: dict[str, dict[str, float]] = defaultdict(dict)
    solved: dict[str, dict[str, bool]] = defaultdict(dict)
    for path in (RESULTS / args.pool / "test").glob("*/*/*.json"):
        with open(path) as f:
            run = json.load(f)
        method = path.parts[-3]
        task = run["config"]["experiments"]["task"]
        if method.startswith("expert-"):
            method = method[7:].replace("__", "/")
            for r in run["records"]:
                solved[task][r["query"]] = solved[task].get(r["query"], False) or bool(r["is_correct"])
        acc[method][task] = run["performance"]
    for task, rows in solved.items():
        acc["oracle"][task] = sum(rows.values()) / len(rows)

    tasks = [t for t in [*ID_TASKS, *sorted(OOD_TASKS)] if any(t in a for a in acc.values())]
    methods = [m for m in [*experts, *ORDER] if m in acc]

    def mean(row: dict[str, float], subset) -> str:
        vals = [row[t] for t in subset if t in row]
        return f"{100 * sum(vals) / len(vals):.1f}" if len(vals) == len([t for t in subset if t in tasks]) else "-"

    header = ["method", *tasks, "ID", "OOD"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for m in methods:
        cells = [f"{100 * acc[m][t]:.1f}" if t in acc[m] else "-" for t in tasks]
        lines.append(f"| {m} | " + " | ".join([*cells, mean(acc[m], ID_TASKS), mean(acc[m], OOD_TASKS)]) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
