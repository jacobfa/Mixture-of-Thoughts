"""Build ARC-AGI-1 prompt/completion JSONL for MoT.

Grids are rendered as rows of space-separated digits. A prompt lists the task's demonstration pairs followed by
the query input; the completion is the query output grid. Training examples come from the 400 training tasks
(every test pair plus leave-one-out over the demonstration pairs); evaluation examples are the test pairs of the
400 evaluation tasks. Examples longer than --max-tokens under any expert tokenizer are dropped.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from mot import load_config

Grid = list[list[int]]


def render(grid: Grid) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


def build_prompt(demos: list[dict], query: Grid) -> str:
    text = "".join(f"Input:\n{render(d['input'])}\nOutput:\n{render(d['output'])}\n\n" for d in demos)
    return text + f"Input:\n{render(query)}\nOutput:\n"


def parse_grid(text: str) -> Grid | None:
    rows = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            break
        cells = line.split()
        if not all(c.isdigit() and len(c) == 1 for c in cells):
            return None
        rows.append([int(c) for c in cells])
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        return None
    return rows


def transform(grid: Grid, k: int, perm: list[int] | None) -> Grid:
    arr = np.array(grid)
    arr = np.rot90(arr, k % 4)
    if k >= 4:
        arr = np.fliplr(arr)
    if perm is not None:
        arr = np.array(perm)[arr]
    return arr.tolist()


def inverse_transform(grid: Grid, k: int) -> Grid:
    arr = np.array(grid)
    if k >= 4:
        arr = np.fliplr(arr)
    return np.rot90(arr, -(k % 4)).tolist()


def transform_pairs(pairs: list[dict], k: int, perm: list[int] | None) -> list[dict]:
    return [{"input": transform(p["input"], k, perm), "output": transform(p["output"], k, perm)} for p in pairs]


def parse_prompt(prompt: str) -> tuple[list[dict], Grid]:
    """Inverse of `build_prompt`: demonstration pairs and the query grid."""
    blocks = [b for b in prompt.split("\n\n") if b.strip()]
    demos = []
    for block in blocks[:-1]:
        inp, out = block.split("\nOutput:\n")
        demos.append({"input": parse_grid(inp.removeprefix("Input:\n")), "output": parse_grid(out)})
    query = parse_grid(blocks[-1].removeprefix("Input:\n").removesuffix("\nOutput:\n"))
    return demos, query


def task_examples(task_id: str, task: dict, leave_one_out: bool) -> list[dict]:
    train, test = task["train"], task["test"]
    out = []
    for i, pair in enumerate(test):
        out.append({"prompt": build_prompt(train, pair["input"]), "completion": render(pair["output"]), "task": task_id, "pair": f"test{i}"})
    if leave_one_out and len(train) > 1:
        for i, pair in enumerate(train):
            demos = train[:i] + train[i + 1 :]
            out.append({"prompt": build_prompt(demos, pair["input"]), "completion": render(pair["output"]), "task": task_id, "pair": f"train{i}"})
    return out


def augmented(task: dict, rng: random.Random, n: int) -> list[dict]:
    variants = []
    for _ in range(n):
        k = rng.randrange(8)
        perm = [0, *rng.sample(range(1, 10), 9)] if rng.random() < 0.5 else None
        variants.append({"train": transform_pairs(task["train"], k, perm), "test": transform_pairs(task["test"], k, perm)})
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arc-dir", required=True, help="ARC-AGI repo data/ directory (training/, evaluation/)")
    parser.add_argument("--config", required=True, help="MoT config; expert tokenizers decide the length filter")
    parser.add_argument("--out", default="data/arc")
    parser.add_argument("--max-tokens", type=int, default=8192, help="training length cap (prompt + completion)")
    parser.add_argument("--eval-max-tokens", type=int, default=16384)
    parser.add_argument("--augment", type=int, default=0, help="extra dihedral/colour-permuted copies per training task")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizers = [AutoTokenizer.from_pretrained(e.tokenizer or e.name) for e in cfg.model.experts]
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load(split: str) -> list[tuple[str, dict]]:
        files = sorted(Path(args.arc_dir, split).glob("*.json"))
        return [(f.stem, json.load(open(f))) for f in files]

    def write(name: str, rows: list[dict], max_tokens: int) -> None:
        kept, lengths = [], []
        for row in rows:
            n = max(len(t(row["prompt"] + row["completion"])["input_ids"]) for t in tokenizers)
            if n + 1 <= max_tokens:
                row["tokens"] = n
                kept.append(row)
                lengths.append(n)
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for row in kept:
                f.write(json.dumps(row) + "\n")
        q = np.percentile(lengths, [50, 90, 99, 100]).astype(int).tolist() if lengths else []
        print(f"{name}: kept {len(kept)}/{len(rows)} examples (max tokens over experts p50/p90/p99/max = {q})")

    train_rows = []
    for task_id, task in load("training"):
        train_rows += task_examples(task_id, task, leave_one_out=True)
        for j, variant in enumerate(augmented(task, rng, args.augment)):
            train_rows += task_examples(f"{task_id}#aug{j}", variant, leave_one_out=False)
    write("train", train_rows, args.max_tokens)
    eval_rows = [ex for task_id, task in load("evaluation") for ex in task_examples(task_id, task, leave_one_out=False)]
    write("eval", eval_rows, args.eval_max_tokens)


if __name__ == "__main__":
    main()
