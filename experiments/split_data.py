"""Download the Avengers benchmark data and write the deterministic 70/30 split every method is run on.

data/train/data/<task>  calibration split: expert sample bank, MoT and RouterDC training, Avengers clustering
data/test/data/<task>   held-out 30% of each in-distribution task plus the whole OOD tasks
"""

import argparse
import json
import random
from pathlib import Path

from common import DATA, ID_TASKS, OOD_FILES


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    full = Path(snapshot_download("Estwld/Avengers", repo_type="dataset", local_dir=DATA / "full"))
    for name in ID_TASKS.values():
        files = sorted(p for p in (full / name).rglob("*") if p.suffix in (".json", ".jsonl"))
        for src in files:
            rows = read_rows(src)
            random.Random(args.seed).shuffle(rows)
            n_test = round(len(rows) * args.test_size)
            rel = src.relative_to(full)
            write_rows(DATA / "test" / "data" / rel, rows[:n_test])
            write_rows(DATA / "train" / "data" / rel, rows[n_test:])
            print(f"{rel}: {len(rows) - n_test} train / {n_test} test")
    for src, (dst, keep) in OOD_FILES.items():
        rows = [r for r in read_rows(full / src) if keep is None or keep(r)]
        write_rows(DATA / "test" / "data" / dst, rows)
        print(f"{dst}: {len(rows)} test (OOD)")


if __name__ == "__main__":
    main()
