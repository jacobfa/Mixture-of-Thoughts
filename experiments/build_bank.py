"""Turn the single-expert runs on the train split into everything the routed methods are fitted on.

results/<pool>/bank.jsonl              {"query", "dataset", "cluster", "records": {expert: solved}}
results/<pool>/mot/train.jsonl         MoT rows: prompt, one correct expert's completion, 0/1 expert_scores
results/<pool>/rank/                   Avengers rank-router artifacts: l2-normalised gte embeddings, K-means centres,
                                       per-cluster accuracy rankings, M01.. mapping (same recipe as generate_rank_router.py)
results/<pool>/bank/{train,val}.json   router training files shared by RouterDC, Zooter and LoraRetriever:
                                       {"question", "dataset", "scores": {expert: 0/1}, "cluster_id"}
"""

import argparse
import json
import os
import random
from collections import defaultdict

import joblib
import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from common import EMBEDDING_URL, RESULTS, Tracker, expert_names, harness_config, slug


def load_runs(pool: str, experts: list[str]) -> dict[tuple[str, str], dict[str, tuple[bool, str]]]:
    table: dict[tuple[str, str], dict[str, tuple[bool, str]]] = defaultdict(dict)
    for name in experts:
        for path in (RESULTS / pool / "train" / slug(f"expert:{name}")).glob("*/*.json"):
            with open(path) as f:
                run = json.load(f)
            task = run["config"]["experiments"]["task"]
            for r in run["records"]:
                table[(task, r["query"])][name] = (bool(r["is_correct"]), r["raw_output"][0])
    complete = {k: v for k, v in table.items() if len(v) == len(experts)}
    print(f"{len(complete)} queries with all {len(experts)} experts ({len(table) - len(complete)} incomplete dropped)")
    return complete


def embed(texts: list[str], model: str, batch: int = 64) -> np.ndarray:
    client = OpenAI(base_url=os.environ.get("EMBEDDING_API_BASE_URL", EMBEDDING_URL), api_key="EMPTY")
    out = []
    for i in range(0, len(texts), batch):
        rsp = client.embeddings.create(input=texts[i : i + batch], model=model)
        out.extend(d.embedding for d in sorted(rsp.data, key=lambda d: d.index))
    return np.asarray(out, dtype=np.float32)


def write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.1, help="model-selection split of the router training files")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    experts = expert_names(args.pool)
    out = RESULTS / args.pool
    runs = load_runs(args.pool, experts)
    keys = sorted(runs)

    embedding_model = harness_config(args.pool)["router"]["rank_router"]["embedding_model"]
    normalizer = Normalizer(norm="l2")
    vectors = normalizer.fit_transform(embed([q for _, q in keys], embedding_model))
    kmeans = KMeans(args.clusters, init="k-means++", n_init="auto", max_iter=1000, algorithm="elkan", random_state=args.seed)
    labels = kmeans.fit_predict(vectors)

    ids = {name: f"M{i + 1:02d}" for i, name in enumerate(experts)}
    bank = [
        {"query": q, "dataset": task, "cluster": int(label), "records": {m: runs[(task, q)][m][0] for m in experts}}
        for (task, q), label in zip(keys, labels, strict=True)
    ]
    write_jsonl(out / "bank.jsonl", bank)

    mot_rows = []
    for task, q in keys:
        solved = [m for m in experts if runs[(task, q)][m][0]]
        if solved:
            completion = runs[(task, q)][rng.choice(solved)][1]
            mot_rows.append({"prompt": q, "completion": completion, "expert_scores": [float(m in solved) for m in experts]})
    write_jsonl(out / "mot" / "train.jsonl", mot_rows)
    print(f"mot/train.jsonl: {len(mot_rows)} rows ({len(keys) - len(mot_rows)} unsolved by every expert)")

    rankings = {}
    for c in range(args.clusters):
        members = [k for k, label in zip(keys, labels, strict=True) if label == c]
        scores = {}
        for m in experts:
            correct = sum(runs[k][m][0] for k in members)
            scores[ids[m]] = {"correct": correct, "total": len(members), "accuracy": correct / len(members) if members else 0}
        ranking = sorted(scores, key=lambda i: scores[i]["accuracy"], reverse=True)
        rankings[c] = {"total": len(members), "scores": scores, "ranking": ranking}
    rank_dir = out / "rank"
    rank_dir.mkdir(parents=True, exist_ok=True)
    np.save(rank_dir / "centres.npy", kmeans.cluster_centers_)
    joblib.dump(normalizer, rank_dir / "normalizer.joblib")
    with open(rank_dir / "rankings.json", "w") as f:
        json.dump(rankings, f, indent=2)
    with open(rank_dir / "mapping.json", "w") as f:
        json.dump({i: m for m, i in ids.items()}, f, indent=2)

    rows = [
        {
            "question": q,
            "dataset": task,
            "scores": {m: float(runs[(task, q)][m][0]) for m in experts},
            "cluster_id": int(label),
        }
        for (task, q), label in zip(keys, labels, strict=True)
    ]
    rng.shuffle(rows)
    n_val = round(len(rows) * args.val_size)
    (out / "bank").mkdir(parents=True, exist_ok=True)
    with open(out / "bank" / "val.json", "w") as f:
        json.dump(rows[:n_val], f, ensure_ascii=False)
    with open(out / "bank" / "train.json", "w") as f:
        json.dump(rows[n_val:], f, ensure_ascii=False)
    print(f"rank/: {args.clusters} clusters; bank/: {len(rows) - n_val} train / {n_val} val")

    tracker = Tracker(args.pool, "bank", "bank/build", config=vars(args), tags=["bank"])
    tracker.summary(
        {
            "queries": len(keys),
            "mot_rows": len(mot_rows),
            "train_rows": len(rows) - n_val,
            "val_rows": n_val,
            "oracle_accuracy": len(mot_rows) / len(keys),
            **{f"expert_accuracy/{m}": sum(runs[k][m][0] for k in keys) / len(keys) for m in experts},
        }
    )
    tracker.table(
        "clusters",
        ["cluster", "total", *experts],
        [[c, r["total"], *(r["scores"][ids[m]]["accuracy"] for m in experts)] for c, r in rankings.items()],
    )
    tracker.finish()


if __name__ == "__main__":
    main()
