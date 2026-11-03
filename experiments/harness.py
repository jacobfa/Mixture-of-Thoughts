"""Run the Avengers harness (harness_app.py -> baselines/avengers/app.py) over our split for one pool.

method            router + generator
expert:<name>     straight(name) + direct   one expert alone; on the train split these runs are the sample bank
random            random + direct
rank              Avengers router, one greedy sample (compute-matched to a single expert and to MoT)
avengers          Avengers router + self-consistency@10, T=0.7 (the published configuration)
routerdc          RouterDC + direct
zooter            Zooter + direct (zooter.py)
loraretriever     LoraRetriever selection + direct (loraretriever.py)
moa               Mixture-of-Agents: every expert proposes, one aggregates
mot               straight(mot) + direct; scripts/serve.py must be running
Results land in results/<pool>/<split>/<method>/<generator>/<task>-<timestamp>.json; existing results are skipped.
"""

import argparse
import copy
import os
import subprocess
import sys

import yaml

from common import DATA, EMBEDDING_URL, EXP, ID_TASKS, MOT, OOD_TASKS, RESULTS, harness_config, slug

METHODS = {
    "random": {"router.type": "random", "generator.type": "direct"},
    "rank": {"router.type": "rank", "generator.type": "direct"},
    "avengers": {"router.type": "rank", "generator.type": "self_consistency"},
    "routerdc": {"router.type": "routerdc", "generator.type": "direct"},
    "zooter": {"router.type": "zooter", "generator.type": "direct"},
    "loraretriever": {"router.type": "loraretriever", "generator.type": "direct"},
    "moa": {"router.type": "moa", "generator.type": "moa"},
    MOT: {"router.type": "straight", "router.straight_router.model": MOT, "generator.type": "direct"},
}
EVAL_METHODS = ["experts", "random", "rank", "avengers", "routerdc", "zooter", "loraretriever", "moa", MOT]


def overrides(method: str) -> dict:
    if method.startswith("expert:"):
        return {"router.type": "straight", "router.straight_router.model": method[7:], "generator.type": "direct"}
    return METHODS[method]


def apply(cfg: dict, dotted: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for path, value in dotted.items():
        node = cfg
        *parents, leaf = path.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--methods", nargs="+", help="default: experts on train, everything on test")
    parser.add_argument("--tasks", nargs="+", help="default: ID tasks on train, ID + OOD on test")
    parser.add_argument("--force", action="store_true", help="rerun (method, task) pairs that already have a result")
    args = parser.parse_args()

    base = harness_config(args.pool)
    experts = [f"expert:{e['name']}" for e in base["experts"] if e["name"] != MOT]
    methods = args.methods or (["experts"] if args.split == "train" else EVAL_METHODS)
    methods = [m for name in methods for m in (experts if name == "experts" else [name])]
    tasks = args.tasks or (list(ID_TASKS) if args.split == "train" else [*ID_TASKS, *sorted(OOD_TASKS)])

    out_root = RESULTS / args.pool / args.split
    (out_root / "configs").mkdir(parents=True, exist_ok=True)
    env = {
        "EMBEDDING_API_BASE_URL": EMBEDDING_URL,
        "EMBEDDING_API_KEY": "EMPTY",
        "EMBEDDING_CACHE_DIR": str(RESULTS / args.pool / "cache"),
        **os.environ,
    }
    for method in methods:
        save_dir = out_root / slug(method)
        for task in tasks:
            if not args.force and list(save_dir.glob(f"*/{task}-*.json")):
                print(f"skip {method} / {task}: done", flush=True)
                continue
            cfg = apply(base, {**overrides(method), "experiments.task": task, "experiments.cache_dir": str(RESULTS / args.pool / "cache")})
            cfg_path = out_root / "configs" / f"{slug(method)}-{task}.yaml"
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"run {method} / {task}", flush=True)
            cmd = [sys.executable, str(EXP / "harness_app.py"), "--config", str(cfg_path), "--save_dir", str(save_dir)]
            if subprocess.run(cmd, cwd=DATA / args.split, env=env, check=False).returncode != 0:
                print(f"FAILED {method} / {task}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
