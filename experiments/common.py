import json
import os
from pathlib import Path
from typing import Any

import yaml

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent
AVENGERS = EXP / "baselines" / "avengers"
ROUTERDC = EXP / "baselines" / "routerdc"
DATA = EXP / "data"
RESULTS = EXP / "results"

# harness task id -> data directory under data/ (in-distribution tasks are split 70/30)
ID_TASKS = {
    "aime": "AIME",
    "math500": "MATH500",
    "livemathbench": "livemathbench",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "kandk": "K_and_K",
    "bbh": "bbh",
    "arcc": "arc_c",
    "mmlupro": "MMLUPro",
    "gpqa": "GPQA",
    "finqa": "FinQA",
    "medqa": "MedQA",
    "emorynlp": "EmoryNLP",
}
# out-of-distribution tasks are evaluated whole: source file in the HF dataset -> (harness path, row filter)
OOD_TASKS = {"mathbench", "winogrande"}
OOD_FILES = {
    "winogrande/valid.json": ("winogrande/valid.json", None),
    "mathbench_v1/single_choice.json": ("MathBench/college.jsonl", lambda row: row.get("category") == "college"),
}

MOT = "mot"
EMBEDDING_URL = "http://localhost:8100/v1"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def harness_config(pool: str) -> dict:
    return load_yaml(EXP / "configs" / pool / "avengers.yaml")


def mot_config(pool: str) -> dict:
    return load_yaml(EXP / "configs" / pool / "mot.yaml")


def expert_names(pool: str) -> list[str]:
    """Served model names of the pool's experts (HF ids), in MoT expert order."""
    return [e["name"] for e in mot_config(pool)["model"]["experts"]]


def load_bank(pool: str, split: str, experts: list[str]) -> tuple[list[str], list[float], list[dict]]:
    """Router training rows written by build_bank.py: questions, 0/1 rewards in `experts` order, raw rows."""
    with open(RESULTS / pool / "bank" / f"{split}.json") as f:
        rows = json.load(f)
    return [r["question"] for r in rows], [[r["scores"][m] for m in experts] for r in rows], rows


def port_of(base_url: str) -> int:
    return int(base_url.rsplit(":", 1)[1].split("/")[0])


def slug(method: str) -> str:
    return method.replace("/", "__").replace(":", "-")


class Tracker:
    """wandb run configured by mot.yaml's train.wandb (project, entity, mode); WANDB_MODE overrides the mode and
    `disabled` makes every call a no-op. Online without an API key falls back to offline."""

    def __init__(self, pool: str, job_type: str, name: str, config: dict | None = None, tags: list[str] | None = None):
        self.run = None
        cfg = mot_config(pool)["train"].get("wandb", {})
        mode = os.environ.get("WANDB_MODE", cfg.get("mode", "online"))
        if mode == "disabled":
            return
        import wandb

        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            try:
                has_key = bool(wandb.api.api_key)
            except Exception:
                has_key = False
            if not has_key:
                print("no wandb API key found; logging offline (sync later with `wandb sync`)", flush=True)
                mode = "offline"
        self.run = wandb.init(
            project=cfg.get("project", "mot-avengers"),
            entity=cfg.get("entity"),
            group=pool,
            job_type=job_type,
            name=name,
            mode=mode,
            tags=tags,
            config={"pool": pool, **(config or {})},
        )

    def log(self, record: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(record, step=step)

    def summary(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.summary.update(values)

    def table(self, key: str, columns: list[str], rows: list[list]) -> None:
        if self.run is not None:
            import wandb

            self.run.log({key: wandb.Table(columns=columns, data=rows)})

    def save(self, path: Path) -> None:
        if self.run is not None:
            self.run.save(str(path), base_path=str(path.parent), policy="now")

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
