import json
from pathlib import Path

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
