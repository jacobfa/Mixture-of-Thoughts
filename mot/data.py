from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

EXPERT_SCORES = "expert_scores"


@dataclass
class Batch:
    prompts: list[str]
    completions: list[str]
    expert_targets: Tensor | None  # [B, M] per-expert quality labels, or None when the data has none


class PromptCompletionDataset(Dataset):
    """JSONL rows of {"prompt": str, "completion": str, "expert_scores": [M floats] (optional)}.

    `expert_scores` holds per-expert quality labels for the same prompt (e.g. 0/1 correctness of each
    expert's own answer, or a reward) and feeds the supervised routing loss. Either every row has it
    or none does."""

    def __init__(self, path: str | Path, num_experts: int | None = None):
        with open(path) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if not self.rows:
            raise ValueError(f"{path}: no rows")
        self.has_targets = EXPERT_SCORES in self.rows[0]
        for i, row in enumerate(self.rows):
            if "prompt" not in row or "completion" not in row:
                raise ValueError(f"{path}:{i}: rows need 'prompt' and 'completion' fields")
            if (EXPERT_SCORES in row) != self.has_targets:
                raise ValueError(f"{path}:{i}: '{EXPERT_SCORES}' must be present in all rows or none")
            if self.has_targets and num_experts is not None and len(row[EXPERT_SCORES]) != num_experts:
                raise ValueError(f"{path}:{i}: expected {num_experts} expert scores, got {len(row[EXPERT_SCORES])}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def collate(rows: list[dict]) -> Batch:
    targets = None
    if EXPERT_SCORES in rows[0]:
        targets = torch.tensor([r[EXPERT_SCORES] for r in rows], dtype=torch.float32)
    return Batch([r["prompt"] for r in rows], [r["completion"] for r in rows], targets)


def make_loader(
    path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    num_experts: int | None = None,
    drop_last: bool = True,
) -> DataLoader:
    return DataLoader(
        PromptCompletionDataset(path, num_experts),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=drop_last,
    )
