from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

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

    @property
    def lengths(self) -> list[int]:
        return [len(r["prompt"]) + len(r["completion"]) for r in self.rows]


def collate(rows: list[dict]) -> Batch:
    targets = None
    if EXPERT_SCORES in rows[0]:
        targets = torch.tensor([r[EXPERT_SCORES] for r in rows], dtype=torch.float32)
    return Batch([r["prompt"] for r in rows], [r["completion"] for r in rows], targets)


class LengthGroupedBatchSampler(Sampler[list[int]]):
    """Rank-sharded batch sampler. Samples are shuffled, then sorted by length inside mega-batches of
    `mega * batch_size` so each batch pads little; batch order is shuffled again. Every rank draws the same
    permutation from `seed + epoch` and takes a disjoint stride of batches, so no communication is needed."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        group_by_length: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        mega: int = 32,
    ):
        self.lengths = torch.tensor(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.group_by_length = group_by_length
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.mega = mega
        self.epoch = 0
        total = len(lengths) // batch_size
        self.num_batches = total // world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        n = len(self.lengths)
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        idx = torch.randperm(n, generator=g) if self.shuffle else torch.arange(n)
        if self.group_by_length:
            chunks = []
            for chunk in idx.split(self.mega * self.batch_size):
                order = self.lengths[chunk].argsort(descending=True)
                chunks.append(chunk[order])
            idx = torch.cat(chunks)
        batches = list(idx.split(self.batch_size))[: self.num_batches * self.world_size]
        if self.shuffle:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]
        for b in batches[self.rank :: self.world_size]:
            yield b.tolist()


def make_loader(
    path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    num_experts: int | None = None,
    group_by_length: bool = True,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = PromptCompletionDataset(path, num_experts)
    sampler = LengthGroupedBatchSampler(
        dataset.lengths, batch_size, shuffle, group_by_length, seed, rank, world_size
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
