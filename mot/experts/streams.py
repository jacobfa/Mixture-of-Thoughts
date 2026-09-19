from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from mot.experts.pool import ExpertPool

IGNORE_INDEX = -100
# Alignment coordinate of padding / dead tokens; never <= any real coordinate.
PAD_TIME = torch.iinfo(torch.int64).max // 2


@dataclass
class ExpertBatch:
    """The sub-batch of samples routed to one expert, in that expert's tokenization."""

    expert: int
    rows: Tensor  # [b] indices into the global batch
    slot: Tensor  # [b] position of this expert in each sample's active set (0 = primary)
    input_ids: Tensor  # [b, T]
    attention_mask: Tensor  # [b, T]
    time: Tensor  # [b, T] alignment coordinate (char offset of token end)
    labels: Tensor | None = None  # [b, T]
    primary_index: Tensor | None = None  # CPU long indices of rows with slot == 0 (kept on CPU by `to`)
    peer_index: Tensor | None = None  # CPU long indices of rows with slot != 0

    @property
    def position_ids(self) -> Tensor:
        return (self.attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    @property
    def primary_rows(self) -> Tensor:
        return self.slot == 0

    def primary_ids(self) -> Tensor:
        if self.primary_index is None:
            self.primary_index = (self.slot == 0).nonzero(as_tuple=True)[0].cpu()
        return self.primary_index

    def peer_ids(self) -> Tensor:
        if self.peer_index is None:
            self.peer_index = (self.slot != 0).nonzero(as_tuple=True)[0].cpu()
        return self.peer_index

    def to(self, device: torch.device | str) -> ExpertBatch:
        return ExpertBatch(
            self.expert,
            self.rows.to(device),
            self.slot.to(device),
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.time.to(device),
            None if self.labels is None else self.labels.to(device),
            self.primary_index,
            self.peer_index,
        )


def encode(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    completions: list[str] | None,
    max_length: int,
    add_eos: bool,
) -> tuple[list[Tensor], list[Tensor], list[Tensor] | None]:
    """Tokenize prompt(+completion) and derive per-token alignment coordinates and labels."""
    completions = completions or [""] * len(prompts)
    texts = [p + c for p, c in zip(prompts, completions)]
    enc = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length - int(add_eos),
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    ids_out, time_out, labels_out = [], [], []
    for ids, offsets, special, prompt, text in zip(
        enc["input_ids"], enc["offset_mapping"], enc["special_tokens_mask"], prompts, texts
    ):
        ids = torch.tensor(ids, dtype=torch.long)
        offsets = torch.tensor(offsets, dtype=torch.long).view(-1, 2)
        special = torch.tensor(special, dtype=torch.bool)
        time = offsets[:, 1]
        is_target = (offsets[:, 0] >= len(prompt)) & ~special
        labels = torch.where(is_target, ids, torch.full_like(ids, IGNORE_INDEX))
        if add_eos:
            eos = torch.tensor([tokenizer.eos_token_id])
            ids = torch.cat([ids, eos])
            time = torch.cat([time, torch.tensor([len(text) + 1])])
            labels = torch.cat([labels, eos])
        ids_out.append(ids)
        time_out.append(time)
        labels_out.append(labels)
    return ids_out, time_out, (labels_out if completions is not None else None)


def pad_stack(seqs: list[Tensor], length: int, value: int, side: str) -> Tensor:
    out = torch.full((len(seqs), length), value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        if side == "left":
            out[i, length - len(s) :] = s
        else:
            out[i, : len(s)] = s
    return out


def build_batches(
    pool: ExpertPool,
    active: Tensor,
    prompts: list[str],
    completions: list[str] | None,
    max_length: int,
    padding_side: str,
    add_eos: bool,
) -> dict[int, ExpertBatch]:
    """Tokenize each sample with every expert it is routed to; pad all experts to a common length."""
    active = active.cpu()
    pending: dict[int, tuple] = {}
    for m in active.unique().tolist():
        hit = active == m
        rows = hit.any(-1).nonzero(as_tuple=True)[0]
        slot = hit[rows].float().argmax(-1)
        sub_prompts = [prompts[i] for i in rows.tolist()]
        sub_completions = None if completions is None else [completions[i] for i in rows.tolist()]
        ids, time, labels = encode(pool[m].tokenizer, sub_prompts, sub_completions, max_length, add_eos)
        pending[m] = (rows, slot, ids, time, labels)

    length = max(len(s) for p in pending.values() for s in p[2])
    batches = {}
    for m, (rows, slot, ids, time, labels) in pending.items():
        expert = pool[m]
        batches[m] = ExpertBatch(
            expert=m,
            rows=rows,
            slot=slot,
            input_ids=pad_stack(ids, length, expert.pad_token_id, padding_side),
            attention_mask=pad_stack([torch.ones_like(s) for s in ids], length, 0, padding_side),
            time=pad_stack(time, length, PAD_TIME, padding_side),
            labels=None if labels is None else pad_stack(labels, length, IGNORE_INDEX, padding_side),
            primary_index=(slot == 0).nonzero(as_tuple=True)[0],
            peer_index=(slot != 0).nonzero(as_tuple=True)[0],
        )
    return batches
