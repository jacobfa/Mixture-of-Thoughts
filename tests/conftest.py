from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import BertConfig, BertModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from mot.config import ExpertSpec, InteractionConfig, ModelConfig, RouterConfig
from mot.model import MixtureOfThoughts

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Solve 12 + 7 and explain each step carefully.",
    "def add(a, b): return a + b",
    "Paris is the capital of France and Berlin of Germany.",
    "Mixture of thoughts lets frozen experts collaborate in latent space.",
]

# (hidden, layers, vocab, tokenizer kind)
EXPERT_SHAPES = [(32, 4, 96, "bpe"), (48, 6, 128, "wordlevel"), (32, 2, 64, "bpe")]


def make_tokenizer(kind: str, vocab_size: int, path) -> PreTrainedTokenizerFast:
    if kind == "bpe":
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
    else:
        tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
    tok.train_from_iterator(CORPUS * 4, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>"
    )
    fast.save_pretrained(path)
    return fast


def make_expert(path, hidden: int, layers: int, vocab: int, kind: str) -> None:
    tokenizer = make_tokenizer(kind, vocab, path)
    cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )
    LlamaForCausalLM(cfg).save_pretrained(path)


def make_encoder(path) -> None:
    tokenizer = make_tokenizer("wordlevel", 128, path)
    BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=48,
            max_position_embeddings=128,
            pad_token_id=3,
        )
    ).save_pretrained(path)


@pytest.fixture(scope="session")
def tiny_repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    for i, (hidden, layers, vocab, kind) in enumerate(EXPERT_SHAPES):
        make_expert(root / f"expert{i}", hidden, layers, vocab, kind)
    make_encoder(root / "encoder")
    return root


def tiny_config(root, **overrides) -> ModelConfig:
    cfg = ModelConfig(
        experts=[ExpertSpec(name=str(root / f"expert{i}")) for i in range(len(EXPERT_SHAPES))],
        top_k=2,
        router=RouterConfig(encoder=str(root / "encoder"), encoder_max_length=64, hidden_size=16, dropout=0.0),
        interaction=InteractionConfig(num_layers=2, shared_dim=16, num_heads=2, dropout=0.0),
        dtype="float32",
        attn_implementation="sdpa",
        devices=["cpu"],
        main_device="cpu",
        gradient_checkpointing=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="session")
def tiny_model(tiny_repo) -> MixtureOfThoughts:
    torch.manual_seed(0)
    return MixtureOfThoughts(tiny_config(tiny_repo))


@pytest.fixture
def jsonl_data(tmp_path):
    path = tmp_path / "train.jsonl"
    with open(path, "w") as f:
        for i, text in enumerate(CORPUS):
            head, tail = text[: len(text) // 2], text[len(text) // 2 :]
            scores = [float(j == i % len(EXPERT_SHAPES)) for j in range(len(EXPERT_SHAPES))]
            f.write(json.dumps({"prompt": head, "completion": tail, "expert_scores": scores}) + "\n")
    return path
