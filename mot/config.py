from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import MISSING, OmegaConf


@dataclass
class ExpertSpec:
    name: str = MISSING
    tokenizer: str | None = None
    device: str | None = None


@dataclass
class RouterConfig:
    encoder: str = "microsoft/deberta-v3-base"
    encoder_max_length: int = 512
    hidden_size: int = 768
    temperature: float = 1.0
    dropout: float = 0.1
    gumbel_noise: bool = True


@dataclass
class InteractionConfig:
    num_layers: int = 8
    shared_dim: int = 2048
    num_heads: int = 8
    dropout: float = 0.1
    placement: str = "uniform"
    cross_attention: bool = True


@dataclass
class ModelConfig:
    experts: list[ExpertSpec] = field(default_factory=list)
    top_k: int = 3
    router: RouterConfig = field(default_factory=RouterConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    dtype: str = "bfloat16"
    attn_implementation: str = "kernels-community/flash-attn3"
    devices: list[str] | None = None
    main_device: str = "cuda:0"
    gradient_checkpointing: bool = True


@dataclass
class LossConfig:
    """total = lm + routing_weight * routing.

    routing: supervised router objective computed from per-sample expert quality labels
      - "ce":   softmax cross-entropy against the label distribution (hard argmax if labels are 0/1)
      - "bce":  independent sigmoid per expert against labels in [0, 1]
      - "none": router learns only through the straight-through LM gradient
    """

    routing: str = "ce"
    routing_weight: float = 1.0


@dataclass
class TrainConfig:
    data: str = MISSING
    output_dir: str = "runs/default"
    epochs: int = 50
    batch_size: int = 64
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    max_length: int = 512
    seed: int = 42
    log_every: int = 10
    save_every: int = 1000
    num_workers: int = 4
    resume: str | None = None
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class GenerationConfig:
    max_prompt_length: int = 2048
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True
    batch_size: int = 16


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg = OmegaConf.structured(Config)
    if path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(cfg)


def to_dict(cfg: Any) -> dict:
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
