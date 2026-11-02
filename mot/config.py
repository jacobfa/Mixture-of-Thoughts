from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING, OmegaConf


@dataclass
class ExpertSpec:
    name: str = MISSING
    tokenizer: str | None = None
    device: str | None = None
    attn_implementation: str | None = None  # overrides model.attn_implementation for this expert


@dataclass
class RouterConfig:
    encoder: str = "dmis-lab/UltraBERT-Embed-150M"
    encoder_max_length: int = 8192
    pooling: str = "cls"  # mean | cls | last
    hidden_size: int = 768
    temperature: float = 1.0
    dropout: float = 0.1
    gumbel_noise: bool = True


@dataclass
class InteractionConfig:
    """Latent interaction block. The paper's block is recovered with
    norm=false rope=false expert_bias=false ffn=false symmetric=false num_registers=0.

    norm:          per-expert RMSNorm on hidden states before projection, pre-norm on latents, QK-norm
    rope:          rotary positions in the shared space computed from the character-offset alignment time
                   (the one coordinate shared by all tokenizers); time_scale ~ characters per position
    expert_bias:   learned per-expert key/value bias so attention knows which stream a key came from
    ffn:           SwiGLU after attention (hidden = ffn_mult * shared_dim)
    symmetric:     every active stream attends over the board (default: only the primary reads)
    num_registers: learned key/value register slots always visible to every query (attention sinks)
    loops:         passes of each interaction block at its position (shared weights, learned per-pass embedding);
                   >1 = recurrent latent refinement without re-running expert layers"""

    num_layers: int = 8
    shared_dim: int = 2048
    num_heads: int = 8
    dropout: float = 0.1
    placement: str = "uniform"
    cross_attention: bool = True
    norm: bool = True
    rope: bool = True
    rope_theta: float = 10000.0
    time_scale: float = 4.0
    expert_bias: bool = True
    ffn: bool = True
    ffn_mult: float = 2.0
    symmetric: bool = True
    num_registers: int = 4
    loops: int = 1
    calibrate: bool = True  # scale each reverse projector by the noise level its expert tolerates (measured once)
    calibration_target: float = 0.02  # tolerated increase of an expert's own NLL (nats/token) at unit gain
    calibration_samples: int = 4


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
    compile: bool = False


@dataclass
class LossConfig:
    """total = lm + routing_weight * routing.

    routing: supervised router objective computed from per-sample expert quality labels
      - "ce":   softmax cross-entropy against the label distribution (hard argmax if labels are 0/1)
      - "bce":  independent sigmoid per expert against labels in [0, 1]
      - "none": router learns only through the straight-through LM gradient
    peer_lm_weight: NLL of the non-primary streams on their own tokenization of the completion; anchors peer
      representations that are otherwise only shaped by what helps the primary. 0 disables it and skips the
      peers' lm_head pass.
    lm_chunk_size: completion tokens per lm_head + cross-entropy chunk (bounds the [N, V] logits buffer)
    fused_ce: use Liger's fused linear-cross-entropy kernel when installed and running on CUDA
    """

    routing: str = "ce"
    routing_weight: float = 1.0
    peer_lm_weight: float = 0.0
    lm_chunk_size: int = 2048
    fused_ce: bool = True


@dataclass
class OptimConfig:
    """muon: Muon on 2D hidden weights (RMS-matched to AdamW, so lr/weight decay are shared) + AdamW on
    biases and the router head. adamw: fused torch AdamW on everything."""

    name: str = "muon"
    muon_lr: float | None = None
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    adjust_lr: str = "match_rms_adamw"
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8


@dataclass
class WandbConfig:
    project: str = "mot"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"  # online | offline | disabled (falls back to offline without an API key)


@dataclass
class TrainConfig:
    data: str = MISSING
    output_dir: str = "runs/default"
    log_file: str | None = None  # default logs/<run name>.log in the working directory
    wandb: WandbConfig = field(default_factory=WandbConfig)
    epochs: int = 50
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.01
    scheduler: str = "cosine_with_min_lr"
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    max_length: int = 512
    group_by_length: bool = True
    seed: int = 42
    log_every: int = 10
    save_every: int = 1000
    num_workers: int = 4
    resume: str | None = None
    deepspeed: str | None = None
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class GenerationConfig:
    max_prompt_length: int = 2048
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True
    batch_size: int = 16
    stop: str | None = None  # stop a sequence once its generated text contains this string (cut from the output)


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
