from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper

from mot.config import GenerationConfig, LossConfig, ModelConfig
from mot.experts import PAD_TIME, ExpertBatch, ExpertPool, build_batches
from mot.interaction import InteractionCache, InteractionLayer
from mot.losses import lm_loss, routing_loss
from mot.routing import PromptEncoder, Router, RoutingDecision

WEIGHTS_FILE = "mot.safetensors"
CONFIG_FILE = "model.yaml"


@dataclass
class StepInputs:
    input_ids: Tensor  # [b, t]
    attention_mask: Tensor  # [b, T_total]
    position_ids: Tensor  # [b, t]
    time: Tensor  # [b, t]


@dataclass
class GenerationOutput:
    texts: list[str]
    token_ids: list[list[int]]
    routing: RoutingDecision
    steps: int = 0


class MixtureOfThoughts(nn.Module):
    """Global router + Q latent interaction layers over a pool of frozen heterogeneous experts."""

    def __init__(self, cfg: ModelConfig, pool: ExpertPool | None = None):
        super().__init__()
        self.cfg = cfg
        self.pool = pool if pool is not None else ExpertPool.from_config(cfg)
        self.main_device = torch.device(cfg.main_device if torch.cuda.is_available() else "cpu")
        self.encoder = PromptEncoder(cfg.router.encoder, cfg.router.encoder_max_length, self.main_device)
        self.router = Router(
            self.encoder.output_dim,
            cfg.router.hidden_size,
            len(self.pool),
            cfg.top_k,
            cfg.router.temperature,
            cfg.router.dropout,
        ).to(self.main_device)
        ia = cfg.interaction
        self.interactions = nn.ModuleList(
            InteractionLayer(
                self.pool.hidden_sizes, ia.shared_dim, ia.num_heads, cfg.top_k, ia.dropout, ia.cross_attention
            )
            for _ in range(ia.num_layers)
        ).to(self.main_device)
        self.pool.set_gradient_checkpointing(cfg.gradient_checkpointing)

    # ------------------------------------------------------------------ module plumbing

    def train(self, mode: bool = True) -> MixtureOfThoughts:
        super().train(mode)
        self.pool.train(mode)
        self.encoder.eval()
        return self

    def trainable_modules(self) -> nn.ModuleDict:
        return nn.ModuleDict({"router": self.router, "interactions": self.interactions})

    def trainable_parameters(self):
        return self.trainable_modules().parameters()

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().contiguous() for k, v in self.trainable_modules().state_dict().items()}
        save_file(state, path / WEIGHTS_FILE)
        OmegaConf.save(OmegaConf.structured(self.cfg), path / CONFIG_FILE)

    @classmethod
    def load(cls, path: str | Path, cfg: ModelConfig | None = None, pool: ExpertPool | None = None) -> MixtureOfThoughts:
        path = Path(path)
        if cfg is None:
            cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.load(path / CONFIG_FILE)))
        model = cls(cfg, pool)
        model.trainable_modules().load_state_dict(load_file(path / WEIGHTS_FILE, device=str(model.main_device)))
        return model

    def _autocast(self):
        if self.main_device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    # ------------------------------------------------------------------ routing

    def route(self, prompts: list[str]) -> RoutingDecision:
        """Discrete top-K routing as used at inference (no noise, no straight-through gate)."""
        return self.router.select(self.router(self.encoder(prompts)), noisy=False, straight_through=False)

    # ------------------------------------------------------------------ core forward

    def _run_stacks(
        self,
        step: dict[int, StepInputs],
        batches: dict[int, ExpertBatch],
        gate: Tensor,
        kv_caches: dict[int, Cache] | None = None,
        ia_caches: list[InteractionCache] | None = None,
    ) -> dict[int, Tensor]:
        """Advance every active expert stream through all segments, interleaving interaction layers."""
        states = {}
        for m, inp in step.items():
            expert = self.pool[m]
            dev = expert.device
            states[m] = expert.begin(
                inp.input_ids.to(dev),
                inp.attention_mask.to(dev),
                inp.position_ids.to(dev),
                kv_caches[m] if kv_caches is not None else None,
            )
        times = {m: inp.time for m, inp in step.items()}
        for q, layer in enumerate(self.interactions):
            hiddens = {m: self.pool[m].run_segment(states[m], q) for m in states}
            with self._autocast():
                updated = layer(hiddens, batches, times, gate, ia_caches[q] if ia_caches is not None else None)
            for m, state in states.items():
                state.hidden = updated[m]
        tail = len(self.interactions)
        return {m: self.pool[m].run_segment(states[m], tail) for m in states}

    def _teacher_forced(self, batches: dict[int, ExpertBatch], gate: Tensor) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        step = {m: StepInputs(b.input_ids, b.attention_mask, b.position_ids, b.time) for m, b in batches.items()}
        hidden = self._run_stacks(step, batches, gate)
        logits, labels = {}, {}
        for m, h in hidden.items():
            primary = batches[m].primary_rows
            if primary.any():
                logits[m] = self.pool[m].logits(h[primary.to(h.device)])
                labels[m] = batches[m].labels[primary].to(logits[m].device)
        return logits, labels

    def forward(
        self,
        prompts: list[str],
        completions: list[str],
        max_length: int,
        expert_targets: Tensor | None = None,
        loss_weights: LossConfig | None = None,
    ) -> dict[str, Tensor]:
        """Teacher-forced MoT step. `expert_targets` [B, M] are per-expert quality labels for the
        supervised routing loss; when absent the router is trained by the straight-through LM gradient only."""
        w = loss_weights or LossConfig()
        scores = self.router(self.encoder(prompts))
        decision = self.router.select(
            scores, noisy=self.training and self.cfg.router.gumbel_noise, straight_through=self.training
        )
        batches = self._batches(decision.active, prompts, completions, max_length, "right", add_eos=True)
        logits, labels = self._teacher_forced(batches, decision.gate)

        out = {"lm": lm_loss(logits, labels).to(self.main_device)}
        total = out["lm"]
        if expert_targets is not None and w.routing != "none":
            out["routing"] = routing_loss(scores, expert_targets, w.routing)
            total = total + w.routing_weight * out["routing"]
        out["loss"] = total
        out["active"] = decision.active
        return out

    def _batches(self, active, prompts, completions, max_length, padding_side, add_eos) -> dict[int, ExpertBatch]:
        batches = build_batches(self.pool, active, prompts, completions, max_length, padding_side, add_eos)
        return {m: b.to(self.main_device) for m, b in batches.items()}

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(self, prompts: list[str], gen: GenerationConfig | None = None) -> GenerationOutput:
        gen = gen or GenerationConfig()
        was_training = self.training
        self.eval()
        try:
            return self._generate(prompts, gen)
        finally:
            self.train(was_training)

    def _generate(self, prompts: list[str], gen: GenerationConfig) -> GenerationOutput:
        dev = self.main_device
        B = len(prompts)
        decision = self.route(prompts)
        batches = self._batches(decision.active, prompts, None, gen.max_prompt_length, "left", add_eos=False)
        base_time = torch.tensor([len(p) for p in prompts], device=dev)
        warpers = self._warpers(gen)

        kv_caches = {m: self.pool[m].new_cache() for m in batches}
        ia_caches = [InteractionCache() for _ in self.interactions] if self.cfg.interaction.cross_attention else None
        step = {m: StepInputs(b.input_ids, b.attention_mask, b.position_ids, b.time) for m, b in batches.items()}
        hidden = self._run_stacks(step, batches, decision.gate, kv_caches, ia_caches)

        attn_mask = {m: b.attention_mask for m, b in batches.items()}
        next_pos = {m: b.position_ids[:, -1] + 1 for m, b in batches.items()}
        alive = {m: torch.ones(len(b.rows), dtype=torch.bool, device=dev) for m, b in batches.items()}
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        output = torch.full((B, gen.max_new_tokens), -1, dtype=torch.long, device=dev)

        steps = 0
        for t in range(gen.max_new_tokens):
            steps = t + 1
            tokens = {}
            for m, h in hidden.items():
                expert, b = self.pool[m], batches[m]
                nxt = self._sample(expert.logits(h[:, -1]), warpers).to(dev)
                nxt = torch.where(alive[m], nxt, torch.full_like(nxt, expert.pad_token_id))
                is_eos = nxt == expert.eos_token_id
                primary = b.primary_rows
                rows = b.rows[primary]
                output[rows, t] = torch.where(done[rows], torch.full_like(rows, -1), nxt[primary])
                done[rows] |= is_eos[primary]
                alive[m] &= ~is_eos
                tokens[m] = nxt[:, None]
            if done.all():
                break

            step = {}
            for m in hidden:
                b = batches[m]
                alive[m] &= ~done[b.rows]
                if not alive[m].any():
                    continue
                attn_mask[m] = torch.cat([attn_mask[m], torch.ones_like(alive[m][:, None], dtype=attn_mask[m].dtype)], -1)
                time = torch.where(alive[m], base_time[b.rows] + 1 + t, torch.full_like(base_time[b.rows], PAD_TIME))
                step[m] = StepInputs(tokens[m], attn_mask[m], next_pos[m][:, None], time[:, None])
                next_pos[m] = next_pos[m] + 1
            hidden = self._run_stacks(step, {m: batches[m] for m in step}, decision.gate, kv_caches, ia_caches)

        texts, token_ids = [], []
        for i, row in enumerate(output[:, :steps].tolist()):
            expert = self.pool[decision.primary[i].item()]
            ids = [tok for tok in row if tok >= 0]
            if ids and ids[-1] == expert.eos_token_id:
                ids = ids[:-1]
            token_ids.append(ids)
            texts.append(expert.tokenizer.decode(ids, skip_special_tokens=True))
        return GenerationOutput(texts=texts, token_ids=token_ids, routing=decision, steps=steps)

    @staticmethod
    def _warpers(gen: GenerationConfig) -> LogitsProcessorList | None:
        if not gen.do_sample:
            return None
        warpers = LogitsProcessorList()
        if gen.temperature and gen.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(gen.temperature))
        if gen.top_p < 1.0:
            warpers.append(TopPLogitsWarper(gen.top_p))
        return warpers

    @staticmethod
    def _sample(logits: Tensor, warpers: LogitsProcessorList | None) -> Tensor:
        if warpers is None:
            return logits.argmax(-1)
        scores = warpers(None, logits.float())
        return torch.multinomial(scores.softmax(-1), num_samples=1).squeeze(1)
