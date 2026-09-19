"""Sample completions from each raw expert with vLLM.

MoT inference itself (cross-expert latent interaction mid-stack) has no vLLM equivalent; this covers the
places where plain causal-LM generation is needed: producing per-expert answers to score into `expert_scores`
routing labels, and routing-only / single-expert baselines.

Output JSONL: {"prompt": str, "completions": {"<expert index>": [str, ...]}}
"""

import argparse
import gc
import json
import logging
import sys

import torch

from mot import load_config


def read_prompts(path: str | None) -> list[str]:
    stream = open(path) if path else sys.stdin
    prompts = []
    for line in stream:
        line = line.strip()
        if line:
            prompts.append(json.loads(line)["prompt"] if line.startswith("{") else line)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-expert generation with vLLM.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", help="JSONL ({'prompt': ...}) or plain text; default stdin")
    parser.add_argument("--output", help="JSONL output path; default stdout")
    parser.add_argument("--experts", help="comma-separated expert indices; default all")
    parser.add_argument("--n", type=int, default=1, help="samples per prompt per expert")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from vllm import LLM, SamplingParams

    cfg = load_config(args.config, args.overrides)
    gen = cfg.generation
    experts = [int(i) for i in args.experts.split(",")] if args.experts else list(range(len(cfg.model.experts)))
    prompts = read_prompts(args.prompts)
    sampling = SamplingParams(
        n=args.n,
        temperature=gen.temperature if gen.do_sample else 0.0,
        top_p=gen.top_p if gen.do_sample else 1.0,
        max_tokens=gen.max_new_tokens,
    )
    results: list[dict] = [{"prompt": p, "completions": {}} for p in prompts]
    for m in experts:
        spec = cfg.model.experts[m]
        llm = LLM(
            model=spec.name,
            tokenizer=spec.tokenizer or spec.name,
            dtype=cfg.model.dtype,
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=gen.max_prompt_length + gen.max_new_tokens,
            enable_prefix_caching=True,
        )
        for row, out in zip(results, llm.generate(prompts, sampling)):
            row["completions"][str(m)] = [o.text for o in out.outputs]
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    out = open(args.output, "w") if args.output else sys.stdout
    for row in results:
        out.write(json.dumps(row) + "\n")
    out.flush()


if __name__ == "__main__":
    main()
