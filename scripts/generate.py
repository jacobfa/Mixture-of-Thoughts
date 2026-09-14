import argparse
import json
import logging
import sys

from tqdm import tqdm

from mot import MixtureOfThoughts, load_config


def read_prompts(path: str | None) -> list[str]:
    stream = open(path) if path else sys.stdin
    prompts = []
    for line in stream:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line) if line.startswith("{") else {"prompt": line}
        prompts.append(row["prompt"])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with a trained MoT checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="directory written by MixtureOfThoughts.save; omit for untrained weights")
    parser.add_argument("--prompts", help="JSONL ({'prompt': ...}) or plain text, one prompt per line; default stdin")
    parser.add_argument("--output", help="JSONL output path; default stdout")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.overrides)
    model = MixtureOfThoughts.load(args.checkpoint, cfg.model) if args.checkpoint else MixtureOfThoughts(cfg.model)
    prompts = read_prompts(args.prompts)
    out = open(args.output, "w") if args.output else sys.stdout
    bs = cfg.generation.batch_size
    for i in tqdm(range(0, len(prompts), bs), disable=out is sys.stdout):
        result = model.generate(prompts[i : i + bs], cfg.generation)
        for prompt, text, active in zip(prompts[i : i + bs], result.texts, result.routing.active.tolist()):
            out.write(json.dumps({"prompt": prompt, "completion": text, "experts": active}) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
