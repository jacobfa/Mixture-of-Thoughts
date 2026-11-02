"""Serve a pool's experts and the Avengers embedding model as OpenAI-compatible vLLM endpoints.

Experts use the raw chat template, so a chat request is tokenized exactly like the plain prompt MoT sees.
Servers are assigned round-robin over --gpus; the memory fraction is divided among the servers sharing a GPU.
"""

import argparse
import os
import subprocess
import time
from collections import Counter

from common import EXP, MOT, harness_config, port_of


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), help="comma-separated ids")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--embedding-port", type=int, default=8100)
    parser.add_argument("--no-embedding", action="store_true")
    args = parser.parse_args()

    cfg = harness_config(args.pool)
    gpus = args.gpus.split(",")
    servers = [
        (e["name"], port_of(e["base_url"]), ["--chat-template", str(EXP / "raw_chat_template.jinja")])
        for e in cfg["experts"]
        if e["name"] != MOT
    ]
    if not args.no_embedding:
        servers.append((cfg["router"]["rank_router"]["embedding_model"], args.embedding_port, ["--task", "embed"]))
    placement = [gpus[i % len(gpus)] for i in range(len(servers))]
    share = Counter(placement)

    procs = []
    for (model, port, extra), gpu in zip(servers, placement, strict=True):
        memory = f"{args.gpu_memory_utilization / share[gpu]:.3f}"
        cmd = ["vllm", "serve", model, "--served-model-name", model, "--port", str(port)]
        cmd += ["--max-model-len", str(args.max_model_len), "--gpu-memory-utilization", memory, *extra]
        print(f"[gpu {gpu}] {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}))
        time.sleep(5)
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(2)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            p.wait()


if __name__ == "__main__":
    main()
