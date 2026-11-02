"""OpenAI-compatible /v1/chat/completions server for a MoT checkpoint.

Messages are flattened to one raw prompt string (contents joined by blank lines), the same text the frozen experts
see in training and in the raw-template vLLM servers, so the comparison is over identical inputs. Requests that
arrive within `--batch-wait` seconds and share sampling parameters are generated as one batch.
"""

import argparse
import json
import logging
import threading
import time
import uuid
from dataclasses import astuple, dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue

import torch

from mot import GenerationConfig, MixtureOfThoughts, load_config

log = logging.getLogger("serve")


def flatten(messages: list[dict]) -> str:
    return "\n\n".join(m["content"] for m in messages if isinstance(m.get("content"), str) and m["content"])


@dataclass
class Job:
    prompt: str
    gen: GenerationConfig
    done: threading.Event
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class Engine:
    def __init__(self, model: MixtureOfThoughts, default: GenerationConfig, batch_wait: float):
        self.model = model
        self.default = default
        self.batch_wait = batch_wait
        self.queue: Queue[Job] = Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def config(self, body: dict) -> GenerationConfig:
        temperature = float(body.get("temperature", self.default.temperature))
        stop = body.get("stop")
        return replace(
            self.default,
            max_new_tokens=int(body.get("max_tokens") or self.default.max_new_tokens),
            temperature=temperature,
            top_p=float(body.get("top_p", self.default.top_p)),
            do_sample=temperature > 0,
            stop=stop[0] if isinstance(stop, list) and stop else stop if isinstance(stop, str) else None,
        )

    def submit(self, prompt: str, gen: GenerationConfig, n: int) -> list[Job]:
        jobs = [Job(prompt, gen, threading.Event()) for _ in range(n)]
        for job in jobs:
            self.queue.put(job)
        return jobs

    def _loop(self) -> None:
        while True:
            jobs = [self.queue.get()]
            deadline = time.monotonic() + self.batch_wait
            while len(jobs) < self.default.batch_size:
                try:
                    jobs.append(self.queue.get(timeout=max(0.0, deadline - time.monotonic())))
                except Empty:
                    break
            groups: dict[tuple, list[Job]] = {}
            for job in jobs:
                groups.setdefault(astuple(job.gen), []).append(job)
            for group in groups.values():
                self._run(group)

    def _run(self, jobs: list[Job]) -> None:
        try:
            out = self.model.generate([j.prompt for j in jobs], jobs[0].gen)
            for job, text, ids, primary in zip(jobs, out.texts, out.token_ids, out.routing.primary.tolist(), strict=True):
                job.text = text
                job.completion_tokens = len(ids)
                job.prompt_tokens = len(self.model.pool[primary].tokenizer(job.prompt).input_ids)
        except Exception as err:
            log.exception("generation failed")
            for job in jobs:
                job.error = str(err)
        finally:
            for job in jobs:
                job.done.set()


class Handler(BaseHTTPRequestHandler):
    engine: Engine
    model_name: str
    timeout_s: float

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": self.model_name, "object": "model"}]})
        elif self.path.rstrip("/") in ("/health", ""):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        prompt = flatten(body.get("messages", []))
        if not prompt:
            return self._send(400, {"error": "messages must contain text content"})
        jobs = self.engine.submit(prompt, self.engine.config(body), max(1, int(body.get("n", 1))))
        for job in jobs:
            if not job.done.wait(self.timeout_s):
                return self._send(504, {"error": "generation timed out"})
            if job.error:
                return self._send(500, {"error": job.error})
        self._send(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", self.model_name),
                "choices": [
                    {"index": i, "message": {"role": "assistant", "content": j.text}, "finish_reason": "stop"}
                    for i, j in enumerate(jobs)
                ],
                "usage": {
                    "prompt_tokens": jobs[0].prompt_tokens,
                    "completion_tokens": sum(j.completion_tokens for j in jobs),
                    "total_tokens": jobs[0].prompt_tokens + sum(j.completion_tokens for j in jobs),
                },
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a MoT checkpoint behind an OpenAI-compatible chat endpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="directory written by MixtureOfThoughts.save; omit for untrained weights")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--served-model-name", default="mot")
    parser.add_argument("--batch-wait", type=float, default=0.05, help="seconds to collect requests into one batch")
    parser.add_argument("--timeout", type=float, default=1800.0, help="per-request generation timeout in seconds")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_float32_matmul_precision("high")

    cfg = load_config(args.config, args.overrides)
    model = MixtureOfThoughts.load(args.checkpoint, cfg.model) if args.checkpoint else MixtureOfThoughts(cfg.model)
    Handler.engine = Engine(model, cfg.generation, args.batch_wait)
    Handler.model_name = args.served_model_name
    Handler.timeout_s = args.timeout
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    log.info("serving %s on http://%s:%d/v1", args.served_model_name, args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
