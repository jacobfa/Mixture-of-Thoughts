"""baselines/avengers/app.py with the Zooter and LoraRetriever routers registered (router.type: zooter | loraretriever)."""

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
import torch

from common import AVENGERS

sys.path.insert(0, str(AVENGERS))

import app
from config.config_loader import load_config
from core.routing import factory
from core.routing.base_router import BaseRouter, RouterOutput
from loraretriever import Encoder, encode
from transformers import AutoTokenizer
from zooter import ZooterModel, tokenizer_for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TopOneRouter(BaseRouter):
    def __init__(self, normal_experts, thinking_experts, model_dir: str):
        super().__init__(normal_experts, thinking_experts)
        self.model_dir = Path(model_dir)
        with open(self.model_dir / "meta.json") as f:
            self.meta = json.load(f)
        by_name = {e.model_name: e for e in normal_experts}
        self.experts = [by_name[name] for name in self.meta["experts"]]
        self._lock = threading.Lock()

    def scores(self, question: str) -> torch.Tensor:
        raise NotImplementedError

    def route(self, question: str) -> RouterOutput:
        with self._lock, torch.no_grad():
            chosen = self.experts[int(self.scores(question).argmax())]
        return RouterOutput(normal_experts=[chosen], thinking_experts=self.thinking_experts)


class ZooterRouter(TopOneRouter):
    def __init__(self, normal_experts, thinking_experts, router_config: dict):
        super().__init__(normal_experts, thinking_experts, router_config["zooter_router"]["model_dir"])
        self.tokenizer = tokenizer_for(self.meta["encoder"])
        self.model = ZooterModel(self.meta["encoder"], len(self.experts))
        self.model.load_state_dict(torch.load(self.model_dir / "best_model.pth", map_location="cpu"))
        self.model.to(DEVICE).eval()

    def scores(self, question: str) -> torch.Tensor:
        inputs = self.tokenizer(question, truncation=True, max_length=self.meta["max_length"], return_tensors="pt")
        return self.model(**inputs.to(DEVICE))[0]


class LoraRetrieverRouter(TopOneRouter):
    def __init__(self, normal_experts, thinking_experts, router_config: dict):
        super().__init__(normal_experts, thinking_experts, router_config["loraretriever_router"]["model_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "encoder")
        self.model = Encoder(str(self.model_dir / "encoder"), self.meta["pooling"]).to(DEVICE).eval()
        cents = torch.from_numpy(np.load(self.model_dir / "centroids.npy")).to(DEVICE)
        self.centroids = torch.nn.functional.normalize(cents, dim=-1)

    def scores(self, question: str) -> torch.Tensor:
        query = encode(
            self.model, self.tokenizer, [question], self.meta["instruction"], self.meta["max_length"], DEVICE
        )
        return (query @ self.centroids.T)[0]


EXTRA_ROUTERS = {"zooter": ZooterRouter, "loraretriever": LoraRetrieverRouter}
_create_router = factory.RouterFactory.create_router


def create_router(normal_experts, thinking_experts, router_config):
    cls = EXTRA_ROUTERS.get(router_config["type"])
    if cls is None:
        return _create_router(
            normal_experts=normal_experts, thinking_experts=thinking_experts, router_config=router_config
        )
    return cls(normal_experts, thinking_experts, router_config)


factory.RouterFactory.create_router = staticmethod(create_router)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--save_dir", default="results")
    args = parser.parse_args()
    app.run_experiment(load_config(args.config), args.save_dir)
