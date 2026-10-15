"""Routing config (routing.yaml) - the single source of truth for task -> model chain."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

ROUTING_PATH = Path(__file__).with_name("routing.yaml")


class ModelSpec(BaseModel):
    tier: int
    price_in: float   # $ per 1M input tokens (standard rate)
    price_out: float  # $ per 1M output tokens


class TaskRoute(BaseModel):
    chain: list[str]
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    mode: Literal["sync", "batch"] = "sync"
    timeout_s: float = 30.0
    escalate_on_invalid: bool = False


class Routing(BaseModel):
    models: dict[str, ModelSpec]
    tasks: dict[str, TaskRoute]

    @model_validator(mode="after")
    def _chains_reference_known_models(self) -> "Routing":
        for name, route in self.tasks.items():
            unknown = [m for m in route.chain if m not in self.models]
            if unknown or not route.chain:
                raise ValueError(f"task {name!r} has an empty chain or unknown models {unknown}")
        return self

    def cost_usd(self, model: str, input_tokens: int, output_tokens: int, cache_read: int = 0,
                 cache_write: int = 0, batch: bool = False) -> float:
        """Cache reads bill at 0.1x and cache writes at 1.25x the input rate; the Batch API halves everything."""
        spec = self.models[model]
        cost = (input_tokens + 0.1 * cache_read + 1.25 * cache_write) * spec.price_in + output_tokens * spec.price_out
        cost /= 1e6
        return cost / 2 if batch else cost


@lru_cache
def load_routing(path: Path = ROUTING_PATH) -> Routing:
    return Routing.model_validate(yaml.safe_load(path.read_text()))
