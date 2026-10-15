"""Message Batches API jobs for offline LLM stages (50% price), using the gateway's task registry so
prompts, schemas, routing and guardrails are identical to the online path.

A job is a list of items (custom_id -> task inputs). `submit` sends each item to the first model in its
task's chain; `collect` validates results with the task guardrail and returns failures, which the caller can
resubmit to `next_model()` (the escalation cascade from the cost policy). State lives in a JSON file so a
pipeline run survives restarts while batches are processing (usually minutes, at most 24 h).
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.llm import anthropic_client
from genai_gateway.config import load_routing
from genai_gateway.core import build_batch_request, next_model, parse_and_validate

STATE_DIR = Path("data/pipeline")


@dataclass
class JobResult:
    ok: dict[str, dict] = field(default_factory=dict)        # custom_id -> validated output
    invalid: dict[str, str] = field(default_factory=dict)    # custom_id -> reason (escalation candidates)
    errored: list[str] = field(default_factory=list)         # API-level failures (retry same model)
    cost_usd: float = 0.0
    tokens: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0, "cache_read": 0})


class BatchJob:
    """One stage of one pipeline run, e.g. run 20261014T1530 / segment."""

    def __init__(self, run_id: str, stage: str):
        self.path = STATE_DIR / run_id / f"{stage}.json"
        self.state: dict[str, Any] = (json.loads(self.path.read_text()) if self.path.exists()
                                      else {"batches": [], "items": {}})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=1))

    @property
    def items(self) -> dict[str, dict]:
        return self.state["items"]  # custom_id -> {"task", "inputs", "meta", "model"}

    def submit(self, items: dict[str, dict], model_for: dict[str, str] | None = None) -> str | None:
        """items: custom_id -> {"task", "inputs", "meta"}. model_for overrides the chain's first model."""
        if not items:
            return None
        requests = []
        for cid, it in items.items():
            model = (model_for or {}).get(cid) or next_model(it["task"])
            requests.append(build_batch_request(it["task"], cid, it["inputs"], model=model))
            self.items[cid] = {**it, "model": model}
        batch = anthropic_client().messages.batches.create(requests=requests)
        self.state["batches"].append({"id": batch.id, "submitted_at": int(time.time()), "n": len(requests),
                                      "collected": False})
        self.save()
        return batch.id

    def status(self) -> list[tuple[str, str, int]]:
        client = anthropic_client()
        out = []
        for b in self.state["batches"]:
            info = client.messages.batches.retrieve(b["id"])
            out.append((b["id"], info.processing_status, info.request_counts.processing))
        return out

    def done(self) -> bool:
        return all(s == "ended" for _, s, _ in self.status())

    def collect(self) -> JobResult:
        """Validate every uncollected, ended batch. Call again after resubmitting escalations."""
        client, routing, res = anthropic_client(), load_routing(), JobResult()
        for b in self.state["batches"]:
            if b["collected"] or client.messages.batches.retrieve(b["id"]).processing_status != "ended":
                continue
            for r in client.messages.batches.results(b["id"]):
                cid, item = r.custom_id, self.items[r.custom_id]
                if r.result.type != "succeeded":
                    res.errored.append(cid)
                    continue
                msg = r.result.message
                u = msg.usage
                res.tokens["input"] += u.input_tokens
                res.tokens["output"] += u.output_tokens
                res.tokens["cache_read"] += u.cache_read_input_tokens or 0
                res.cost_usd += routing.cost_usd(item["model"], u.input_tokens, u.output_tokens,
                                                 u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0,
                                                 batch=True)
                if msg.stop_reason != "end_turn":
                    res.invalid[cid] = f"stop_reason={msg.stop_reason}"
                    continue
                text = next((blk.text for blk in msg.content if blk.type == "text"), "")
                ok, output, reason = parse_and_validate(item["task"], item["inputs"], text)
                if ok:
                    res.ok[cid] = output
                    item["output"], item["final_model"] = output, item["model"]
                else:
                    res.invalid[cid] = reason or "invalid"
            b["collected"] = True
        self.state.setdefault("cost_usd", 0.0)
        self.state["cost_usd"] += res.cost_usd
        self.save()
        return res

    def escalate(self, invalid: dict[str, str]) -> tuple[str | None, list[str]]:
        """Resubmit invalid items to the next model in their task's chain. Returns (batch id, exhausted ids)."""
        retry, model_for, exhausted = {}, {}, []
        for cid in invalid:
            item = self.items[cid]
            nxt = next_model(item["task"], item["model"])
            if nxt is None:
                exhausted.append(cid)
            else:
                retry[cid] = {k: item[k] for k in ("task", "inputs", "meta")}
                model_for[cid] = nxt
        return self.submit(retry, model_for), exhausted
