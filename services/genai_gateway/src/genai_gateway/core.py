"""GenAI gateway core (doc p.157-158): provider abstraction, task-based model routing, resiliency
(circuit breakers, time budgets, failover) and cost accounting.

Two entry points:
- Gateway.run(task, inputs): synchronous call that walks the task's model chain.
- build_batch_request(...) / next_model(...): for offline pipelines, which submit through the
  Message Batches API (50% price) using the same prompts, schemas and routing, then escalate
  invalid results to the next model in a follow-up batch.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ValidationError

from genai_gateway import metrics
from genai_gateway.breaker import CircuitBreaker, State
from genai_gateway.config import Routing, load_routing
from genai_gateway.llm import LLMCaller, LLMResult, TransientLLMError
from genai_gateway.tasks import TASKS, TaskSpec

_BREAKER_GAUGE = {State.closed: 0, State.half_open: 1, State.open: 2}


class Attempt(BaseModel):
    model: str
    outcome: str  # ok | invalid | timeout | error | refusal | truncated | skipped_open
    latency_ms: float = 0.0
    reason: str | None = None


class GatewayResult(BaseModel):
    task: str
    model: str
    valid: bool
    output: dict[str, Any]
    escalated: bool
    attempts: list[Attempt]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class GatewayUnavailable(Exception):
    def __init__(self, task: str, attempts: list[Attempt]):
        super().__init__(f"no model in the {task!r} chain produced a result")
        self.attempts = attempts


class UnknownTask(KeyError):
    pass


@dataclass
class Gateway:
    caller: LLMCaller
    routing: Routing = field(default_factory=load_routing)
    failure_threshold: int = 5
    reset_after_s: float = 30.0
    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    def breaker(self, model: str) -> CircuitBreaker:
        if model not in self.breakers:
            self.breakers[model] = CircuitBreaker(self.failure_threshold, self.reset_after_s)
        return self.breakers[model]

    def breaker_states(self) -> dict[str, str]:
        return {m: self.breaker(m).state.value for m in self.routing.models}

    async def run(self, task: str, inputs: dict[str, Any] | BaseModel) -> GatewayResult:
        if task not in TASKS or task not in self.routing.tasks:
            raise UnknownTask(task)
        spec, route = TASKS[task], self.routing.tasks[task]
        inp = inputs if isinstance(inputs, BaseModel) else spec.input_model.model_validate(inputs)
        system, user, schema = spec.system_prompt, spec.user_message(inp), spec.output_schema

        started = time.monotonic()
        deadline = started + route.timeout_s
        attempts: list[Attempt] = []
        best: tuple[str, dict] | None = None  # last parsed-but-invalid output, returned if nothing better
        tokens_in = tokens_out = 0
        cost = 0.0

        for i, model in enumerate(route.chain):
            nxt = route.chain[i + 1] if i + 1 < len(route.chain) else None
            breaker = self.breaker(model)
            if not breaker.allow():
                attempts.append(Attempt(model=model, outcome="skipped_open"))
                metrics.REQUESTS.labels(task, model, "skipped_open").inc()
                self._escalate(task, model, nxt, "breaker_open")
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.05:
                break

            t0 = time.monotonic()
            try:
                res = await self.caller(model=model, system=system, user=user, schema=schema,
                                        effort=route.effort, max_tokens=spec.max_tokens, timeout_s=remaining)
            except TransientLLMError as e:
                breaker.record_failure()
                self._record(task, model, attempts, "error", t0, str(e))
                self._escalate(task, model, nxt, "error")
                continue
            finally:
                metrics.BREAKER.labels(model).set(_BREAKER_GAUGE[breaker.state])
            breaker.record_success()  # the provider answered; outcome below is about content
            metrics.BREAKER.labels(model).set(_BREAKER_GAUGE[breaker.state])

            tokens_in += res.input_tokens + res.cache_read + res.cache_write
            tokens_out += res.output_tokens
            cost += self._account(task, model, res)

            if res.stop_reason in ("refusal", "max_tokens"):
                outcome = "refusal" if res.stop_reason == "refusal" else "truncated"
                self._record(task, model, attempts, outcome, t0)
                self._escalate(task, model, nxt, outcome)
                continue
            try:
                out = spec.output_model.model_validate_json(res.text)
            except ValidationError as e:
                self._record(task, model, attempts, "invalid", t0, f"schema: {e.error_count()} errors")
                self._escalate(task, model, nxt, "schema")
                continue

            check = spec.validate(inp, out)
            if check.ok or not route.escalate_on_invalid or nxt is None:
                self._record(task, model, attempts, "ok" if check.ok else "invalid", t0, check.reason)
                return self._result(task, model, check.ok, check.output, attempts, tokens_in, tokens_out, cost,
                                    started)
            best = (model, check.output)
            self._record(task, model, attempts, "invalid", t0, check.reason)
            self._escalate(task, model, nxt, "invalid")

        if best is not None:
            return self._result(task, best[0], False, best[1], attempts, tokens_in, tokens_out, cost, started)
        raise GatewayUnavailable(task, attempts)

    # -- helpers --

    def _account(self, task: str, model: str, res: LLMResult) -> float:
        for kind, n in (("input", res.input_tokens), ("output", res.output_tokens),
                        ("cache_read", res.cache_read), ("cache_write", res.cache_write)):
            if n:
                metrics.TOKENS.labels(task, model, kind).inc(n)
        cost = self.routing.cost_usd(model, res.input_tokens, res.output_tokens, res.cache_read, res.cache_write)
        metrics.COST.labels(task, model).inc(cost)
        return cost

    @staticmethod
    def _record(task: str, model: str, attempts: list[Attempt], outcome: str, t0: float,
                reason: str | None = None) -> None:
        elapsed = time.monotonic() - t0
        metrics.REQUESTS.labels(task, model, outcome).inc()
        metrics.LATENCY.labels(task, model).observe(elapsed)
        attempts.append(Attempt(model=model, outcome=outcome, latency_ms=round(elapsed * 1000, 1), reason=reason))

    @staticmethod
    def _escalate(task: str, model: str, nxt: str | None, reason: str) -> None:
        if nxt is not None:
            metrics.ESCALATIONS.labels(task, model, nxt, reason).inc()

    @staticmethod
    def _result(task, model, valid, output, attempts, tokens_in, tokens_out, cost, started) -> GatewayResult:
        return GatewayResult(
            task=task, model=model, valid=valid, output=output,
            escalated=any(a.outcome != "ok" for a in attempts),
            attempts=attempts, input_tokens=tokens_in, output_tokens=tokens_out, cost_usd=round(cost, 6),
            latency_ms=round((time.monotonic() - started) * 1000, 1),
        )


# ---- offline / Batch API helpers ----

def task_spec(task: str) -> TaskSpec:
    if task not in TASKS:
        raise UnknownTask(task)
    return TASKS[task]


def next_model(task: str, current: str | None = None, routing: Routing | None = None) -> str | None:
    """First model in the task's chain, or the one after `current` (None when the chain is exhausted)."""
    chain = (routing or load_routing()).tasks[task].chain
    if current is None:
        return chain[0]
    i = chain.index(current)
    return chain[i + 1] if i + 1 < len(chain) else None


def build_batch_request(task: str, custom_id: str, inputs: dict[str, Any] | BaseModel,
                        model: str | None = None, routing: Routing | None = None) -> Request:
    routing = routing or load_routing()
    spec, route = task_spec(task), routing.tasks[task]
    inp = inputs if isinstance(inputs, BaseModel) else spec.input_model.model_validate(inputs)
    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=model or route.chain[0],
            max_tokens=spec.max_tokens,
            system=[{"type": "text", "text": spec.system_prompt, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": route.effort, "format": {"type": "json_schema", "schema": spec.output_schema}},
            messages=[{"role": "user", "content": spec.user_message(inp)}],
        ),
    )


def parse_and_validate(task: str, inputs: dict[str, Any] | BaseModel, text: str) -> tuple[bool, dict, str | None]:
    """Apply the same schema check + guardrail to a batch result that Gateway.run applies online."""
    spec = task_spec(task)
    inp = inputs if isinstance(inputs, BaseModel) else spec.input_model.model_validate(inputs)
    try:
        out = spec.output_model.model_validate_json(text)
    except ValidationError as e:
        return False, {}, f"schema: {e.error_count()} errors"
    check = spec.validate(inp, out)
    return check.ok, check.output, check.reason
