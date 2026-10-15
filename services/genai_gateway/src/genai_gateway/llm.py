"""Provider adapter. The gateway depends on the LLMCaller protocol only, so providers can be swapped
(doc p.158 'provider abstraction') and tests can inject a fake."""

from dataclasses import dataclass
from typing import Protocol

import anthropic

from common.config import get_settings


@dataclass
class LLMResult:
    text: str
    stop_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0


class TransientLLMError(Exception):
    """Timeouts, connection errors, 429 and 5xx: count against the model's circuit breaker."""


class LLMCaller(Protocol):
    async def __call__(self, *, model: str, system: str, user: str, schema: dict, effort: str,
                       max_tokens: int, timeout_s: float) -> LLMResult: ...


class AnthropicCaller:
    # No SDK-level retries: each retry would get a fresh timeout and overrun the task's time budget.
    # Failover is the gateway's job (next model in the chain / circuit breaker).
    def __init__(self, client: anthropic.AsyncAnthropic | None = None, max_retries: int = 0):
        self._client = client or anthropic.AsyncAnthropic(
            api_key=get_settings().anthropic_api_key or None, max_retries=max_retries)

    async def __call__(self, *, model: str, system: str, user: str, schema: dict, effort: str,
                       max_tokens: int, timeout_s: float) -> LLMResult:
        try:
            msg = await self._client.with_options(timeout=timeout_s).messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": user}],
            )
        except (anthropic.APITimeoutError, anthropic.APIConnectionError, anthropic.RateLimitError) as e:
            raise TransientLLMError(type(e).__name__) from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise TransientLLMError(f"HTTP {e.status_code}") from e
            raise  # 4xx other than 429 is a request bug, not an outage
        u = msg.usage
        return LLMResult(
            text=next((b.text for b in msg.content if b.type == "text"), ""),
            stop_reason=msg.stop_reason,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read=u.cache_read_input_tokens or 0,
            cache_write=u.cache_creation_input_tokens or 0,
        )
