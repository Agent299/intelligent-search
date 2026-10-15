"""Analytic event emission (doc p.166) and click attribution.

Search must never fail or slow down because analytics is unavailable: Kafka sends are fire-and-forget and
every failure is counted, not raised.
"""

import json
import logging
import uuid
from datetime import UTC, datetime

from aiokafka import AIOKafkaProducer
from prometheus_client import Counter
from redis.asyncio import Redis

from common.schemas import AnalyticEvent, EventType, SearchType
from common.topics import QUERY_LOGS

log = logging.getLogger(__name__)
EMIT_ERRORS = Counter("search_event_emit_errors_total", "Analytic events that could not be sent", ["sink"])
CONTEXT_TTL_S = 3600


class EventSink:
    def __init__(self, producer: AIOKafkaProducer | None, redis: Redis | None):
        self.producer = producer
        self.redis = redis

    async def search(self, *, search_id: str, user_id: str | None, session_id: str | None, raw_query: str,
                     filters: dict[str, str], search_type: SearchType, displayed: list[str],
                     device_type: str | None) -> None:
        event = AnalyticEvent(
            event_id=uuid.uuid4().hex, event_type=EventType.search, search_id=search_id,
            event_timestamp=datetime.now(UTC), user_id=user_id, session_id=session_id, raw_query=raw_query,
            filters_applied=filters, search_type=search_type, displayed_product_ids=displayed,
            device_type=device_type,
        )
        await self._send(event)
        if self.redis is not None:
            ctx = {"user_id": user_id, "session_id": session_id, "raw_query": raw_query, "displayed": displayed,
                   "device_type": device_type, "search_type": search_type.value}
            try:
                await self.redis.set(f"search:{search_id}", json.dumps(ctx), ex=CONTEXT_TTL_S)
            except Exception:
                EMIT_ERRORS.labels("redis").inc()

    async def interaction(self, search_id: str, event_type: EventType, product_id: str) -> dict | None:
        """Record a click / add-to-cart / purchase against the search that displayed the product.
        Returns the search context, or None when it has expired or is unknown."""
        ctx = None
        if self.redis is not None:
            try:
                raw = await self.redis.get(f"search:{search_id}")
                ctx = json.loads(raw) if raw else None
            except Exception:
                EMIT_ERRORS.labels("redis").inc()
        if ctx is None:
            return None
        event = AnalyticEvent(
            event_id=uuid.uuid4().hex, event_type=event_type, search_id=search_id,
            event_timestamp=datetime.now(UTC), user_id=ctx["user_id"], session_id=ctx["session_id"],
            raw_query=ctx["raw_query"], search_type=SearchType(ctx["search_type"]),
            clicked_product_id=product_id if event_type is EventType.click else None,
            added_to_cart_product_id=product_id if event_type is EventType.add_to_cart else None,
            purchased_product_ids=[product_id] if event_type is EventType.purchase else [],
            device_type=ctx["device_type"],
        )
        await self._send(event)
        return ctx

    async def _send(self, event: AnalyticEvent) -> None:
        if self.producer is None:
            return
        try:
            key = (event.session_id or event.search_id or "").encode()
            await self.producer.send(QUERY_LOGS, event.model_dump_json().encode(), key=key)  # buffered, not awaited
        except Exception:
            EMIT_ERRORS.labels("kafka").inc()
            log.warning("failed to emit analytic event", exc_info=True)
