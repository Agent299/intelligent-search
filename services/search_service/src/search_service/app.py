"""Search service (doc p.143-145, p.156-157).

    GET  /v1/search?q=&userId=&sessionId=&page=&pageSize=&filters=&sortBy=&debug=
    POST /v1/events   {"searchId", "type": "click"|"add_to_cart"|"purchase", "productId"}
    GET  /healthz, /metrics
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from aiokafka import AIOKafkaProducer
from fastapi import FastAPI, HTTPException, Query
from opensearchpy import AsyncOpenSearch
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from redis.asyncio import Redis

from common.config import get_settings
from common.embedding_service import EmbeddingService
from common.embeddings import warm_up
from common.schemas import EventType, SearchResponse, SortBy
from search_service.engine import EmptyQuery, SearchEngine, SearchRequest
from search_service.events import EventSink
from search_service.fallback import FallbackPath, upsert_smart_cache
from search_service.filters import FilterError, parse_filters
from search_service.gateway_client import GatewayClient
from search_service.retrieval import Retriever
from search_service.taxonomy import TaxonomyIndex
from search_service.vocab import build_spell_corrector

log = logging.getLogger(__name__)


class InteractionEvent(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    search_id: str
    type: EventType
    product_id: str


async def _start_producer(bootstrap: str) -> AIOKafkaProducer | None:
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap, linger_ms=50)
    try:
        await asyncio.wait_for(producer.start(), timeout=5)
        return producer
    except Exception:
        log.warning("Kafka unavailable: analytic events will be dropped", exc_info=True)
        return None


def create_app(engine: SearchEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resources = []
        if engine is None:
            s = get_settings()
            os_client = AsyncOpenSearch(s.opensearch_url, timeout=5)
            redis = Redis.from_url(s.redis_url)
            producer = await _start_producer(s.kafka_bootstrap)
            embeddings = EmbeddingService(redis)
            spell = await asyncio.to_thread(build_spell_corrector)
            await asyncio.to_thread(warm_up)  # load the model now, not on the first uncached query
            retriever = Retriever(os_client)
            fallback, gateway = None, None
            if s.fallback_enabled:
                gateway = GatewayClient(s.genai_gateway_url)
                taxonomy = await TaxonomyIndex.build(embeddings)
                fallback = FallbackPath(retriever, embeddings, gateway, taxonomy, redis, os_client,
                                        persist=upsert_smart_cache)
            app.state.engine = SearchEngine(retriever, embeddings, spell, EventSink(producer, redis), s, fallback)
            resources = [os_client, redis, producer, gateway]
        else:
            app.state.engine = engine
        yield
        for r in resources:
            if r is None:
                continue
            closer = getattr(r, "stop", None) or getattr(r, "aclose", None) or r.close
            await closer()

    app = FastAPI(title="Search service", lifespan=lifespan)
    Instrumentator().instrument(app).expose(app)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/search", response_model=SearchResponse, response_model_by_alias=True,
             response_model_exclude_none=True)
    async def search(
        q: Annotated[str, Query(min_length=1, max_length=200)],
        userId: str | None = None,  # noqa: N803 - public API uses camelCase (doc p.145)
        sessionId: str | None = None,  # noqa: N803
        page: Annotated[int, Query(ge=1, le=50)] = 1,
        pageSize: Annotated[int, Query(ge=1, le=50)] = 20,  # noqa: N803
        filters: str | None = None,
        sortBy: SortBy = SortBy.relevance,  # noqa: N803
        device: str | None = None,
        debug: bool = False,
    ) -> SearchResponse:
        try:
            req = SearchRequest(q=q, user_id=userId, session_id=sessionId, page=page, page_size=pageSize,
                                filters=parse_filters(filters), sort_by=sortBy, device_type=device, debug=debug)
            return await app.state.engine.search(req)
        except (FilterError, EmptyQuery) as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/v1/events", status_code=202)
    async def events(evt: InteractionEvent) -> dict:
        if evt.type is EventType.search:
            raise HTTPException(422, "search events are emitted by /v1/search")
        if not await app.state.engine.sink.interaction(evt.search_id, evt.type, evt.product_id):
            raise HTTPException(404, "unknown or expired searchId")
        return {"status": "accepted"}

    return app


app = create_app()
