"""GenAI gateway HTTP service.

    POST /v1/llm/{task}   {"inputs": {...}}  -> GatewayResult
    POST /v1/embed        {"texts": [...], "kind": "query"|"document"}
    GET  /v1/routing      task -> model chain (from routing.yaml)
    GET  /v1/breakers     circuit breaker state per model
    GET  /healthz, /metrics
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, ValidationError
from redis.asyncio import Redis

from common.config import get_settings
from common.embeddings import warm_up
from genai_gateway.core import Gateway, GatewayResult, GatewayUnavailable, UnknownTask
from genai_gateway.embed import EmbeddingService, Kind
from genai_gateway.llm import AnthropicCaller


class LLMRequest(BaseModel):
    inputs: dict[str, Any]


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)
    kind: Kind = "query"


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dim: int


def create_app(gateway: Gateway | None = None, embeddings: EmbeddingService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        redis = None
        if embeddings is None:
            redis = Redis.from_url(get_settings().redis_url)
            await asyncio.to_thread(warm_up)  # load the embedding model now, not on the first cache miss
        app.state.gateway = gateway or Gateway(caller=AnthropicCaller())
        app.state.embeddings = embeddings or EmbeddingService(redis)
        yield
        if redis is not None:
            await redis.aclose()

    app = FastAPI(title="GenAI gateway", lifespan=lifespan)
    Instrumentator().instrument(app).expose(app)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/routing")
    async def routing() -> dict:
        r = app.state.gateway.routing
        return {name: route.model_dump() for name, route in r.tasks.items()}

    @app.get("/v1/breakers")
    async def breakers() -> dict:
        return app.state.gateway.breaker_states()

    @app.post("/v1/llm/{task}", response_model=GatewayResult)
    async def llm(task: str, req: LLMRequest) -> GatewayResult:
        try:
            return await app.state.gateway.run(task, req.inputs)
        except UnknownTask as e:
            raise HTTPException(404, f"unknown task {task!r}") from e
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False)) from e
        except GatewayUnavailable as e:
            raise HTTPException(503, {"message": str(e), "attempts": [a.model_dump() for a in e.attempts]}) from e

    @app.post("/v1/embed", response_model=EmbedResponse)
    async def embed(req: EmbedRequest) -> EmbedResponse:
        vectors = await app.state.embeddings.embed(req.texts, req.kind)
        return EmbedResponse(vectors=vectors, dim=len(vectors[0]))

    return app


app = create_app()
