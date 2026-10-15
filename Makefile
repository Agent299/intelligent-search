.PHONY: install up down migrate bootstrap ingest index enrich-submit enrich-status enrich-collect synth catalog-consumer pipeline pipeline-pilot pipeline-dry pipeline-schedule hot-list discovery rank-publish rank-train test lint gateway search

install:
	uv sync --all-extras

up:
	docker compose up -d --wait

down:
	docker compose down

migrate:
	uv run alembic upgrade head

bootstrap:
	uv run python -m common.bootstrap

# Phase 1: catalog ingestion (download -> load -> index; LLM enrichment then re-index)
# SCALE=1.0 for the full 50k sample; 0.2 (10k products) is the current default.
SCALE ?= 0.2
ingest:
	uv run python -m pipelines.ingestion.download --scale $(SCALE)
	uv run python -m pipelines.ingestion.load_catalog
	uv run python -m pipelines.ingestion.index

index:
	uv run python -m pipelines.ingestion.index

enrich-submit:
	uv run python -m pipelines.ingestion.enrich_llm submit $(if $(LIMIT),--limit $(LIMIT))

enrich-status:
	uv run python -m pipelines.ingestion.enrich_llm status

enrich-collect:
	uv run python -m pipelines.ingestion.enrich_llm collect

synth:
	uv run python -m pipelines.synthetic.generate --publish-recent 2000

catalog-consumer:
	uv run python -m pipelines.ingestion.catalog_updates consume

# Phase 5: offline query pipeline (aggregate -> enrich -> segment -> expand -> cache); LLM stages use the Batch API
pipeline:
	uv run python -m pipelines.offline.run all

pipeline-pilot:
	uv run python -m pipelines.offline.run all --limit $(or $(LIMIT),100)

# Phase 6: hot list -> S3 + list_update (search instances reload L1); warm L2 in Redis. Also the last step of `pipeline`
hot-list:
	uv run python -m pipelines.offline.run hot-list

# Phase 7: co-purchase mining -> complements per category (Haiku batch, only new/changed lists) -> Redis.
# Also part of `pipeline`. LIMIT=n for a pilot.
discovery:
	uv run python -m pipelines.offline.run co-purchase
	R=disc-$$(date -u +%Y%m%dT%H%M%S); \
	uv run python -m pipelines.offline.run discover $$R $(if $(LIMIT),--limit $(LIMIT)) && \
	uv run python -m pipelines.offline.run discover-collect $$R --wait

# Phase 8: replay the logs -> static_score in the index + query/user ranking features in Redis (also in `pipeline`)
rank-publish:
	uv run python -m pipelines.ranking.publish

# Phase 8: retrain the LightGBM re-ranker (models/ranker.txt + NDCG report in models/ranker.json); restart search
rank-train:
	uv run python -m pipelines.ranking.train

# Daily at 02:00 UTC; HOURLY=1 for every hour
pipeline-schedule:
	uv run python -m pipelines.offline.run schedule $(if $(HOURLY),--hourly)

pipeline-dry:
	uv run python -m pipelines.offline.run aggregate
	uv run python -m pipelines.offline.run enrich
	uv run python -m pipelines.offline.run segment dry-$$(date +%s) --dry-run

test:
	uv run pytest -q

lint:
	uv run ruff check .

gateway:
	uv run uvicorn genai_gateway.app:app --port 8001 --reload

search:
	uv run uvicorn search_service.app:app --port 8000 --reload
