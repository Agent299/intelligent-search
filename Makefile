.PHONY: install up down migrate bootstrap ingest index enrich-submit enrich-status enrich-collect synth catalog-consumer pipeline pipeline-pilot pipeline-dry test lint gateway search

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
