SHELL := /bin/bash
export PATH := $(HOME)/.local/bin:$(PATH)

COMPOSE := docker compose -f infra/docker-compose.yml
UV := uv

.PHONY: setup demo ingest eval redteam voice-bench test test-unit lint fmt up down db-reset migrate ollama-up api ui stop help

help:
	@grep -E '^[a-z-]+:.*#' Makefile | sed 's/:.*#/ —/'

setup: # install deps, copy .env, start stack, run migrations, ensure ollama + model
	@command -v uv >/dev/null || (echo "installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	@test -f .env || cp .env.example .env
	$(UV) sync --all-extras
	$(COMPOSE) up -d --wait postgres langfuse-db langfuse
	$(UV) run python -m common.db
	./infra/ensure_ollama.sh
	@echo "setup complete."

up: # start postgres + langfuse
	$(COMPOSE) up -d --wait postgres langfuse-db langfuse

down: # stop the stack (keeps volumes)
	$(COMPOSE) down

db-reset: # drop and recreate the civiclens database (destructive)
	$(COMPOSE) exec postgres psql -U civiclens -d postgres -c "DROP DATABASE IF EXISTS civiclens WITH (FORCE)"
	$(COMPOSE) exec postgres psql -U civiclens -d postgres -c "CREATE DATABASE civiclens"
	$(COMPOSE) exec postgres psql -U civiclens -d civiclens -c "CREATE EXTENSION IF NOT EXISTS vector"
	$(UV) run python -m common.db

migrate: # apply pending migrations
	$(UV) run python -m common.db

ingest: # ingest the bundled sample corpus into Postgres, then embed + topic-tag
	$(UV) run python -m ingestion.cli samples
	$(UV) run python -m retrieval.index

ingest-live: # ingest real meetings from Legistar + YouTube (network required)
	$(UV) run python -m ingestion.cli live
	$(UV) run python -m retrieval.index

eval: # run the eval harness (golden dataset must exist; see evals/README)
	$(UV) run python -m evals.run

redteam: # safety eval: prompt-injection red-team (before/after) + PII precision/recall
	$(UV) run python -m safety.run_safety

voice-bench: # measure voice-turn latency percentiles (warmed, N turns)
	$(UV) run python -m voice.bench --turns 5

test: # run all tests (integration tests auto-skip if the stack is down)
	$(UV) run pytest -q

test-unit: # run unit tests only
	$(UV) run pytest -q -m "not integration"

lint: # ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: # auto-format
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

api: # run the FastAPI server (foreground)
	$(UV) run uvicorn api.main:app --host 127.0.0.1 --port 8000

ui: # run the Streamlit UI (foreground)
	$(UV) run streamlit run ui/app.py --server.port 8501

demo: setup ingest # end-to-end demo: stack up, samples ingested, api+ui running
	./infra/run_demo.sh
