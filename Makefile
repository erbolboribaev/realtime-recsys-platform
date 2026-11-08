COMPOSE = docker compose -f infra/compose/docker-compose.yml

.PHONY: install lint test up pipeline down logs backfill lake evaluate train rank publish api

install:
	pip install -e ".[dev,ml,rank,serve]"

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest -v

## Start infrastructure (Redpanda, Redis, Postgres, Qdrant, S3-compatible store) and create topics.
up:
	$(COMPOSE) up -d --wait redpanda redis postgres qdrant minio
	$(COMPOSE) up topics-init

## Start infrastructure plus the event simulator and the lake sink.
pipeline: up
	$(COMPOSE) --profile pipeline up -d --build simulator lake-sink feature-job

## Summarise the Parquet files the sink has written to the lake.
lake:
	python -m recsys.sink.lake_report

## Offline evaluation: popularity baseline vs ALS (no Docker needed).
evaluate:
	python -m recsys.ml.evaluate

## Train ALS, log to MLflow, publish item vectors to Qdrant (needs `make up`).
train:
	python -m recsys.ml.train --track --index

## Train the production ranker + ALS and write models/ (needs the [rank] extra); add --index for Qdrant.
publish:
	python -m recsys.ml.publish

## Run the recommendation API locally on http://localhost:8000 (needs make up + publish --index).
api:
	python -m uvicorn --factory recsys.serving.app:create_app_from_env --port 8000

## Second-stage ranker experiment (logistic by default, no extra dependencies).
rank:
	python -m recsys.ml.rank

logs:
	$(COMPOSE) --profile pipeline logs -f --tail=50

## Send 50,000 simulated historical events as fast as possible (needs `make up`).
backfill:
	python -m recsys.events.producer --mode backfill --max-events 50000

down:
	$(COMPOSE) --profile pipeline down -v
