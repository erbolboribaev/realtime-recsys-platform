# Real-Time Recommendation Platform

A production-style recommendation system: user events stream through Kafka, features are
computed online and offline from one definition, a two-stage model (vector retrieval +
LightGBM ranker) serves recommendations under a latency budget, and an LLM explains each
recommendation asynchronously. Everything is containerized, deployed with Kubernetes and
Terraform, and measured with offline metrics and online A/B tests.

> **Status: Phases 1 to 4 of 6 done (LLM layer); Phase 5 (frontend) next.** See [Roadmap](#roadmap).

## Architecture

```
Frontend ──► Recommendation API ──► Qdrant (candidates), Redis (features)
                  │   ▲
                  │   └── SSE explanations ── LLM worker ──► LLM gateway ──► provider
                  ▼
               Kafka (user-events, impressions, explain-requests)
                  ├──► Stream job ──► Redis (online features)
                  └──► Lake sink ──► S3 / MinIO (Parquet) ──► training pipeline ──► MLflow ──► API
```

Fast path (request time) and slow path (LLM explanations) are separate on purpose: a slow
LLM never delays a recommendation.

## What works today (Phase 1)

- **Event contract** (`recsys.events.schema`): validated, timezone-aware user events.
- **Traffic simulator** (`recsys.events.simulator`): deterministic, seeded; users have
  category tastes, items follow a long-tail popularity curve, sessions form a
  view → add-to-cart → purchase funnel. The taste signal is what personalisation must learn.
- **Kafka producer** (`recsys.events.producer`): realtime (paced) and backfill modes.
- **Lake sink** (`recsys.sink`): Kafka → validated → buffered → date-partitioned Parquet on
  S3-compatible storage. Offsets are committed only after files are written
  (at-least-once; readers deduplicate on `event_id`).
- **Local stack** (`infra/compose`): Redpanda, Redis, PostgreSQL, Qdrant and an S3-compatible
  object store (see the note below on the storage image).
- **CI** (GitHub Actions): lint, tests, compose file validation.

## Offline models (Phase 2, part 1)

- **Dataset** (`recsys.ml.dataset`): deduplicates on `event_id`, weights events
  (view 1, add-to-cart 3, purchase 5), and splits **by time**: train on the past, test on
  the future. Items a user already touched are excluded from the targets.
- **Metrics** (`recsys.ml.metrics`): Recall@K and NDCG@K.
- **Baseline** (`recsys.ml.baselines`): popularity, the bar every model must clear.
- **ALS** (`recsys.ml.als`): implicit-feedback matrix factorisation in plain NumPy.

```bash
make evaluate        # python -m recsys.ml.evaluate  (simulated events, no Docker)
python -m recsys.ml.evaluate --source lake   # same, on the Parquet lake
```

Reference run (100,000 simulated events, 1,000 users, 2,000 items):

| Model | Recall@50 | NDCG@50 |
|---|---|---|
| Popularity | 0.1725 | 0.1046 |
| ALS (32 factors) | 0.2389 | 0.1564 |

The data is simulated, so these numbers show that the model learns the simulated tastes,
not how it would do on real traffic.

- **Retrieval** (`recsys.ml.index`, `recsys.ml.retrieval`): item vectors live in a Qdrant
  collection (dot-product distance, same as the ALS score); an exact NumPy index is the
  reference. Retrieval through the index must equal the model's own ranking.
- **Tracking** (`recsys.ml.tracking`): parameters and metrics go to MLflow (local `./mlruns`).

```bash
make train           # ALS + MLflow run + Qdrant index + agreement check (needs make up)
mlflow ui            # browse runs at http://localhost:5000 (run from the project folder)
```

- **Online features** (`recsys.features`): a Kafka consumer keeps per-user and per-item
  counters, each user's recent items and a trending list in Redis. Redelivered events are
  ignored (`SET NX` on `event_id`) and offsets are committed after each batch is applied.

Feast is deferred: with one online store and one feature set it adds infrastructure
without solving a problem the project has yet.

## Ranking (Phase 3, part 1)

ALS retrieves 100 candidates per user; a second-stage model re-orders them using extra
signals (ALS score and rank, item popularity and conversion rates, user activity,
similarity to the user's recent items). Training labels come from a later time period than
the features, and the final test uses a still later one, so nothing leaks.

```bash
make rank                                   # logistic ranker, no extra dependencies
pip install -e ".[rank]"                    # LightGBM + ONNX export
python -m recsys.ml.rank --model lightgbm --onnx models/ranker.onnx
```

The gate is NDCG@10 above plain ALS ordering on the held-out period.

Offline result (simulated data, NDCG@10): ALS 0.1106, logistic 0.1163, LambdaRank 0.1213,
LightGBM (binary, regularised) **0.1271**. Per-feature ablation showed that item
popularity and similarity to the user's recent items carry the signal; conversion rates and
per-user activity were noise, and an unregularised tree model on them did *worse* than ALS.
The ranker's headroom is capped by retrieval: the 100 candidates contain only 35% of the
items users touched next.

## Serving (Phase 3, part 2)

```
GET /recommend?user_id=user_00001&k=10
```

Flow: user vector -> Qdrant candidates (minus items already seen) -> features from Redis
counters -> ONNX ranker -> top k. Users are split deterministically into `control` (ALS
order) and `treatment` (ranker order); every response is written to the `impressions` topic
with its `request_id`. Unknown users get the trending list. Ranker features come from one
shared module used by both training and serving, and a test checks that Redis-derived
features equal the training features.

### Online A/B experiment (simulated users)

```bash
python -m recsys.experiments.simulate_ab --rounds 5     # API and Redis must be up
```

Simulated users request recommendations, notice higher positions more often, and click
items they would truly like (the simulator's hidden tastes). Clicks flow back into Redis so
later requests see the new history. Control (ALS order) and treatment (ranker) are compared
on **per-user** click rates with a z-test and a 95% confidence interval; repeated requests
by one user are averaged first because they are correlated. The click model is an assumption
(`--click-scale`; at the default a random list gets ~1% of requests clicked, a popularity
list ~18%, a perfect-knowledge list ~30%), so the result says whether the machinery detects
an effect inside the simulated world, not how real shoppers would behave.

#### Results and what they mean

| Check | Ranker vs ALS | Notes |
|---|---|---|
| Offline NDCG@10 (held-out period) | +14.9% (0.1271 vs 0.1106) | simulated events |
| Expected clicks per list, clean state (`diagnose_ab`) | +14.9% (CI +0.6 to +1.8 pp, p=0.0001) | no sampling noise, no feedback |
| Sampled clicks with feedback, 5 rounds | +2.1%, p=0.83 | ~500 users per arm: too noisy to see 15% |
| Sampled clicks with feedback, 40 rounds | -8.8%, p=0.04 | click rate fell 7.8% -> 5.7% as history filled |
| Sampled clicks with feedback, 10 more rounds (state already used) | -0.9%, p=0.92 | no difference |

Honest reading: on a clean state the ranker's lists are better under the click model, and
the noise-free check agrees with the offline metric. The sampled experiment cannot confirm
that on its own (about +-1.5 pp at 5 rounds) and, once simulated clicks pile up in Redis,
the ranker's advantage disappears or reverses: the same lists scored *11.8% worse* than
ALS on the polluted state before it was reset. Our working explanation, not yet tested, is
a feedback loop: clicks inflate item counters and recent-item windows, which the ranker
reads (`item_pop_log`, `recent_similarity`) and ALS does not. Fix candidates: freeze
counters during an experiment, decay them, or train the ranker on its own logged
impressions. Reset state before each experiment (`down -v`, backfill, `publish --index`)
and run `diagnose_ab` first: it takes seconds and has far more statistical power than
sampled clicks.

```bash
make up && make pipeline                      # stack + live events -> Redis counters
python -m recsys.events.producer --mode backfill --max-events 100000   # fill counters
python -m recsys.ml.publish --index           # train, write models/, fill Qdrant
make api                                      # http://localhost:8000/docs
# or in Docker:
docker compose -f infra/compose/docker-compose.yml --profile serve up -d --build api
curl "http://localhost:8000/recommend?user_id=user_00001&k=5"
```

## Quick start

Requires Docker and Python 3.12+.

```bash
python -m venv .venv && source .venv/bin/activate
make install          # package + dev tools + ML extras (Qdrant client, MLflow)
make test             # unit tests (no Docker needed)

make up               # infrastructure + topics
make pipeline         # + simulator and lake sink
make logs
```

Consoles: Redpanda <http://localhost:8080>, Qdrant <http://localhost:6333/dashboard>, and the
object store console on <http://localhost:9001> if the chosen image provides one
(`minioadmin` / `minioadmin`, local development only).

Check what reached the lake without any console (with `make pipeline` running):

```bash
make lake             # python -m recsys.sink.lake_report
```

**About the storage image.** The upstream MinIO images were removed from Docker Hub and Quay
in 2026, so the compose file uses a community MinIO fork (`pgsty/silo`) as a drop-in
replacement. Any S3-compatible image that accepts the same `server /data` command and
`MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` variables works: `S3_IMAGE=<image> make up`.
The lake bucket is created by the sink itself, so no init container is needed.

Load a historical dataset quickly (needs `make up`):

```bash
make backfill         # 50,000 simulated events, original timestamps
```

### Phase 1 gate

- [ ] `make up` and `make pipeline` start cleanly
- [ ] `make lake` shows Parquet files and a growing row count under
      `recsys-lake/raw/user-events/dt=YYYY-MM-DD/`
- [ ] CI is green

## LLM explanations

Each recommendation can come with a short "why these?" text, produced off the request path:

```
GET /recommend?user_id=...&explain=true     -> items + explain_url, request queued in Kafka
explain-worker (Kafka -> LLM -> Redis)       -> stores the explanation under the request id
GET /explain/{request_id}                    -> Server-Sent Events: pending ... explanation
```

- **Gateway** (`recsys.llm.gateway`): one entry point for every model call, with retries and
  backoff, a Redis response cache, and a daily cost cap (`LLM_DAILY_BUDGET_USD`). The provider
  sits behind a small interface; Groq and any OpenAI-compatible server work through settings
  (`LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_PRICE_*`).
- **Guardrails** (`recsys.llm.explain`): the model sees only three suggested products and the
  shopper's browsed categories; answers that invent a price, a product, or run too long are
  dropped. When the model is down, over budget, or rejected, a template explanation is served,
  so the product always has one.
- **Evals** (`python -m recsys.llm.evals --n 30 --seed 2`): scores what the model said on fixed
  cases (guardrail acceptance, on-topic, no unsupported claims, at most three products) and
  exits non-zero below 90%. History: the first prompt scored 70% on unsupported claims
  ("great choices"), a stricter prompt fixed that, and a model that ignored "at most three
  products" (73%) was fixed by showing it only three. The cases are simulated and the claim
  check is keyword based, so this catches known failures and regressions, not every bad answer.

If the worker is down or slow, the SSE stream ends with a template explanation instead of a
timeout (`source: "template"`), so the shopper always gets text.

The evals also run against the real model weekly and on demand in GitHub Actions
(`.github/workflows/llm-evals.yml`, needs the `GROQ_API_KEY` secret); the normal CI run tests
the eval harness with fake models and needs no key.

Run it (API, Redis and Kafka up; the key goes in `.env`, which is git-ignored):

```bash
python -m recsys.llm.smoke                   # key and cache check
python -m recsys.llm.worker                  # or: docker compose ... --profile llm up -d explain-worker
curl "http://127.0.0.1:8000/recommend?user_id=user_00001&explain=true"
curl -N "http://127.0.0.1:8000/explain/<request_id>"
```

## Storefront

The shop and the dashboard are a small TypeScript app built with Vite (sources in `web/src`, no
framework). The API serves the built pages at `/`. Pick a shopper, press View, Cart or Buy under a
product, and the list updates a few seconds later. A chat panel shows why the list was chosen,
using the explanation stream.

```bash
cd web && npm install && npm run build      # once, and after changing the frontend
cd .. && uvicorn --factory recsys.serving.app:create_app_from_env --port 8000
# open http://127.0.0.1:8000
```

For frontend work run `npm run dev` in `web/` instead. Vite serves the pages with hot reload on
http://localhost:5173 and forwards the API calls to the backend on port 8000 (`API_URL` changes
that). Add `?mock=1` to run the page on made-up data with no backend at all.

Each action is sent to `POST /events` and lands on the `user-events` topic with the id of the
list it came from, so A/B analysis can credit it to the right variant. Product names and prices
come from `GET /items?ids=...`. `?shopper=user_00009` opens the page as a given shopper.

The response shapes live in `web/src/types.ts` and the texts in `web/src/locales`. English and
Uzbek must have the same keys, which the compiler enforces. Checks, all run in CI:

```bash
cd web
npm run typecheck   # tsc --noEmit, strict
npm test            # unit tests (node:test, run on the TypeScript files directly)
npm run build       # type-check, then the production bundle in web/dist
```

## Dashboard

`http://127.0.0.1:8000/admin.html` shows the platform's state and refreshes every 3 seconds:

- whether the API, Redis, the feature job and the explanation worker are running (the two jobs
  write a heartbeat key to Redis that expires after 30 seconds, so a dead job turns red by itself),
- lists served per minute and response time (typical and slowest 5%) over the last 5 minutes,
- per variant: lists shown, actions taken and the action rate. The comparison between the
  baseline and the new ranker appears only after 30 lists per variant, so a few clicks never read
  as a result,
- AI answers (written by the model, template, reused from cache) and today's spend against the cap.

The numbers come from `GET /admin/stats`. Traffic figures are kept in the API process, so with
several API replicas each would report its own share. Set `ADMIN_TOKEN` to require an
`X-Admin-Token` header. The dashboard then asks for the token on a small sign-in form and keeps
it only for the browser session. The shop does not link to the dashboard. With no token the route
is open, which is fine for a local demo and not for a public deployment.

## End-to-end test

`tests/test_e2e_storefront.py` starts the real API app and the real recommendation service with an
in-memory Redis, opens the shop in headless Chromium, buys a product and checks that the event is
published with the list's id and that the bought product leaves the next list. A second test
opens the dashboard and checks that it reflects the traffic and notices a stopped job.

```bash
cd web && npm install && npm run build && cd ..   # the test opens the built pages
pip install playwright && playwright install chromium
pytest tests/test_e2e_storefront.py -v
```

Without Playwright the tests are skipped; CI sets `E2E_REQUIRED=1` so a missing browser fails.

## Roadmap

| # | Phase | Gate |
|---|---|---|
| 1 | Foundation: repo, local stack, CI, simulator, Kafka → lake | `make up` works, events land in the lake, CI green |
| 2 | Features and retrieval: ALS, Qdrant, MLflow, Redis feature job (done); Feast deferred | Recall@50 above the popularity baseline (met offline) |
| 3 | Ranker and API: LightGBM (ONNX), FastAPI, A/B groups, impressions (done) | NDCG@10 above ALS (met: 0.1271 vs 0.1106), p95 under 100 ms (met: 43.7 ms) |
| 4 | LLM layer: gateway, worker, SSE, evals, cost cap (done) | Evals pass (met: 100% on 30 unseen cases), daily cost within cap (met: ~$0.00005 per explanation, hard cap enforced) |
| 5 | Frontend and dashboard: storefront, admin dashboard, end-to-end test (done) | End-to-end test passes, dashboard shows live data |
| 6 | Kubernetes and Terraform: Helm, Argo CD, KEDA, observability, canary | Full stack on kind, 200 req/s load test |

## Project layout

```
src/recsys/
  config.py          settings from environment variables
  events/            schema, catalog, simulator, Kafka producer
  sink/              buffering, Parquet writer, Kafka consumer, lake report
  ml/                dataset + time split, metrics, baseline, ALS, index, retrieval, tracking,
                     shared feature logic, rankers, publish
  serving/           model bundle, Redis feature readers, A/B, recommendation service, API, storefront
web/                 shop page and admin dashboard (TypeScript, Vite) and their tests
infra/
  compose/           local Docker Compose stack
  docker/            application image
tests/               pytest suite
```
