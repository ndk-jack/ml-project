# Earthquake Sequence Forecasting

Production-oriented earthquake follow-up risk forecasting system.

The project combines:
- offline ML experimentation and benchmarking,
- a real-time FastAPI scoring service,
- a Supabase-backed feedback loop,
- a public API consumed by the frontend dashboard.

## Problem

Given an earthquake at time `t` and location `P`, estimate the probability that at least one significant nearby follow-up earthquake (`M≥5.0`) will occur within `R = 200 km` over three horizons:

- `7d`
- `30d`
- `365d`

Each horizon uses its own LightGBM model.

## Current production architecture

### Offline / experimentation
- Benchmark baseline: `benchmark_v2`
- Dataset: `dataset_v5_dedup`
- Primary compact feature set: `candidate_feature_set_v1`
- Experiment tracking: MLflow (currently local SQLite-backed setup)

### Online / production
- FastAPI service on Railway
- Supabase for product-facing persistence and feedback loop
- Frontend dashboard in separate repo: `ndk-jack/quakehub-live`

## Production data model

### `scored_events`
Product-facing table used by the frontend dashboard.

Purpose:
- latest scored events
- public API source
- idempotent per `event_id`

Behavior:
- upsert on `event_id`
- now stores serving metadata on newly scored rows:
  - `model_version`
  - `benchmark_id`
  - `feature_set_version`
  - `dataset_version`
  - `mlflow_run_id`

### `prediction_log`
Append-only internal truth for served predictions.

Purpose:
- audit trail of predictions actually served
- feedback loop source of truth
- future retraining / analysis support

Behavior:
- append-only for manual scoring endpoints
- idempotence guard for scheduled auto-score on `(event_id, model_version)` at application level

### `prediction_outcomes` (managed by ml-data-plane)
Delayed ground-truth table — **not written by this service**.

Outcome evaluation (USGS follow-up lookup, labelling, quality checks, publishing) is fully
handled by the `outcomes_pipeline` in `ml-data-plane`. This service no longer inserts stubs
into `prediction_outcomes` at scoring time.

### `model_eval_snapshots` (managed by ml-data-plane)
Aggregated evaluation snapshots built from prediction logs + delayed outcomes.

Purpose:
- monitor model quality over time
- support champion / challenger comparisons later

## Service responsibility split

| Responsibility | Service |
|---|---|
| Real-time scoring | **ml-project** (this repo) |
| `scored_events` upsert | **ml-project** |
| `prediction_log` append | **ml-project** |
| Delayed outcome evaluation | **ml-data-plane** (`outcomes_pipeline`) |
| `prediction_outcomes` writes | **ml-data-plane** |
| `model_eval_snapshots` writes | **ml-data-plane** |

## Public API

The frontend dashboard must use the public API, not direct table access.

### `GET /health`
Operational health endpoint.

Returns:
- model readiness
- rolling catalog freshness
- external artifact readiness
- scheduler state indicators

### `GET /api/v1/health`
Minimal public health endpoint.

Example response:

```json
{
  "status": "ok"
}
```

### `GET /api/v1/scored-events?limit=50`
Public scored events feed for the dashboard.

Example response:

```json
{
  "data": [
    {
      "event_id": "us6000spa5",
      "event_datetime": "2026-04-11T09:10:04.765000Z",
      "latitude": 39.9153,
      "longitude": 141.5227,
      "depth": 79.374,
      "magnitude": 4.3,
      "prob_7d": 0.443,
      "prob_30d": 0.6169,
      "prob_365d": 0.9919,
      "risk_7d": "🟡 Modéré",
      "risk_30d": "🟠 Élevé",
      "risk_365d": "🔴 Très élevé",
      "risk_7d_code": "moderate",
      "risk_30d_code": "high",
      "risk_365d_code": "very_high",
      "scored_at": "2026-04-11T09:50:12.636086Z",
      "model_version": "benchmark_v2__compact22__bestparams_v2",
      "benchmark_id": "benchmark_v2",
      "feature_set_version": "candidate_feature_set_v1",
      "dataset_version": "dataset_v5_dedup"
    }
  ],
  "meta": {
    "count": 1,
    "limit": 50,
    "rejected": 0
  }
}
```

### `GET /api/v1/scored-events/{event_id}`
Returns one scored event from the public feed.

## Risk representation

The public API exposes both:
- human-readable labels:
  - `risk_7d`
  - `risk_30d`
  - `risk_365d`
- structured codes:
  - `risk_7d_code`
  - `risk_30d_code`
  - `risk_365d_code`

Stable codes:
- `very_low`
- `low`
- `moderate`
- `high`
- `very_high`

Frontend logic must rely on `*_code`, not on parsing the decorated label strings.

## Live scoring behavior

### Rolling catalog refresh
The service refreshes the rolling USGS catalog on a schedule.

### Auto-score job
A scheduled auto-score job periodically:
1. fetches the most recent USGS events,
2. scores the latest events,
3. persists dashboard-facing rows into `scored_events`,
4. writes append-only audit rows into `prediction_log` when appropriate,
5. creates delayed outcome stubs in `prediction_outcomes`.

This is what keeps the dashboard feed fresh in production.

## Feedback loop MVP

Current MVP design:
- `scored_events` = product-facing read model
- `prediction_log` = append-only prediction truth
- `prediction_outcomes` = delayed labels
- `model_eval_snapshots` = aggregated delayed evaluation output

This design keeps the frontend simple while preserving auditability and future retraining paths.

## Runtime artifacts required by the API

Production API expects these runtime files:

- `data/external/historical_catalog_pre2010_m3.csv.gz`
- `data/external/feature_medians_v3.json`

The rolling catalog is built at runtime from USGS.

## ML benchmark baseline

Official benchmark:
- `benchmark_v2`

Current production serving metadata:
- `model_version = benchmark_v2__compact22__bestparams_v2`
- `benchmark_id = benchmark_v2`
- `feature_set_version = candidate_feature_set_v1`
- `dataset_version = dataset_v5_dedup`

## Local development

### Run the API

```bash
source .venv/bin/activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Open MLflow UI locally

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:////$(pwd)/mlflow.db --host 127.0.0.1 --port 5001
```

## Repository structure

```text
ml-project/
├── src/
│   ├── api/                 # FastAPI app, public API, scheduler, scoring logic
│   ├── jobs/                # Delayed outcome evaluation jobs
│   ├── labels.py
│   ├── features.py
│   ├── train_multi_horizon.py
│   └── ...
├── data/
├── models/
├── experiments/
├── migrations/
└── reports/
```

## Repo scope

This repository owns:
- backend
- ML pipelines
- jobs
- Supabase integration
- public API

Frontend lives in:
- https://github.com/ndk-jack/quakehub-live