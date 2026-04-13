# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed
- Removed `insert_prediction_outcome_stub` call from the scoring path in `main.py`. Delayed outcome evaluation is now the sole responsibility of `ml-data-plane` (`outcomes_pipeline`).
- Deprecated `insert_prediction_outcome_stub` in `database.py` — logs a warning and returns `False`. Function kept for import compatibility; safe to remove after confirming no external callers.
- `evaluate_prediction_outcomes.py` now raises `RuntimeError` immediately to prevent accidental runs of the old evaluation job.

---

### Added
- Added feedback loop MVP with `prediction_log`, `prediction_outcomes`, and delayed evaluation snapshots.
- Added public API v1 endpoints for scored events:
  - `GET /api/v1/health`
  - `GET /api/v1/scored-events`
  - `GET /api/v1/scored-events/{event_id}`
- Added structured risk codes to the public API:
  - `risk_7d_code`
  - `risk_30d_code`
  - `risk_365d_code`
- Added scheduled auto-scoring job to keep `scored_events` fresh in production.
- Added serving metadata persistence on newly scored `scored_events` rows:
  - `model_version`
  - `benchmark_id`
  - `feature_set_version`
  - `dataset_version`
  - `mlflow_run_id`

### Changed
- Promoted `scored_events` as the product-facing read model for the frontend dashboard.
- Kept `prediction_log` as append-only internal truth for served predictions.
- Kept `prediction_outcomes` as delayed-label storage for 7d / 30d evaluation.
- Hardened the public scored-events feed against legacy invalid rows.
- Added `meta.rejected` to the public scored-events response.
- Standardized frontend consumption on the public API instead of direct dashboard reads from Supabase.

### Fixed
- Fixed Railway deployment packaging by aligning API package-relative imports.
- Fixed scheduled auto-scoring idempotence for prediction logging.
- Fixed stale dashboard feed by introducing scheduled auto-scoring in addition to rolling catalog refresh.
- Fixed frontend event identity by aligning the dashboard on `event_id`.

### Infra
- Corrected Railway service source to deploy from `main` instead of `feat/public-api-contract-v1`.
- Confirmed production scheduler registration for both:
  - `CatalogManager.refresh_rolling`
  - `auto_score`

### Docs
- Consolidated README to reflect the real production architecture, public API, feedback loop, and scheduled scoring behavior.
- Clarified backend / frontend repository ownership.

### Added
-

### Changed
-

### Fixed
-

### Infra
-

### Docs
-

## [v0.2.0-prod-stable] - 2026-04-08

### Added
- Added production runtime artifacts for the historical catalog and feature medians.
- Added richer `/health` reporting for rolling catalog, historical catalog, scorer readiness, and Supabase status.

### Changed
- Standardized the project runtime on Python 3.11.
- Moved API runtime artifacts to `data/external/`.
- Made the initial rolling refresh non-blocking at startup.

### Fixed
- Fixed historical catalog loading in production.
- Fixed rolling USGS refresh to cover the full 92-day M≥2 window with chunked fetch and deduplication.
- Fixed scorer median loading visibility and fallback tracking.
- Fixed Supabase env variable compatibility across `SUPABASE_SERVICE_KEY` and `SUPABASE_SERVICE_ROLE_KEY`.

### Infra
- Stabilized Railway deployment startup behavior and readiness.
- Tagged the stable production release as `v0.2.0-prod-stable`.

### Docs
- Updated README, CLAUDE, and API docstrings for Python 3.11 and Railway runtime artifacts.

## 2026-04-13
- documented minimal production monitoring contract in docs/monitoring_minimal_contract.md
