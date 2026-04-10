# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed
- Introduced `benchmark_v2` based on `dataset_v5_dedup.csv` for cleaner 7d / 30d benchmarking.
- Removed `label_365d` from the active benchmark scope because the target is heavily skewed toward the positive class and is less informative for early model iteration.
- Deduplicated perfectly redundant features in the new candidate dataset (`rate_*`, `ref_lat/ref_lon`, `moment_*` removed in favor of simpler equivalents).

### Added
- Added dataset cleaning rules and manifests for `dataset_v5_dedup`.
- Added comparable naive baseline metrics for benchmark evaluation.
- Added baseline vs challenger comparison script for MLflow runs.

### Changed
- Completed a first Optuna pass on `benchmark_v2` for `label_7d` and `label_30d`.
- Kept `label_365d` out of the active benchmark scope.
- Observed only marginal tuning gains versus the simple benchmarked LightGBM challenger, especially for `label_7d`.

### Notes
- `label_7d` appears close to a performance ceiling with the current feature set.
- `label_30d` benefits slightly from tuned hyperparameters, but gains remain modest.
- Next optimization work should focus on feature quality and feature selection rather than larger Optuna budgets.

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

- 2026-04-10: feedback loop MVP merged on feat/data-clean-rebuild-v4 (prediction_log, prediction_outcomes, batch delayed evaluation).
