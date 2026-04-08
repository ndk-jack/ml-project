# CHANGELOG — Earthquake Sequence Forecasting

## [1.3.0] — 2026-04-08 — Production deployment
### Added
- src/api/database.py — Supabase persistence layer
- migrations/001_scored_events.sql — PostgreSQL schema + RLS
- Dockerfile + railway.toml — Railway deployment
- Lovable dashboard — https://earthquake-watch.lovable.app
### Infrastructure
- API: https://ml-project-production-794e.up.railway.app
- Docs: https://ml-project-production-794e.up.railway.app/docs
- DB: Supabase sfykwnhynwwuvientblh, table scored_events
- Frontend: https://earthquake-watch.lovable.app

## [1.2.0] — 2026-04 — Real-time scoring API
### Added
- FastAPI + APScheduler (USGS polling every 5 min)
- catalog_manager, feature_engine, scorer, database modules
- 71 features computed per event in real-time
### Fixed
- mag_std missing per window (60 vs 71 features)
- scorer uses model.feature_name() — no hardcoded lists

## [1.1.0] — 2026-04 — Multi-horizon models
- 7d ROC-AUC 0.8586 / 30d 0.8362 / 365d 0.9194
- accel_count, accel_energy, mag_excess features added
- Removed b_value_trend (NaN fill bug)

## [1.0.0] — 2026-04 — Dual-catalog pipeline
- Completeness bias fix: M>=4 + M>=2 dual catalog
- background_rate_yr = #1 feature across all horizons
- GEM faults + WSM + plate boundaries integrated
