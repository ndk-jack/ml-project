# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

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
