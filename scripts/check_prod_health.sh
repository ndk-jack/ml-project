#!/usr/bin/env bash
set -euo pipefail

echo "== ml-project health =="
curl -fsS https://ml-project-production-794e.up.railway.app/api/v1/health | python -m json.tool

echo "== ml-project scored-events =="
curl -fsS "https://ml-project-production-794e.up.railway.app/api/v1/scored-events?limit=1" | python -m json.tool

echo "== ml-data-plane live =="
curl -fsS https://ml-data-plane-production.up.railway.app/health/live | python -m json.tool

echo "== ml-data-plane ready =="
curl -fsS https://ml-data-plane-production.up.railway.app/health/ready | python -m json.tool

echo "== ml-data-plane scheduler metric =="
curl -fsS https://ml-data-plane-production.up.railway.app/metrics | grep -E '^data_plane_scheduler_up '

echo "== ml-data-plane outcomes metrics =="
curl -fsS https://ml-data-plane-production.up.railway.app/metrics | grep -E '^outcomes_(tasks_total|batches_total|usgs_calls_total|candidates_total|run_duration_seconds_count|run_duration_seconds_sum)'
