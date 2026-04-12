#!/usr/bin/env bash
set -euo pipefail

echo "== ml-data-plane pipeline freshness =="
curl -fsS https://ml-data-plane-production.up.railway.app/metrics | grep -E "^data_plane_pipeline_last_(finished_unixtime|success)|^data_plane_pipeline_last_published_rows"

echo "== ml-data-plane scoring/outcomes focus =="
curl -fsS https://ml-data-plane-production.up.railway.app/metrics | grep -E 'pipeline_name="(scoring_pipeline|outcomes_pipeline)"'
