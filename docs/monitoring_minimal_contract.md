# Monitoring minimal contract

## Canonical public URLs
- ml-project: https://ml-project-production-794e.up.railway.app
- ml-data-plane: https://ml-data-plane-production.up.railway.app

## Block 1 — service health
- GET /api/v1/health -> 200
- GET /health/live -> 200
- GET /health/ready -> 200
- metric data_plane_scheduler_up == 1

## Block 2 — pipeline freshness
- metric data_plane_pipeline_last_finished_unixtime{pipeline_name="scoring_pipeline"}
- metric data_plane_pipeline_last_finished_unixtime{pipeline_name="outcomes_pipeline"}
- metric data_plane_pipeline_last_success{pipeline_name="scoring_pipeline"} == 1
- metric data_plane_pipeline_last_success{pipeline_name="outcomes_pipeline"} == 1
- metric data_plane_pipeline_last_published_rows{pipeline_name="scoring_pipeline"}
- metric data_plane_pipeline_last_published_rows{pipeline_name="outcomes_pipeline"}

## Suggested alert thresholds
- scoring stale if last finished run > 90 min
- outcomes stale if last finished run > 48 h
- scheduler down if data_plane_scheduler_up != 1
- pipeline failed if data_plane_pipeline_last_success != 1
