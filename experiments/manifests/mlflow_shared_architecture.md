# MLflow shared architecture

## Purpose

Move MLflow from a local single-machine setup to a shared team tracking service for offline experimentation, while keeping production serving deterministic and manually controlled.

## Current state

Today, MLflow tracking is local and SQLite-backed via `mlflow.db`.

This works for solo experimentation but is not a good collaboration setup:
- runs are not shared by default
- lineage is machine-local
- auditability is fragmented
- comparison across experiments is harder than necessary

## Target decision

MLflow becomes a **shared offline tracking layer**.

It does **not** become the production control plane.

Production serving remains controlled by:
- GitHub-tracked code
- explicit runtime artifacts
- explicit serving metadata
- manual promotion decisions

## Scope of MLflow shared

Shared MLflow is used for:
- experiment tracking
- params / metrics logging
- artifact logging
- benchmark comparison
- calibration artifact storage
- run-level lineage for offline work

Shared MLflow is **not** used for:
- automatic production promotion
- automatic champion selection in the API runtime
- direct runtime model lookup from the serving API

## Source of truth split

### Offline truth
MLflow is the source of truth for:
- experiment runs
- run metrics
- run params
- training artifacts
- calibration artifacts
- run tags

### Production truth
Production serving truth remains outside MLflow:
- deployed API code version
- runtime model files actually loaded by the API
- serving metadata written into product tables
- manual release decision

## Mandatory tags and metadata

Every new MLflow run must log at minimum:
- `benchmark_id`
- `dataset_version`
- `dataset_path`
- `feature_set_version`
- `split_version`
- `git_sha`
- `run_kind`
- `target`
- `model_family`

Recommended additional tags:
- `candidate_status`
- `calibration_method`
- `training_timestamp`
- `owner`

## Serving contract alignment

Serving metadata written by production must stay aligned with the serving versioning policy:
- `model_version`
- `benchmark_id`
- `feature_set_version`
- `dataset_version`
- `mlflow_run_id`

Important:
- `mlflow_run_id` is required for internal lineage and auditability
- `mlflow_run_id` is **not part of the public API contract**
- the public API must remain frontend-safe and stable

## Promotion policy

Promotion stays manual.

Minimum decision inputs before a promotion:
- benchmark comparison on the same frozen temporal split
- baseline comparison
- calibration review if probabilities are used directly
- feature availability check at inference time
- explicit review of serving metadata values

No automatic promotion from MLflow registry to production runtime is allowed in this phase.

## Infra recommendation

Recommended minimal shared setup:
- MLflow tracking server
- PostgreSQL backend store
- shared artifact storage
- authenticated internal access only

Keep the first version simple:
- no complex model registry workflow required on day 1
- no serving dependency on MLflow availability
- no auto-deploy integration

## Migration path

### Phase 1
Keep current local workflow working.

Add a shared MLflow target for team use and document how to point experiments to it.

### Phase 2
Standardize mandatory run tags and artifact naming.

### Phase 3
Update training / benchmark scripts to default to shared tracking.

### Phase 4
Optionally add registry discipline later, only if collaboration volume justifies it.

## Non-goals

This change does not aim to:
- redesign the serving API
- rewrite historical predictions
- expose MLflow internals in the public API
- add automated retraining
- add automated deployment from MLflow

## Backfill position

Historical metadata backfill must be treated separately from MLflow shared architecture.

Rule:
- only backfill what is provable
- do not invent historical `mlflow_run_id` values
- prefer leaving `NULL` over fake lineage
- if needed, use an analytics view for legacy rows instead of rewriting source truth

## Acceptance criteria

This architecture is considered accepted when:
- the repo documents the shared/offline-only MLflow role clearly
- production serving is explicitly documented as manual and pinned
- public API contract remains unchanged
- backfill is documented as a separate conservative decision