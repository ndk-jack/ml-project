# Model decision — benchmark_v2

## Decision
Keep the compact 22-feature candidate as the preferred working model for benchmark_v2.

## Scope
- dataset: dataset_v5_dedup
- targets: label_7d, label_30d
- feature set: candidate_feature_set_v1
- params: best_params_v2

## Rationale
- 7d compact+best_params is effectively equal to full+best_params.
- 30d compact+best_params is only slightly below full+best_params.
- The compact model is simpler, easier to maintain, and less redundant.

## Deferred work
- No further broad Optuna pass for now.
- No active 365d benchmark.
- Grafana is deferred to P2.
- Next ML priority: final candidate packaging + champion/challenger promotion rules.
