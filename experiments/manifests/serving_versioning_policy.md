# Serving versioning policy

Each served prediction must record:
- model_version
- benchmark_id
- feature_set_version
- dataset_version
- mlflow_run_id

Recommended current values for the active challenger:
- model_version: benchmark_v2__compact22__bestparams_v2
- benchmark_id: benchmark_v2
- feature_set_version: candidate_feature_set_v1
- dataset_version: dataset_v5_dedup
