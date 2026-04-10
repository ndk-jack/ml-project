import os

SERVING_MODEL_VERSION = os.getenv(
    "SERVING_MODEL_VERSION",
    "benchmark_v2__compact22__bestparams_v2",
)

SERVING_BENCHMARK_ID = os.getenv(
    "SERVING_BENCHMARK_ID",
    "benchmark_v2",
)

SERVING_FEATURE_SET_VERSION = os.getenv(
    "SERVING_FEATURE_SET_VERSION",
    "candidate_feature_set_v1",
)

SERVING_DATASET_VERSION = os.getenv(
    "SERVING_DATASET_VERSION",
    "dataset_v5_dedup",
)

SERVING_MLFLOW_RUN_ID = os.getenv(
    "SERVING_MLFLOW_RUN_ID",
    "",
)


def get_serving_metadata() -> dict:
    return {
        "model_version": SERVING_MODEL_VERSION,
        "benchmark_id": SERVING_BENCHMARK_ID,
        "feature_set_version": SERVING_FEATURE_SET_VERSION,
        "dataset_version": SERVING_DATASET_VERSION,
        "mlflow_run_id": SERVING_MLFLOW_RUN_ID or None,
    }
