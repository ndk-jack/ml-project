# ml-project/src/api/serving_metadata.py
"""
serving_metadata.py — Serving metadata for the active production model.

Ordre de priorité :
1. Variables d'environnement Railway (SERVING_MODEL_VERSION, etc.)
2. Defaults statiques correspondant à mlp_v1

Ajouter dans Railway (Settings > Variables) :
    SERVING_MODEL_VERSION       = mlp_v1__full__keras
    SERVING_BENCHMARK_ID        = benchmark_v2
    SERVING_FEATURE_SET_VERSION = mlp_full_v1
    SERVING_DATASET_VERSION     = dataset_v5_dedup
    SERVING_MLFLOW_RUN_ID       = <ton_run_id>   # optionnel
"""

import os

_DEFAULTS = {
    "model_version":       "mlp_v1__full__keras",
    "benchmark_id":        "benchmark_v2",
    "feature_set_version": "mlp_full_v1",
    "dataset_version":     "dataset_v5_dedup",
    "mlflow_run_id":       "",
}


def get_serving_metadata() -> dict:
    return {
        "model_version":       os.getenv("SERVING_MODEL_VERSION",       _DEFAULTS["model_version"]),
        "benchmark_id":        os.getenv("SERVING_BENCHMARK_ID",        _DEFAULTS["benchmark_id"]),
        "feature_set_version": os.getenv("SERVING_FEATURE_SET_VERSION", _DEFAULTS["feature_set_version"]),
        "dataset_version":     os.getenv("SERVING_DATASET_VERSION",     _DEFAULTS["dataset_version"]),
        "mlflow_run_id":       os.getenv("SERVING_MLFLOW_RUN_ID",       _DEFAULTS["mlflow_run_id"]),
    }