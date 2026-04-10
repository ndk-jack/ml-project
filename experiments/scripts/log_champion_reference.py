import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from config import PROJECT_ROOT, BENCHMARK_PATH, load_benchmark_config


MODEL_PATHS = {
    "7d": PROJECT_ROOT / "models" / "lgbm_7d.txt",
    "30d": PROJECT_ROOT / "models" / "lgbm_30d.txt",
    "365d": PROJECT_ROOT / "models" / "lgbm_365d.txt",
}


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def build_dataset_manifest(dataset_path: Path, time_column: str, target_columns: list[str]) -> dict:
    usecols = [time_column] + target_columns
    df = pd.read_csv(dataset_path, usecols=usecols, low_memory=False)
    df[time_column] = pd.to_datetime(df[time_column], format="mixed", errors="coerce", utc=True)
    df = df.dropna(subset=[time_column]).copy()

    manifest = {
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_path.exists(),
        "row_count": int(len(df)),
        "datetime_min": str(df[time_column].min()),
        "datetime_max": str(df[time_column].max()),
        "target_columns": target_columns,
        "positive_rates": {},
    }

    for col in target_columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        manifest["positive_rates"][col] = float(s.mean()) if len(s) else None

    return manifest


def build_model_manifest() -> dict:
    out = {}
    for horizon, path in MODEL_PATHS.items():
        out[horizon] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return out


def main():
    cfg = load_benchmark_config()
    tracking_uri = cfg["tracking_uri"]
    experiment_name = cfg["experiment_name"]

    dataset_path = Path(cfg["dataset"]["path"])
    time_column = cfg["dataset"]["time_column"]
    target_columns = cfg["dataset"]["target_columns"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    git_sha = get_git_sha()
    run_name = f"{cfg['benchmark_id']}_champion_reference_{git_sha[:7]}"

    dataset_manifest = build_dataset_manifest(dataset_path, time_column, target_columns)
    model_manifest = build_model_manifest()

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "benchmark_id": cfg["benchmark_id"],
                "dataset_version": cfg["dataset"]["version"],
                "feature_set_version": cfg["champion"]["feature_set_version"],
                "split_version": cfg["champion"]["split_version"],
                "git_sha": git_sha,
                "run_kind": "champion_reference",
                "candidate_status": "champion",
                "champion_tag": cfg["champion"]["tag"],
            }
        )

        mlflow.log_params(
            {
                "dataset_path": str(dataset_path),
                "time_column": time_column,
                "target_columns": ",".join(target_columns),
                "model_family": cfg["champion"]["model_family"],
                "champion_tag": cfg["champion"]["tag"],
                "benchmark_path": cfg["_benchmark_path"],
            }
        )

        mlflow.log_metrics(
            {
                "dataset_row_count": float(dataset_manifest["row_count"]),
                "model_count": 3.0,
                "label_7d_positive_rate": float(dataset_manifest["positive_rates"]["label_7d"]),
                "label_30d_positive_rate": float(dataset_manifest["positive_rates"]["label_30d"]),
                "label_365d_positive_rate": float(dataset_manifest["positive_rates"]["label_365d"]),
            }
        )

        mlflow.log_dict(cfg, "benchmark_config.json")
        mlflow.log_dict(dataset_manifest, "dataset_manifest.json")
        mlflow.log_dict(model_manifest, "model_manifest.json")
        mlflow.log_artifact(str(BENCHMARK_PATH))

        for horizon, path in MODEL_PATHS.items():
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=f"models/{horizon}")

        print("Champion reference logged successfully.")
        print(f"tracking_uri={tracking_uri}")
        print(f"experiment_name={experiment_name}")
        print(f"run_name={run_name}")


if __name__ == "__main__":
    main()
