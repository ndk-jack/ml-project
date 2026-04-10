import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from config import PROJECT_ROOT, BENCHMARK_PATH, load_benchmark_config


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


def build_dataset_manifest(dataset_path: Path) -> dict:
    sample = pd.read_csv(dataset_path, nrows=1000, low_memory=False)
    with dataset_path.open("r", encoding="utf-8", errors="ignore") as f:
        row_count = max(sum(1 for _ in f) - 1, 0)

    return {
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_path.exists(),
        "dataset_version": "dataset_v3",
        "row_count": row_count,
        "column_count": len(sample.columns),
        "columns": list(sample.columns),
        "sample_null_rate_top20": {
            k: float(v)
            for k, v in sample.isna().mean().sort_values(ascending=False).head(20).to_dict().items()
        },
    }


def main():
    cfg = load_benchmark_config()
    tracking_uri = cfg["tracking_uri"]
    experiment_name = cfg["experiment_name"]
    dataset_path = Path(cfg["dataset"]["path"])

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    git_sha = get_git_sha()
    run_name = f"{cfg['benchmark_id']}_bootstrap_{git_sha[:7]}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "benchmark_id": cfg["benchmark_id"],
                "dataset_version": cfg["dataset"]["version"],
                "feature_set_version": cfg["champion"]["feature_set_version"],
                "split_version": cfg["champion"]["split_version"],
                "git_sha": git_sha,
                "run_kind": "bootstrap",
                "candidate_status": "champion_reference",
            }
        )

        mlflow.log_params(
            {
                "tracking_uri": tracking_uri,
                "dataset_path": str(dataset_path),
                "primary_metric": cfg["metrics"]["primary"],
                "baseline_type": cfg["baseline"]["type"],
                "seed": cfg["seed"],
                "champion_tag": cfg["champion"]["tag"],
            }
        )

        manifest = build_dataset_manifest(dataset_path)
        mlflow.log_metrics(
            {
                "dataset_row_count": float(manifest["row_count"]),
                "dataset_column_count": float(manifest["column_count"]),
            }
        )

        mlflow.log_dict(cfg, "benchmark_config.json")
        mlflow.log_dict(manifest, "dataset_manifest.json")
        mlflow.log_artifact(str(BENCHMARK_PATH))

        print("MLflow bootstrap run logged successfully.")
        print(f"tracking_uri={tracking_uri}")
        print(f"experiment_name={experiment_name}")
        print(f"run_name={run_name}")


if __name__ == "__main__":
    main()
