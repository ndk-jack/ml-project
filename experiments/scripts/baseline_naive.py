from pathlib import Path
import subprocess

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from config import load_benchmark_config


PROJECT_ROOT = Path("/Users/nazlidecker/ml-project")


def get_git_sha(project_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def compute_metrics(y_true, y_score) -> dict:
    out = {}
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = None

    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = None

    try:
        out["brier_score"] = float(brier_score_loss(y_true, y_score))
    except Exception:
        out["brier_score"] = None

    return out


def main():
    cfg = load_benchmark_config()
    project_root = Path("/Users/nazlidecker/ml-project")
    dataset_path = Path(cfg["dataset"]["path"])
    time_col = cfg["dataset"]["time_column"]
    targets = cfg["dataset"]["target_columns"]

    train_end = pd.Timestamp(cfg["split"]["train_end"], tz="UTC")
    val_end = pd.Timestamp(cfg["split"]["validation_end"], tz="UTC")
    test_end = pd.Timestamp(cfg["split"]["test_end"], tz="UTC")

    df = pd.read_csv(dataset_path, usecols=[time_col] + targets, low_memory=False)
    df[time_col] = pd.to_datetime(
        df[time_col],
        format="mixed",
        errors="coerce",
        utc=True,
    )
    df = df.dropna(subset=[time_col]).copy()

    train = df[df[time_col] <= train_end].copy()
    val = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
    test = df[(df[time_col] > val_end) & (df[time_col] <= test_end)].copy()

    tracking_uri = cfg["tracking_uri"]
    experiment_name = cfg["experiment_name"]
    git_sha = get_git_sha(project_root)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{cfg['benchmark_id']}_naive_baseline_{git_sha[:7]}"):
        mlflow.set_tags(
            {
                "benchmark_id": cfg["benchmark_id"],
                "dataset_version": cfg["dataset"]["version"],
                "split_version": cfg["champion"]["split_version"],
                "git_sha": git_sha,
                "run_kind": "baseline",
                "baseline_type": "class_frequency",
            }
        )

        mlflow.log_params(
            {
                "dataset_path": str(dataset_path),
                "time_column": time_col,
                "target_columns": ",".join(targets),
                "benchmark_path": cfg["_benchmark_path"],
            }
        )

        metrics = {}
        report = {
            "split_sizes": {
                "train": int(len(train)),
                "validation": int(len(val)),
                "test": int(len(test)),
            },
            "targets": {},
        }

        for target in targets:
            y_train = pd.to_numeric(train[target], errors="coerce").dropna().astype(int)
            y_val = pd.to_numeric(val[target], errors="coerce").dropna().astype(int)
            y_test = pd.to_numeric(test[target], errors="coerce").dropna().astype(int)

            train_positive_rate = float(y_train.mean())
            val_score = np.full(shape=len(y_val), fill_value=train_positive_rate, dtype=float)
            test_score = np.full(shape=len(y_test), fill_value=train_positive_rate, dtype=float)

            val_metrics = compute_metrics(y_val, val_score)
            test_metrics = compute_metrics(y_test, test_score)

            metrics[f"{target}_train_positive_rate"] = train_positive_rate
            for k, v in val_metrics.items():
                if v is not None:
                    metrics[f"{target}_val_{k}"] = v
            for k, v in test_metrics.items():
                if v is not None:
                    metrics[f"{target}_test_{k}"] = v

            report["targets"][target] = {
                "train_positive_rate": train_positive_rate,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }

        mlflow.log_metrics(metrics)
        mlflow.log_dict(report, "naive_baseline_report.json")

        print("Naive baseline logged successfully.")
        print(f"tracking_uri={tracking_uri}")
        print(f"experiment_name={experiment_name}")


if __name__ == "__main__":
    main()
