from pathlib import Path
import subprocess

import mlflow
import pandas as pd

from config import load_benchmark_config


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


def compute_rates(df: pd.DataFrame, targets: list[str]) -> dict:
    out = {}
    for col in targets:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        out[col] = float(s.mean()) if len(s) else None
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
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[time_col]).copy()

    train = df[df[time_col] <= train_end].copy()
    val = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
    test = df[(df[time_col] > val_end) & (df[time_col] <= test_end)].copy()

    train_rates = compute_rates(train, targets)
    val_rates = compute_rates(val, targets)
    test_rates = compute_rates(test, targets)

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
            }
        )

        metrics = {}
        for target in targets:
            metrics[f"{target}_train_positive_rate"] = train_rates[target]
            metrics[f"{target}_validation_positive_rate"] = val_rates[target]
            metrics[f"{target}_test_positive_rate"] = test_rates[target]

        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            {
                "train_positive_rates": train_rates,
                "validation_positive_rates": val_rates,
                "test_positive_rates": test_rates,
                "split_sizes": {
                    "train": int(len(train)),
                    "validation": int(len(val)),
                    "test": int(len(test)),
                },
            },
            "naive_baseline_report.json",
        )

        print("Naive baseline logged successfully.")
        print(f"tracking_uri={tracking_uri}")
        print(f"experiment_name={experiment_name}")


if __name__ == "__main__":
    main()
