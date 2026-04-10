import mlflow
from mlflow.tracking import MlflowClient

from config import load_benchmark_config

from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:////{_PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "earthquake-forecasting"


def main():
    cfg = load_benchmark_config()
    benchmark_id = cfg["benchmark_id"]
    allowed_targets = set(cfg["dataset"]["target_columns"])

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=200,
    )

    latest_baseline = None
    latest_challengers = {}

    for run in runs:
        tags = run.data.tags
        metrics = run.data.metrics
        run_kind = tags.get("run_kind")
        target = tags.get("target")
        run_benchmark_id = tags.get("benchmark_id")

        if run_benchmark_id != benchmark_id:
            continue

        if run_kind == "baseline" and latest_baseline is None:
            latest_baseline = run

        elif (
            run_kind == "benchmark_train"
            and target in allowed_targets
            and target not in latest_challengers
            and "test_pr_auc" in metrics
            and "test_roc_auc" in metrics
            and "test_brier_score" in metrics
        ):
            latest_challengers[target] = run

    if latest_baseline is None:
        print(f"No baseline run found for benchmark_id={benchmark_id}")
        return

    print("benchmark_id:", benchmark_id)
    print("benchmark_path:", cfg["_benchmark_path"])
    print("BASELINE RUN:", latest_baseline.data.tags.get("mlflow.runName"))
    print()

    for target in sorted(allowed_targets):
        run = latest_challengers.get(target)
        print("=" * 80)
        print("TARGET:", target)

        if run is None:
            print("No valid challenger run found.")
            print()
            continue

        print("CHALLENGER:", run.data.tags.get("mlflow.runName"))

        baseline_metrics = latest_baseline.data.metrics
        challenger_metrics = run.data.metrics

        for metric in ["pr_auc", "roc_auc", "brier_score"]:
            b = baseline_metrics.get(f"{target}_test_{metric}")
            c = challenger_metrics.get(f"test_{metric}")
            print(f"{metric}: baseline={b} challenger={c}")
        print()

if __name__ == "__main__":
    main()
