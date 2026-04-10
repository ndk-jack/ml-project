import mlflow
from mlflow.tracking import MlflowClient

from config import load_benchmark_config

TRACKING_URI = "sqlite:////Users/nazlidecker/ml-project/mlflow.db"
EXPERIMENT_NAME = "earthquake-forecasting"


def main():
    cfg = load_benchmark_config()
    benchmark_id = cfg["benchmark_id"]
    targets = cfg["dataset"]["target_columns"]

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=200,
    )

    wanted = {
        "best_params_retrain",
        "feature_selection_train",
        "compact_best_params_retrain",
    }

    latest = {}

    for run in runs:
        tags = run.data.tags
        target = tags.get("target")
        run_kind = tags.get("run_kind")
        run_benchmark_id = tags.get("benchmark_id")

        if run_benchmark_id != benchmark_id:
            continue
        if run_kind not in wanted:
            continue
        if target not in targets:
            continue

        key = (target, run_kind)
        if key not in latest:
            latest[key] = run

    for target in targets:
        print("=" * 100)
        print("TARGET:", target)

        for run_kind in ["best_params_retrain", "feature_selection_train", "compact_best_params_retrain"]:
            run = latest.get((target, run_kind))
            if run is None:
                print(f"{run_kind}: missing")
                continue

            metrics = run.data.metrics
            run_name = run.data.tags.get("mlflow.runName")
            print(
                f"{run_kind} | run={run_name} | "
                f"test_pr_auc={metrics.get('test_pr_auc')} | "
                f"test_roc_auc={metrics.get('test_roc_auc')} | "
                f"test_brier_score={metrics.get('test_brier_score')}"
            )
        print()

if __name__ == "__main__":
    main()
