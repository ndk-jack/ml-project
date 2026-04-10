from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:////Users/nazlidecker/ml-project/mlflow.db"
EXPERIMENT_NAME = "earthquake-forecasting"

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print("Experiment not found.")
        return

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )

    for run in runs:
        tags = run.data.tags
        metrics = run.data.metrics
        print("=" * 80)
        print("run_name:", tags.get("mlflow.runName"))
        print("run_kind:", tags.get("run_kind"))
        print("candidate_status:", tags.get("candidate_status"))
        print("target:", tags.get("target"))
        print("git_sha:", tags.get("git_sha"))
        print("metrics:")
        for k in sorted(metrics.keys()):
            print(f"  {k}: {metrics[k]}")

if __name__ == "__main__":
    main()
