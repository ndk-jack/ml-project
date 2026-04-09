from pathlib import Path
import json
import subprocess

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from config import load_benchmark_config


PROJECT_ROOT = Path("/Users/nazlidecker/ml-project")


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
    tracking_uri = cfg["tracking_uri"]
    experiment_name = cfg["experiment_name"]
    dataset_path = Path(cfg["dataset"]["path"])
    time_col = cfg["dataset"]["time_column"]
    target_columns = cfg["dataset"]["target_columns"]

    df = pd.read_csv(dataset_path, low_memory=False)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[time_col]).copy()

    train_end = pd.Timestamp(cfg["split"]["train_end"], tz="UTC")
    val_end = pd.Timestamp(cfg["split"]["validation_end"], tz="UTC")
    test_end = pd.Timestamp(cfg["split"]["test_end"], tz="UTC")

    train_df = df[df[time_col] <= train_end].copy()
    val_df = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
    test_df = df[(df[time_col] > val_end) & (df[time_col] <= test_end)].copy()

    excluded_cols = {time_col, "label_7d", "label_30d", "label_365d"}
    feature_cols = [c for c in df.columns if c not in excluded_cols]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    git_sha = get_git_sha()

    common_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "verbosity": -1,
        "seed": int(cfg["seed"]),
    }

    for target in target_columns:
        horizon = target.replace("label_", "")
        run_name = f"{cfg['benchmark_id']}_lightgbm_{horizon}_{git_sha[:7]}"

        train = train_df.dropna(subset=[target]).copy()
        val = val_df.dropna(subset=[target]).copy()
        test = test_df.dropna(subset=[target]).copy()

        X_train = train[feature_cols]
        y_train = train[target].astype(int)
        X_val = val[feature_cols]
        y_val = val[target].astype(int)
        X_test = test[feature_cols]
        y_test = test[target].astype(int)

        model = lgb.LGBMClassifier(**common_params, n_estimators=300)

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "benchmark_id": cfg["benchmark_id"],
                    "dataset_version": cfg["dataset"]["version"],
                    "feature_set_version": cfg["champion"]["feature_set_version"],
                    "split_version": cfg["champion"]["split_version"],
                    "git_sha": git_sha,
                    "run_kind": "benchmark_train",
                    "candidate_status": "challenger",
                    "target": target,
                    "horizon": horizon,
                }
            )

            mlflow.log_params(
                {
                    "dataset_path": str(dataset_path),
                    "time_column": time_col,
                    "target": target,
                    "feature_count": len(feature_cols),
                    **common_params,
                    "n_estimators": 300,
                }
            )

            model.fit(X_train, y_train)

            val_score = model.predict_proba(X_val)[:, 1]
            test_score = model.predict_proba(X_test)[:, 1]

            val_metrics = compute_metrics(y_val, val_score)
            test_metrics = compute_metrics(y_test, test_score)

            mlflow.log_metrics(
                {
                    **{f"val_{k}": v for k, v in val_metrics.items() if v is not None},
                    **{f"test_{k}": v for k, v in test_metrics.items() if v is not None},
                    "train_rows": float(len(train)),
                    "val_rows": float(len(val)),
                    "test_rows": float(len(test)),
                }
            )

            artifact_dir = PROJECT_ROOT / "experiments" / "artifacts" / run_name
            artifact_dir.mkdir(parents=True, exist_ok=True)

            model_path = artifact_dir / f"model_{target}.txt"
            features_path = artifact_dir / f"features_{target}.json"
            report_path = artifact_dir / f"report_{target}.json"

            model.booster_.save_model(str(model_path))
            features_path.write_text(json.dumps(feature_cols, indent=2))
            report_path.write_text(
                json.dumps(
                    {
                        "target": target,
                        "horizon": horizon,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "feature_count": len(feature_cols),
                    },
                    indent=2,
                )
            )

            mlflow.log_artifact(str(model_path), artifact_path=f"models/{horizon}")
            mlflow.log_artifact(str(features_path), artifact_path=f"features/{horizon}")
            mlflow.log_artifact(str(report_path), artifact_path=f"reports/{horizon}")

            print(f"Logged benchmark run for {target}: {run_name}")


if __name__ == "__main__":
    main()
