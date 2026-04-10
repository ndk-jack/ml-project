from pathlib import Path
import json
import subprocess

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from config import load_benchmark_config


PROJECT_ROOT = Path("/Users/nazlidecker/ml-project")
FEATURE_SELECTION_REPORT = PROJECT_ROOT / "experiments" / "reports" / "feature_selection" / "feature_selection_report_v1.json"


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


def load_dataset(cfg: dict, target: str, selected_features: list[str]):
    dataset_path = Path(cfg["dataset"]["path"])
    time_col = cfg["dataset"]["time_column"]

    df = pd.read_csv(dataset_path, low_memory=False)
    df[time_col] = pd.to_datetime(df[time_col], format="mixed", errors="coerce", utc=True)
    df = df.dropna(subset=[time_col, target]).copy()

    train_end = pd.Timestamp(cfg["split"]["train_end"], tz="UTC")
    val_end = pd.Timestamp(cfg["split"]["validation_end"], tz="UTC")
    test_end = pd.Timestamp(cfg["split"]["test_end"], tz="UTC")

    train_df = df[df[time_col] <= train_end].copy()
    val_df = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
    test_df = df[(df[time_col] > val_end) & (df[time_col] <= test_end)].copy()

    feature_cols = [c for c in selected_features if c in df.columns]

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_train = train_df[target].astype(int)
    y_val = val_df[target].astype(int)
    y_test = test_df[target].astype(int)

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def main():
    cfg = load_benchmark_config()
    tracking_uri = cfg["tracking_uri"]
    experiment_name = cfg["experiment_name"]
    git_sha = get_git_sha()

    feature_report = json.loads(FEATURE_SELECTION_REPORT.read_text())
    selected_features = feature_report["candidate_feature_set_v1"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    params = {
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
        "n_estimators": 300,
    }

    for target in cfg["dataset"]["target_columns"]:
        horizon = target.replace("label_", "")
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_dataset(cfg, target, selected_features)

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        val_score = model.predict_proba(X_val)[:, 1]
        test_score = model.predict_proba(X_test)[:, 1]

        val_metrics = compute_metrics(y_val, val_score)
        test_metrics = compute_metrics(y_test, test_score)

        run_name = f"{cfg['benchmark_id']}_minimal_features_{horizon}_{git_sha[:7]}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "benchmark_id": cfg["benchmark_id"],
                    "run_kind": "feature_selection_train",
                    "candidate_status": "challenger",
                    "target": target,
                    "horizon": horizon,
                    "feature_set_version": "candidate_feature_set_v1",
                    "git_sha": git_sha,
                    "benchmark_path": cfg["_benchmark_path"],
                }
            )

            mlflow.log_params(
                {
                    "dataset_path": cfg["dataset"]["path"],
                    "dataset_version": cfg["dataset"]["version"],
                    "feature_count": len(feature_cols),
                    "feature_selection_report": str(FEATURE_SELECTION_REPORT),
                    **params,
                }
            )

            mlflow.log_metrics(
                {
                    **{f"val_{k}": v for k, v in val_metrics.items() if v is not None},
                    **{f"test_{k}": v for k, v in test_metrics.items() if v is not None},
                    "train_rows": float(len(y_train)),
                    "val_rows": float(len(y_val)),
                    "test_rows": float(len(y_test)),
                }
            )

            artifact_dir = PROJECT_ROOT / "experiments" / "reports" / "minimal_features"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            report_path = artifact_dir / f"{target}_minimal_features.json"
            report_path.write_text(
                json.dumps(
                    {
                        "target": target,
                        "selected_features": feature_cols,
                        "feature_count": len(feature_cols),
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                    },
                    indent=2,
                )
            )

            mlflow.log_artifact(str(report_path), artifact_path=f"minimal_features/{horizon}")

            print(f"Minimal-features run logged for {target}")
            print(f"feature_count={len(feature_cols)}")
            print(f"val_metrics={val_metrics}")
            print(f"test_metrics={test_metrics}")
            print()


if __name__ == "__main__":
    main()
