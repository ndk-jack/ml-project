from pathlib import Path
import json
import subprocess
import argparse

import lightgbm as lgb
import mlflow
from mlflow.tracking import MlflowClient
import optuna
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


def load_dataset(cfg: dict, target: str):
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

    excluded_cols = {time_col, "label_7d", "label_30d", "label_365d"}
    candidate_feature_cols = [c for c in df.columns if c not in excluded_cols]
    numeric_feature_cols = [c for c in candidate_feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    valid_cols = [c for c in numeric_feature_cols if train_df[c].notna().sum() > 0]

    X_train = train_df[valid_cols].copy()
    X_val = val_df[valid_cols].copy()
    X_test = test_df[valid_cols].copy()

    y_train = train_df[target].astype(int)
    y_val = val_df[target].astype(int)
    y_test = test_df[target].astype(int)

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    return X_train, y_train, X_val, y_val, X_test, y_test, valid_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["label_7d", "label_30d"])
    parser.add_argument("--trials", type=int, default=20)
    args, _ = parser.parse_known_args()

    cfg = load_benchmark_config()
    tracking_uri = cfg["tracking_uri"]
    experiment_name = cfg["experiment_name"]
    target = args.target
    horizon = target.replace("label_", "")
    git_sha = get_git_sha()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_dataset(cfg, target)

    study_name = f"{cfg['benchmark_id']}_optuna_{horizon}"

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "seed": int(cfg["seed"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        val_score = model.predict_proba(X_val)[:, 1]
        test_score = model.predict_proba(X_test)[:, 1]

        val_metrics = compute_metrics(y_val, val_score)
        test_metrics = compute_metrics(y_test, test_score)

        with mlflow.start_run(nested=True, run_name=f"{study_name}_trial_{trial.number}"):
            mlflow.set_tags(
                {
                    "benchmark_id": cfg["benchmark_id"],
                    "run_kind": "optuna_trial",
                    "candidate_status": "challenger",
                    "target": target,
                    "horizon": horizon,
                    "git_sha": git_sha,
                    "benchmark_path": cfg["_benchmark_path"],
                    "study_name": study_name,
                }
            )
            mlflow.log_params(params)
            mlflow.log_param("feature_count", len(feature_cols))
            mlflow.log_metric("val_pr_auc", val_metrics["pr_auc"])
            mlflow.log_metric("val_roc_auc", val_metrics["roc_auc"])
            mlflow.log_metric("val_brier_score", val_metrics["brier_score"])
            mlflow.log_metric("test_pr_auc", test_metrics["pr_auc"])
            mlflow.log_metric("test_roc_auc", test_metrics["roc_auc"])
            mlflow.log_metric("test_brier_score", test_metrics["brier_score"])

        return val_metrics["pr_auc"]

    with mlflow.start_run(run_name=f"{study_name}_{git_sha[:7]}"):
        mlflow.set_tags(
            {
                "benchmark_id": cfg["benchmark_id"],
                "run_kind": "optuna_study",
                "candidate_status": "challenger",
                "target": target,
                "horizon": horizon,
                "git_sha": git_sha,
                "benchmark_path": cfg["_benchmark_path"],
                "study_name": study_name,
            }
        )
        mlflow.log_params(
            {
                "target": target,
                "trials": args.trials,
                "dataset_path": cfg["dataset"]["path"],
                "dataset_version": cfg["dataset"]["version"],
                "split_version": cfg["champion"]["split_version"],
                "feature_count": len(feature_cols),
            }
        )

        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(objective, n_trials=args.trials)

        best_params = study.best_params
        best_value = study.best_value

        mlflow.log_metric("best_val_pr_auc", best_value)
        mlflow.log_dict(best_params, f"best_params_{target}.json")

        artifact_dir = PROJECT_ROOT / "experiments" / "artifacts" / study_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        summary_path = artifact_dir / f"optuna_summary_{target}.json"
        summary_path.write_text(json.dumps(
            {
                "target": target,
                "study_name": study_name,
                "best_value": best_value,
                "best_params": best_params,
                "trials": args.trials,
            },
            indent=2,
        ))
        mlflow.log_artifact(str(summary_path), artifact_path=f"optuna/{horizon}")

        print("Optuna study completed successfully.")
        print(f"target={target}")
        print(f"study_name={study_name}")
        print(f"best_val_pr_auc={best_value}")
        print(f"best_params={best_params}")


if __name__ == "__main__":
    main()
