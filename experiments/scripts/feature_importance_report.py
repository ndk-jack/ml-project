from pathlib import Path
import json
import subprocess

import lightgbm as lgb
import mlflow
import pandas as pd

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
    y_train = train_df[target].astype(int)

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)

    return X_train, y_train, valid_cols


def main():
    cfg = load_benchmark_config()
    tracking_uri = cfg["tracking_uri"]
    experiment_name = cfg["experiment_name"]
    git_sha = get_git_sha()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

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
        "n_estimators": 300,
    }

    for target in cfg["dataset"]["target_columns"]:
        horizon = target.replace("label_", "")
        X_train, y_train, feature_cols = load_dataset(cfg, target)

        model = lgb.LGBMClassifier(**common_params)
        model.fit(X_train, y_train)

        importance_gain = model.booster_.feature_importance(importance_type="gain")
        importance_split = model.booster_.feature_importance(importance_type="split")

        report = []
        for feat, gain, split in zip(feature_cols, importance_gain, importance_split):
            report.append(
                {
                    "feature": feat,
                    "importance_gain": float(gain),
                    "importance_split": float(split),
                }
            )

        report = sorted(report, key=lambda x: x["importance_gain"], reverse=True)

        artifact_dir = PROJECT_ROOT / "experiments" / "reports"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        out_path = artifact_dir / f"feature_importance_{target}.json"
        out_path.write_text(json.dumps(report, indent=2))

        with mlflow.start_run(run_name=f"{cfg['benchmark_id']}_feature_importance_{horizon}_{git_sha[:7]}"):
            mlflow.set_tags(
                {
                    "benchmark_id": cfg["benchmark_id"],
                    "run_kind": "feature_importance",
                    "target": target,
                    "horizon": horizon,
                    "git_sha": git_sha,
                    "benchmark_path": cfg["_benchmark_path"],
                }
            )
            mlflow.log_param("feature_count", len(feature_cols))
            mlflow.log_artifact(str(out_path), artifact_path=f"feature_importance/{horizon}")

        print(f"Feature importance report generated for {target}: {out_path}")
        print("Top 15 features by gain:")
        for row in report[:15]:
            print(f"{row['feature']} | gain={row['importance_gain']:.4f} | split={row['importance_split']:.0f}")
        print()

if __name__ == "__main__":
    main()
