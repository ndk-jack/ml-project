from datetime import datetime, timezone
from collections import defaultdict
import os

import requests
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from supabase import create_client


USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
MAX_RADIUS_KM = 200
MIN_MAG = 5.0

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:////Users/nazlidecker/ml-project/mlflow.db")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "earthquake-forecasting")


def get_client():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY missing")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def has_followup(event_dt: str, lat: float, lon: float, horizon_days: int) -> bool:
    start_dt = pd.Timestamp(event_dt, tz="UTC") + pd.Timedelta(seconds=1)
    end_dt = pd.Timestamp(event_dt, tz="UTC") + pd.Timedelta(days=horizon_days)

    params = {
        "format": "geojson",
        "starttime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": MAX_RADIUS_KM,
        "minmagnitude": MIN_MAG,
        "orderby": "time",
        "limit": 2000,
    }

    resp = requests.get(USGS_API, params=params, timeout=30)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    return len(features) > 0


def evaluate_pending_outcomes(client):
    now_iso = datetime.now(timezone.utc).isoformat()

    pending = (
        client.table("prediction_outcomes")
        .select("*")
        .or_(f"and(actual_label_7d.is.null,maturity_7d_at.lte.{now_iso}),and(actual_label_30d.is.null,maturity_30d_at.lte.{now_iso})")
        .limit(500)
        .execute()
    )

    rows = pending.data if hasattr(pending, "data") else []
    print(f"pending_rows={len(rows)}")

    updated = 0

    for row in rows:
        prediction_id = row["prediction_id"]
        event_dt = row["event_datetime"]
        lat = row["latitude"]
        lon = row["longitude"]

        updates = {
            "updated_at": now_iso,
        }

        if row.get("actual_label_7d") is None and pd.Timestamp(row["maturity_7d_at"]) <= pd.Timestamp(now_iso):
            updates["actual_label_7d"] = has_followup(event_dt, lat, lon, 7)
            updates["evaluated_7d_at"] = now_iso

        if row.get("actual_label_30d") is None and pd.Timestamp(row["maturity_30d_at"]) <= pd.Timestamp(now_iso):
            updates["actual_label_30d"] = has_followup(event_dt, lat, lon, 30)
            updates["evaluated_30d_at"] = now_iso

        if len(updates) > 1:
            client.table("prediction_outcomes").update(updates).eq("prediction_id", prediction_id).execute()
            updated += 1

    print(f"updated_rows={updated}")


def compute_metrics(y_true, y_score):
    y_true = pd.Series(y_true).astype(int)
    y_score = pd.Series(y_score).astype(float)

    metrics = {
        "pr_auc": None,
        "roc_auc": None,
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "positive_rate": float(y_true.mean()),
        "sample_size": int(len(y_true)),
    }

    if y_true.nunique() < 2:
        return metrics

    metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def build_eval_snapshots(client):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    prediction_rows = client.table("prediction_log").select("*").execute()
    outcome_rows = client.table("prediction_outcomes").select("*").execute()

    pred_df = pd.DataFrame(prediction_rows.data if hasattr(prediction_rows, "data") else [])
    out_df = pd.DataFrame(outcome_rows.data if hasattr(outcome_rows, "data") else [])

    if pred_df.empty or out_df.empty:
        print("No prediction/outcome data to evaluate.")
        return

    df = pred_df.merge(
        out_df,
        on=["prediction_id", "event_id"],
        how="inner",
        suffixes=("_pred", "_out"),
    )

    snapshots = []

    for horizon, pred_col, label_col in [
        ("7d", "prob_7d", "actual_label_7d"),
        ("30d", "prob_30d", "actual_label_30d"),
    ]:
        subset = df.dropna(subset=[pred_col, label_col]).copy()
        if subset.empty:
            continue

        for (model_version, benchmark_id, feature_set_version, dataset_version), part in subset.groupby(
            ["model_version", "benchmark_id", "feature_set_version", "dataset_version"]
        ):
            y_true = part[label_col].astype(int)
            y_score = part[pred_col].astype(float)

            if y_true.nunique() < 2:
                print(
                    f"Skipping snapshot horizon={horizon} model_version={model_version}: "
                    f"only one class present (sample_size={len(y_true)})"
                )
                continue

            metrics = compute_metrics(y_true, y_score)

            row = {
                "benchmark_id": benchmark_id,
                "model_version": model_version,
                "feature_set_version": feature_set_version,
                "dataset_version": dataset_version,
                "horizon": horizon,
                "sample_size": metrics["sample_size"],
                "positive_rate": metrics["positive_rate"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "window_start": part["event_datetime_pred"].min(),
                "window_end": part["event_datetime_pred"].max(),
                "notes": "generated by evaluate_prediction_outcomes.py",
            }

            snapshots.append(row)
            client.table("model_eval_snapshots").insert(row).execute()

    print(f"snapshot_rows_written={len(snapshots)}")


def main():
    client = get_client()
    evaluate_pending_outcomes(client)
    build_eval_snapshots(client)


if __name__ == "__main__":
    main()
