"""
add_ratio_features.py — Add acceleration/ratio features to dataset.csv
and retrain LightGBM to attempt to beat ROC-AUC 0.856.

New features (4):
  accel_count    = count_7d  / (count_90d / 90 * 7)   seismic acceleration
  accel_energy   = energy_7d / (energy_90d / 90 * 7)  energy acceleration
  b_value_trend  = b_value_7d - b_value_90d            stress regime change
  mag_excess     = magnitude  - mag_mean_90d           anomaly vs background

Usage : python3 src/add_ratio_features.py
Output:
  data/features/dataset_v2.csv
  models/lgbm_v2_label7d.txt
  reports/lgbm_v2_classification_report.txt
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score,
)
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_V1    = Path("data/features/dataset.csv")
DATASET_V2    = Path("data/features/dataset_v2.csv")
MODEL_OUT     = Path("models/lgbm_v2_label7d.txt")
REPORT_OUT    = Path("reports/lgbm_v2_classification_report.txt")

TARGET        = "label_7d"
TEST_YEAR     = 2010
RANDOM_STATE  = 42
EPS           = 1e-9      # avoid division by zero


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 4 acceleration / anomaly features."""

    # Expected background count over 7 days given 90-day window
    background_count_7d  = (df["count_90d"] / 90.0) * 7.0
    background_energy_7d = (df["energy_90d"] / 90.0) * 7.0

    # 1. Seismic acceleration (count)
    df["accel_count"]   = df["count_7d"]  / (background_count_7d  + EPS)

    # 2. Energy acceleration
    df["accel_energy"]  = df["energy_7d"] / (background_energy_7d + EPS)

    # 3. b-value trend (short term minus long term)
    #    Negative = b-value dropping = stress increasing
    df["b_value_trend"] = df["b_value_7d"].fillna(1.0) - df["b_value_90d"].fillna(1.0)

    # 4. Magnitude excess vs 90-day local background
    #    Positive = this event is larger than usual for this zone
    df["mag_excess"]    = df["magnitude"] - df["mag_mean_90d"].fillna(df["magnitude"])

    # Clip extremes (e.g. divide-by-near-zero gives huge ratios)
    df["accel_count"]  = df["accel_count"].clip(0, 500)
    df["accel_energy"] = df["accel_energy"].clip(0, 500)

    return df


def temporal_split(df, feature_cols):
    df["_year"]  = pd.to_datetime(df["datetime"]).dt.year
    train_mask   = df["_year"] < TEST_YEAR
    test_mask    = df["_year"] >= TEST_YEAR

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, TARGET]
    X_test  = df.loc[test_mask,  feature_cols]
    y_test  = df.loc[test_mask,  TARGET]
    return X_train, X_test, y_train, y_test


def train_and_eval(X_train, y_train, X_test, y_test, label=""):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos

    model = lgb.LGBMClassifier(
        objective         = "binary",
        metric            = "auc",
        n_estimators      = 1000,
        learning_rate     = 0.05,
        num_leaves        = 63,
        max_depth         = -1,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_samples = 50,
        scale_pos_weight  = spw,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbosity         = -1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    y_proba  = model.predict_proba(X_test)[:, 1]
    y_pred   = model.predict(X_test)
    roc_auc  = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    report   = classification_report(y_test, y_pred,
                                     target_names=["No event", "Event ≥M5"])
    cm       = confusion_matrix(y_test, y_pred)

    print(f"\n   [{label}]")
    print(f"   ROC-AUC       : {roc_auc:.4f}")
    print(f"   Avg Precision : {avg_prec:.4f}")
    print(f"   Best iter     : {model.best_iteration_}")
    print(f"\n   Confusion matrix:")
    print(f"             Predicted 0   Predicted 1")
    print(f"   Actual 0  {cm[0,0]:>10,}   {cm[0,1]:>10,}")
    print(f"   Actual 1  {cm[1,0]:>10,}   {cm[1,1]:>10,}")
    print(f"\n{report}")

    return model, roc_auc, avg_prec, report


def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # ── Load v1 dataset ───────────────────────────────────────────────────────
    print("── Loading dataset_v1 ──────────────────────────────────────────")
    df = pd.read_csv(DATASET_V1, low_memory=False)
    print(f"   {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── Add ratio features ────────────────────────────────────────────────────
    print("\n── Adding ratio / acceleration features ────────────────────────")
    df = add_ratio_features(df)
    new_feats = ["accel_count", "accel_energy", "b_value_trend", "mag_excess"]
    print(f"   Added: {new_feats}")
    for f in new_feats:
        print(f"   {f:<20}  mean={df[f].mean():.3f}  std={df[f].std():.3f}  "
              f"NaN={df[f].isna().mean():.1%}")

    # Save v2
    df.to_csv(DATASET_V2, index=False)
    print(f"\n   Saved → {DATASET_V2}")

    # ── Feature columns ───────────────────────────────────────────────────────
    drop_cols    = ["datetime", "ref_lat", "ref_lon", "latitude", "longitude",
                    "label_7d", "label_30d", "label_365d", "_year"]
    feature_cols_v1 = [c for c in df.columns
                       if c not in drop_cols + new_feats]
    feature_cols_v2 = feature_cols_v1 + new_feats

    # ── Baseline (v1 features, same model for fair comparison) ───────────────
    print("\n── Training comparison ─────────────────────────────────────────")
    print("── 4. Evaluation ───────────────────────────────────────────────")

    X_train_v1, X_test_v1, y_train, y_test = temporal_split(df, feature_cols_v1)
    X_train_v2, X_test_v2, _,       _      = temporal_split(df, feature_cols_v2)

    print("\n   Training v1 (51 features, baseline) …")
    model_v1, roc_v1, prec_v1, report_v1 = train_and_eval(
        X_train_v1, y_train, X_test_v1, y_test,
        label="v1 — 51 features (baseline)"
    )

    print("\n   Training v2 (55 features, +ratio) …")
    model_v2, roc_v2, prec_v2, report_v2 = train_and_eval(
        X_train_v2, y_train, X_test_v2, y_test,
        label="v2 — 55 features (+accel_count, accel_energy, b_value_trend, mag_excess)"
    )

    # ── Delta ─────────────────────────────────────────────────────────────────
    print("\n── Results summary ─────────────────────────────────────────────")
    delta_roc  = roc_v2 - roc_v1
    delta_prec = prec_v2 - prec_v1
    arrow_roc  = "▲" if delta_roc > 0 else "▼"
    arrow_prec = "▲" if delta_prec > 0 else "▼"

    print(f"   ROC-AUC       v1={roc_v1:.4f}  v2={roc_v2:.4f}  "
          f"{arrow_roc} {abs(delta_roc):.4f}")
    print(f"   Avg Precision v1={prec_v1:.4f}  v2={prec_v2:.4f}  "
          f"{arrow_prec} {abs(delta_prec):.4f}")

    # ── Feature importances of new features ───────────────────────────────────
    importances = pd.Series(
        model_v2.feature_importances_, index=feature_cols_v2
    ).sort_values(ascending=False)

    print("\n   New features rank in v2:")
    for f in new_feats:
        rank = list(importances.index).index(f) + 1
        val  = importances[f]
        bar  = "█" * int(val / importances.max() * 30)
        print(f"   #{rank:<3} {f:<22} {val:>8.0f}  {bar}")

    print("\n   Top 10 overall:")
    for feat, imp in importances.head(10).items():
        mark = " ← NEW" if feat in new_feats else ""
        bar  = "█" * int(imp / importances.max() * 30)
        print(f"   {feat:<30} {imp:>8.0f}  {bar}{mark}")

    # ── Save ──────────────────────────────────────────────────────────────────
    model_v2.booster_.save_model(str(MODEL_OUT))
    report_text = (
        f"Model         : LightGBM v2 (+ratio features)\n"
        f"Target        : {TARGET}\n"
        f"Test from year: {TEST_YEAR}\n\n"
        f"ROC-AUC       v1={roc_v1:.4f}  v2={roc_v2:.4f}  {arrow_roc}{abs(delta_roc):.4f}\n"
        f"Avg Precision v1={prec_v1:.4f}  v2={prec_v2:.4f}  {arrow_prec}{abs(delta_prec):.4f}\n\n"
        f"{report_v2}\n\n"
        f"Top 20 features:\n{importances.head(20).to_string()}\n"
    )
    REPORT_OUT.write_text(report_text)
    print(f"\n   Model  → {MODEL_OUT}")
    print(f"   Report → {REPORT_OUT}")


if __name__ == "__main__":
    main()
