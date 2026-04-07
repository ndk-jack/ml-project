"""
prepare_and_train.py — Merge features + labels, impute NaN, temporal split,
train XGBoost classifier on label_7d, evaluate.

Usage : python3 src/prepare_and_train.py
Output:
  data/features/dataset.csv      — merged & imputed dataset
  models/xgb_label7d.json        — trained XGBoost model
  reports/classification_report.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import json

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH = Path("data/features/features.csv")
LABELS_PATH   = Path("data/features/labels.csv")
DATASET_OUT   = Path("data/features/dataset.csv")
MODEL_OUT     = Path("models/xgb_label7d.json")
REPORT_OUT    = Path("reports/classification_report.txt")

TARGET        = "label_7d"          # most interesting horizon
TEST_YEAR     = 2010                # train < TEST_YEAR, test >= TEST_YEAR
RANDOM_STATE  = 42


# ── 1. Load & merge ───────────────────────────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    print("── 1. Loading features & labels ────────────────────────────────")
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    labels   = pd.read_csv(LABELS_PATH,   low_memory=False)
    print(f"   features : {features.shape[0]:,} rows × {features.shape[1]} cols")
    print(f"   labels   : {labels.shape[0]:,} rows × {labels.shape[1]} cols")

    # Build a common join key: ref_lat + ref_lon + date string
    features["_date"] = pd.to_datetime(features["datetime"]).dt.date.astype(str)
    labels["_date"]   = pd.to_datetime(labels["ref_date"]).dt.date.astype(str)

    features["_key"] = (features["ref_lat"].round(4).astype(str) + "|" +
                        features["ref_lon"].round(4).astype(str) + "|" +
                        features["_date"])
    labels["_key"]   = (labels["ref_lat"].round(4).astype(str) + "|" +
                        labels["ref_lon"].round(4).astype(str) + "|" +
                        labels["_date"])

    # Merge (inner join to keep only matched rows)
    label_cols = labels[["_key", "label_7d", "label_30d", "label_365d"]]

    # Handle potential duplicates on key (keep first)
    label_cols = label_cols.drop_duplicates(subset="_key", keep="first")
    features   = features.drop_duplicates(subset="_key", keep="first")

    df = features.merge(label_cols, on="_key", how="inner")
    df = df.drop(columns=["_key", "_date"])

    print(f"   merged   : {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ── 2. NaN imputation ─────────────────────────────────────────────────────────

def impute(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── 2. NaN imputation ───────────────────────────────────────────")

    strategies = {}

    for col in df.columns:
        if col in ["datetime", "ref_lat", "ref_lon",
                   "label_7d", "label_30d", "label_365d"]:
            continue

        nan_pct = df[col].isna().mean()
        if nan_pct == 0:
            continue

        # count / rate / energy / moment  → 0 (no activity = 0)
        if any(col.startswith(p) for p in
               ["count_", "rate_", "energy_", "moment_",
                "mag_std_", "depth_std_"]):
            df[col] = df[col].fillna(0.0)
            strategies[col] = "0"

        # elapsed_since_last → very large number (no prior activity)
        elif col == "elapsed_since_last_s":
            fill_val = df[col].quantile(0.99) * 10
            df[col] = df[col].fillna(fill_val)
            strategies[col] = f"99pct×10 = {fill_val:.0f}s"

        # b_value → 1.0 (global average)
        elif col.startswith("b_value_"):
            df[col] = df[col].fillna(1.0)
            strategies[col] = "1.0 (global avg)"

        # everything else → median
        else:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            strategies[col] = f"median={median:.3f}"

    print(f"   Imputed {len(strategies)} columns.")
    remaining_nan = df.isna().sum().sum()
    print(f"   Remaining NaN after imputation: {remaining_nan}")
    return df


# ── 3. Temporal split ─────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, target: str):
    print(f"\n── 3. Temporal split (train < {TEST_YEAR}, test >= {TEST_YEAR}) ─")

    df["_year"] = pd.to_datetime(df["datetime"]).dt.year

    drop_cols = ["datetime", "ref_lat", "ref_lon",
                 "latitude", "longitude",          # duplicated in ref features
                 "_year",
                 "label_7d", "label_30d", "label_365d"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    train_mask = df["_year"] <  TEST_YEAR
    test_mask  = df["_year"] >= TEST_YEAR

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target]
    X_test  = df.loc[test_mask,  feature_cols]
    y_test  = df.loc[test_mask,  target]

    print(f"   Train : {len(X_train):>7,} rows  "
          f"({train_mask.sum()/len(df):.0%})  "
          f"pos_rate={y_train.mean():.2%}")
    print(f"   Test  : {len(X_test):>7,} rows  "
          f"({test_mask.sum()/len(df):.0%})  "
          f"pos_rate={y_test.mean():.2%}")
    print(f"   Features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


# ── 4. Train XGBoost ──────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n── 4. Training XGBoost ─────────────────────────────────────────")

    try:
        import xgboost as xgb
    except ImportError:
        print("   XGBoost not found. Installing...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "xgboost", "-q"])
        import xgboost as xgb

    # Class imbalance: scale_pos_weight = neg/pos
    neg  = (y_train == 0).sum()
    pos  = (y_train == 1).sum()
    spw  = neg / pos
    print(f"   scale_pos_weight = {spw:.2f}  (neg={neg:,} / pos={pos:,})")

    model = xgb.XGBClassifier(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = spw,
        eval_metric       = "auc",
        early_stopping_rounds = 30,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbosity         = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    print(f"\n   Best iteration : {model.best_iteration}")
    return model


# ── 5. Evaluate ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, feature_cols):
    print("\n── 5. Evaluation ───────────────────────────────────────────────")

    from sklearn.metrics import (
        roc_auc_score, classification_report,
        confusion_matrix, average_precision_score,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=["No event", "Event ≥M5"])

    print(f"\n   ROC-AUC          : {roc_auc:.4f}")
    print(f"   Avg Precision    : {avg_prec:.4f}")
    print(f"\n   Confusion matrix:")
    print(f"             Predicted 0   Predicted 1")
    print(f"   Actual 0  {cm[0,0]:>10,}   {cm[0,1]:>10,}")
    print(f"   Actual 1  {cm[1,0]:>10,}   {cm[1,1]:>10,}")
    print(f"\n{report}")

    # Top 20 feature importances
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("   Top 20 feature importances:")
    for feat, imp in importances.head(20).items():
        bar = "█" * int(imp * 400)
        print(f"   {feat:<40} {imp:.4f}  {bar}")

    return roc_auc, avg_prec, report, importances


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # 1. Merge
    df = load_and_merge()

    # 2. Impute
    df = impute(df)

    # 3. Save merged dataset
    df.to_csv(DATASET_OUT, index=False)
    print(f"\n   Dataset saved → {DATASET_OUT}  ({DATASET_OUT.stat().st_size/1e6:.1f} MB)")

    # 4. Split
    X_train, X_test, y_train, y_test, feature_cols = temporal_split(df, TARGET)

    # 5. Train
    model = train_xgboost(X_train, y_train, X_test, y_test)

    # 6. Evaluate
    roc_auc, avg_prec, report, importances = evaluate(
        model, X_test, y_test, feature_cols
    )

    # 7. Save model & report
    model.save_model(str(MODEL_OUT))
    print(f"\n   Model saved → {MODEL_OUT}")

    report_text = (
        f"Target        : {TARGET}\n"
        f"Test from year: {TEST_YEAR}\n"
        f"ROC-AUC       : {roc_auc:.4f}\n"
        f"Avg Precision : {avg_prec:.4f}\n\n"
        f"{report}\n\n"
        f"Top 20 features:\n{importances.head(20).to_string()}\n"
    )
    REPORT_OUT.write_text(report_text)
    print(f"   Report saved → {REPORT_OUT}")


if __name__ == "__main__":
    main()
