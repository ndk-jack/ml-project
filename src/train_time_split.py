"""
train_time_split.py — V1 baseline with temporal train/val/test split.

Split:
  train      : Year <= 2018
  validation : 2019 <= Year <= 2022
  test       : 2023 <= Year <= 2025
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH        = "data/database_updated.csv"
OUT_METRICS      = "outputs/metrics/metrics_time_split.json"
OUT_REG_PREDS    = "outputs/metrics/predictions_regression_test.csv"
OUT_CLF_PREDS    = "outputs/metrics/predictions_classification_test.csv"
OUT_BEST_REG     = "outputs/models/best_regressor.joblib"
OUT_BEST_CLF     = "outputs/models/best_classifier.joblib"

# ── Feature lists ─────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "Latitude", "Longitude", "Depth",
    "Azimuthal Gap", "Horizontal Distance", "Root Mean Square",
    "Horizontal Error", "depthError",
]
CATEGORICAL_FEATURES = ["Magnitude Type"]

EXPLICIT_DROPS = {
    "Magnitude",                  # regression target — leakage
    "Magnitude Error",            # post-hoc measurement of target
    "Magnitude Seismic Stations", # post-hoc measurement of target
    "Depth Seismic Stations",     # 57% null + correlated with Azimuthal Gap
    "ID", "updated", "place",
    "Source", "Status",
    "Location Source", "Magnitude Source",
    "Time",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def fbeta_score(precision, recall, beta=2.0):
    """F-beta from scalars (avoids sklearn import for a one-liner)."""
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def tune_threshold_f2(y_true, y_proba, thresholds=None):
    """Return threshold that maximises F2 on the given set."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 199)
    best_t, best_f2 = 0.5, 0.0
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f2 = fbeta_score(p, r, beta=2.0)
        if f2 > best_f2:
            best_f2, best_t = f2, t
    return best_t, best_f2


def reg_metrics(y_true, y_pred, label):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"    {label:10s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}


def clf_metrics(y_true, y_pred, label):
    p   = float(precision_score(y_true, y_pred, zero_division=0))
    r   = float(recall_score(y_true, y_pred, zero_division=0))
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    f2  = float(fbeta_score(p, r, beta=2.0))
    cm  = confusion_matrix(y_true, y_pred).tolist()
    print(f"    {label:10s}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  F2={f2:.3f}")
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "f2": round(f2, 4), "confusion_matrix": cm}


def make_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_cols),
        ],
        remainder="drop",
    )


def make_reg_pipeline(model, numeric_cols, categorical_cols):
    return Pipeline([
        ("prep",  make_preprocessor(numeric_cols, categorical_cols)),
        ("model", model),
    ])


def make_clf_pipeline(model, numeric_cols, categorical_cols):
    return Pipeline([
        ("prep",  make_preprocessor(numeric_cols, categorical_cols)),
        ("model", model),
    ])


# ── 1. Load & filter ──────────────────────────────────────────────────────────

print("=" * 60)
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Raw shape : {df.shape}")

if "Type" in df.columns:
    before = len(df)
    df = df[df["Type"].str.strip().str.lower() == "earthquake"].copy()
    print(f"  After Type filter : {len(df):,} rows (dropped {before - len(df):,})")

# ── 2. Parse dates & derive Year/Month ────────────────────────────────────────

df["_dt"]  = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["_dt"].dt.year.astype("Int64")
df["Month"] = df["_dt"].dt.month.astype("Int64")
df = df.dropna(subset=["Year", "Magnitude"])
df["Year"]  = df["Year"].astype(int)
df["Month"] = df["Month"].astype(int)
print(f"  After date parse   : {len(df):,} rows | years {df['Year'].min()}–{df['Year'].max()}")

# ── 3. Targets ────────────────────────────────────────────────────────────────

df["target_reg"] = df["Magnitude"].astype(float)
df["target_clf"] = (df["Magnitude"] >= 6.0).astype(int)

# ── 4. Feature selection ──────────────────────────────────────────────────────

available_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
available_cat     = [c for c in CATEGORICAL_FEATURES if c in df.columns]
missing_numeric   = [c for c in NUMERIC_FEATURES if c not in df.columns]

if missing_numeric:
    print(f"  Warning: numeric features not found, skipped: {missing_numeric}")

print(f"\nFeatures used:")
print(f"  Numeric ({len(available_numeric)}) : {available_numeric}")
print(f"  Categorical ({len(available_cat)}) : {available_cat}")

# ── 5. Time split ─────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
train_years = sorted(df[df["Year"] <= 2018]["Year"].unique())
val_years   = sorted(df[(df["Year"] >= 2019) & (df["Year"] <= 2022)]["Year"].unique())
test_years  = sorted(df[(df["Year"] >= 2023) & (df["Year"] <= 2025)]["Year"].unique())

print(f"Split (actual years present):")
print(f"  Train : {train_years[0]}–{train_years[-1]}  ({len(train_years)} years)")
print(f"  Val   : {val_years}")
print(f"  Test  : {test_years}")

train_mask = df["Year"] <= 2018
val_mask   = (df["Year"] >= 2019) & (df["Year"] <= 2022)
test_mask  = (df["Year"] >= 2023) & (df["Year"] <= 2025)

df_train = df[train_mask]
df_val   = df[val_mask]
df_test  = df[test_mask]

print(f"\n  Train : {len(df_train):>7,} rows  |  "
      f"dangerous {df_train['target_clf'].mean()*100:.1f}%")
print(f"  Val   : {len(df_val):>7,} rows  |  "
      f"dangerous {df_val['target_clf'].mean()*100:.1f}%")
print(f"  Test  : {len(df_test):>7,} rows  |  "
      f"dangerous {df_test['target_clf'].mean()*100:.1f}%")

X_train = df_train[available_numeric + available_cat + ["Month"]]
X_val   = df_val  [available_numeric + available_cat + ["Month"]]
X_test  = df_test [available_numeric + available_cat + ["Month"]]

yr_train, yr_val, yr_test = df_train["target_reg"], df_val["target_reg"], df_test["target_reg"]
yc_train, yc_val, yc_test = df_train["target_clf"], df_val["target_clf"], df_test["target_clf"]

all_numeric  = available_numeric + ["Month"]
all_cat      = available_cat

# ── 6. Regression ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("REGRESSION — predicting Magnitude")
print("=" * 60)

regressors = [
    ("LinearRegression",    LinearRegression()),
    ("RandomForest",        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ("ExtraTrees",          ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
]

reg_results = {}
best_reg_name, best_reg_val_r2, best_reg_pipe = None, -np.inf, None

for name, model in regressors:
    print(f"\n  [{name}]")
    pipe = make_reg_pipeline(model, all_numeric, all_cat)
    pipe.fit(X_train, yr_train)

    val_m  = reg_metrics(yr_val,  pipe.predict(X_val),  "val")
    test_m = reg_metrics(yr_test, pipe.predict(X_test), "test")
    reg_results[name] = {"val": val_m, "test": test_m}

    if val_m["R2"] > best_reg_val_r2:
        best_reg_val_r2  = val_m["R2"]
        best_reg_name    = name
        best_reg_pipe    = pipe

print(f"\n  Best regressor (val R²): {best_reg_name}  R²={best_reg_val_r2:.4f}")

# ── 7. Classification ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CLASSIFICATION — predicting Dangerous (Magnitude >= 6.0)")
print("=" * 60)

classifiers = [
    ("LogisticRegression",  LogisticRegression(
                                class_weight="balanced", max_iter=1000,
                                random_state=42, solver="saga", n_jobs=-1)),
    ("RandomForest",        RandomForestClassifier(
                                n_estimators=200, class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)),
    ("ExtraTrees",          ExtraTreesClassifier(
                                n_estimators=200, class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)),
]

clf_results = {}
best_clf_name, best_clf_f2, best_clf_pipe, best_threshold = None, -np.inf, None, 0.5

for name, model in classifiers:
    print(f"\n  [{name}]")
    pipe = make_clf_pipeline(model, all_numeric, all_cat)
    pipe.fit(X_train, yc_train)

    val_proba  = pipe.predict_proba(X_val)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]

    pr_auc_val = float(average_precision_score(yc_val, val_proba))
    print(f"    val PR-AUC : {pr_auc_val:.4f}")

    # Tune threshold on val using F2
    threshold, f2_val = tune_threshold_f2(yc_val, val_proba)
    print(f"    Best threshold (F2 on val) : {threshold:.2f}  F2={f2_val:.4f}")

    # Apply on val and test
    val_preds  = (val_proba  >= threshold).astype(int)
    test_preds = (test_proba >= threshold).astype(int)

    pr_auc_test = float(average_precision_score(yc_test, test_proba))
    print(f"    test PR-AUC : {pr_auc_test:.4f}")
    print(f"    Val  metrics :")
    val_m  = clf_metrics(yc_val,  val_preds,  "val")
    print(f"    Test metrics :")
    test_m = clf_metrics(yc_test, test_preds, "test")

    clf_results[name] = {
        "threshold": round(float(threshold), 4),
        "val":  {**val_m,  "pr_auc": round(pr_auc_val,  4)},
        "test": {**test_m, "pr_auc": round(pr_auc_test, 4)},
    }

    if val_m["f2"] > best_clf_f2:
        best_clf_f2    = val_m["f2"]
        best_clf_name  = name
        best_clf_pipe  = pipe
        best_threshold = threshold

print(f"\n  Best classifier (val F2): {best_clf_name}  F2={best_clf_f2:.4f}  threshold={best_threshold:.2f}")

# ── 8. Save predictions ───────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("Saving predictions & models...")

reg_test_preds = best_reg_pipe.predict(X_test)
pd.DataFrame({
    "Year":           df_test["Year"].values,
    "Magnitude_true": yr_test.values,
    "Magnitude_pred": reg_test_preds,
}).to_csv(OUT_REG_PREDS, index=False)

clf_test_proba = best_clf_pipe.predict_proba(X_test)[:, 1]
pd.DataFrame({
    "Year":           df_test["Year"].values,
    "Dangerous_true": yc_test.values,
    "proba":          clf_test_proba,
    "Dangerous_pred": (clf_test_proba >= best_threshold).astype(int),
}).to_csv(OUT_CLF_PREDS, index=False)

joblib.dump(best_reg_pipe, OUT_BEST_REG)
joblib.dump(best_clf_pipe, OUT_BEST_CLF)

# ── 9. Save metrics JSON ──────────────────────────────────────────────────────

metrics = {
    "split": {
        "train_years": f"{train_years[0]}–{train_years[-1]}",
        "val_years":   str(val_years),
        "test_years":  str(test_years),
        "train_n": len(df_train),
        "val_n":   len(df_val),
        "test_n":  len(df_test),
    },
    "features": {
        "numeric":     all_numeric,
        "categorical": all_cat,
        "total":       len(all_numeric) + len(all_cat),
    },
    "regression": {
        "best_model": best_reg_name,
        "results":    reg_results,
    },
    "classification": {
        "best_model":      best_clf_name,
        "best_threshold":  round(float(best_threshold), 4),
        "results":         clf_results,
    },
}

with open(OUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"  Metrics   → {OUT_METRICS}")
print(f"  Reg preds → {OUT_REG_PREDS}")
print(f"  Clf preds → {OUT_CLF_PREDS}")
print(f"  Models    → {OUT_BEST_REG}, {OUT_BEST_CLF}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
