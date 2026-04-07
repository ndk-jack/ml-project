"""
tune_classification_v2.py — V2 classification, RandomForest only.

Improvements over V1:
  - RandomizedSearchCV 5 iter × 2-fold on 20% subsample (≤120k rows)
  - Refit best params on full training set
  - Compare sigmoid vs isotonic calibration (cv="prefit" on held-out 20%)
  - Finer threshold search (500 points) on validation, optimise F2
  - No new dependencies, test set untouched until final evaluation

Expected runtime: ~3–5 min on a Mac.
"""

import json
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PATH   = "data/database_updated.csv"
OUT_METRICS = "outputs/metrics/metrics_classification_v2.json"
OUT_PREDS   = "outputs/metrics/predictions_classification_v2_test.csv"
OUT_MODEL   = "outputs/models/best_classifier_v2.joblib"

# ── Features (identical to V1) ────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "Latitude", "Longitude", "Depth",
    "Azimuthal Gap", "Horizontal Distance", "Root Mean Square",
    "Horizontal Error", "depthError", "Month",
]
CATEGORICAL_FEATURES = ["Magnitude Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# V1 baseline for delta reporting
V1_VAL_F2    = 0.546
V1_VAL_PRAUC = 0.348
V1_TEST_F2   = 0.520

# ── Helpers ────────────────────────────────────────────────────────────────────

def now():
    return datetime.now().strftime("%H:%M:%S")

def step(msg):
    print(f"\n[{now()}] {msg}")

def fbeta(p, r, beta=2.0):
    b2 = beta ** 2
    d = b2 * p + r
    return (1 + b2) * p * r / d if d > 0 else 0.0

def tune_threshold(y_true, y_proba, n_points=500, max_t=0.60):
    thresholds = np.linspace(0.01, max_t, n_points)
    best_t, best_f2 = 0.5, 0.0
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f2 = fbeta(p, r)
        if f2 > best_f2:
            best_f2, best_t = f2, t
    return best_t, best_f2

def report(y_true, y_pred, y_proba, label):
    p     = float(precision_score(y_true, y_pred, zero_division=0))
    r     = float(recall_score(y_true, y_pred, zero_division=0))
    f1    = float(f1_score(y_true, y_pred, zero_division=0))
    f2    = float(fbeta(p, r))
    prauc = float(average_precision_score(y_true, y_proba))
    cm    = confusion_matrix(y_true, y_pred).tolist()
    print(f"  {label:6s}  PR-AUC={prauc:.4f}  P={p:.3f}  R={r:.3f}  "
          f"F1={f1:.3f}  F2={f2:.3f}")
    return {"pr_auc": round(prauc, 4), "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(f1, 4), "f2": round(f2, 4),
            "confusion_matrix": cm}

def make_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ], remainder="drop")

def elapsed(t0):
    return f"{time.time() - t0:.1f}s"


# ── 1. Load & split ────────────────────────────────────────────────────────────

t0_total = time.time()
step("Loading and splitting data...")
t0 = time.time()

df = pd.read_csv(DATA_PATH)
if "Type" in df.columns:
    df = df[df["Type"].str.strip().str.lower() == "earthquake"].copy()

df["_dt"]    = pd.to_datetime(df["Date"], errors="coerce")
df["Year"]   = df["_dt"].dt.year.astype("Int64")
df["Month"]  = df["_dt"].dt.month.astype("Int64")
df = df.dropna(subset=["Year", "Magnitude"])
df["Year"]   = df["Year"].astype(int)
df["Month"]  = df["Month"].astype(int)
df["target"] = (df["Magnitude"] >= 6.0).astype(int)

train_df = df[df["Year"] <= 2018]
val_df   = df[(df["Year"] >= 2019) & (df["Year"] <= 2022)]
test_df  = df[(df["Year"] >= 2023) & (df["Year"] <= 2025)]

X_train, y_train = train_df[FEATURES], train_df["target"]
X_val,   y_val   = val_df[FEATURES],   val_df["target"]
X_test,  y_test  = test_df[FEATURES],  test_df["target"]

print(f"  Train {len(X_train):,} | Val {len(X_val):,} | Test {len(X_test):,}")
print(f"  Dangerous — train {y_train.mean()*100:.1f}%  "
      f"val {y_val.mean()*100:.1f}%  test {y_test.mean()*100:.1f}%")
print(f"  ({elapsed(t0)})")


# ── 2. Subsample training set for search ──────────────────────────────────────

step("Subsampling train for hyperparameter search (20%, max 120k rows)...")
t0 = time.time()

MAX_SEARCH_ROWS = 120_000
sample_frac = min(0.20, MAX_SEARCH_ROWS / len(X_train))

X_search, _, y_search, _ = train_test_split(
    X_train, y_train,
    train_size=sample_frac,
    stratify=y_train,
    random_state=42,
)
print(f"  Search subset: {len(X_search):,} rows  "
      f"({y_search.mean()*100:.1f}% dangerous)  ({elapsed(t0)})")


# ── 3. RandomizedSearchCV on subsample ────────────────────────────────────────

step("RandomizedSearchCV — 5 iter × 2-fold on subsample...")
t0 = time.time()

param_dist = {
    "model__n_estimators":     randint(100, 401),
    "model__max_depth":        [None, 20, 30, 40],
    "model__min_samples_leaf": randint(1, 11),
    "model__max_features":     ["sqrt", "log2", 0.3, 0.5],
}

search_pipe = Pipeline([
    ("prep",  make_preprocessor()),
    ("model", RandomForestClassifier(
        class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
])

search = RandomizedSearchCV(
    search_pipe,
    param_distributions=param_dist,
    n_iter=5,
    cv=2,
    scoring="average_precision",
    refit=False,        # we refit manually on full train below
    random_state=42,
    n_jobs=1,
    verbose=1,
)
search.fit(X_search, y_search)

best_params = {k.replace("model__", ""): v
               for k, v in search.best_params_.items()}
print(f"  Best CV PR-AUC : {search.best_score_:.4f}")
print(f"  Best params    : {best_params}")
print(f"  ({elapsed(t0)})")


# ── 4. Refit on full training set ─────────────────────────────────────────────

step("Refitting best RandomForest on full training set...")
t0 = time.time()

rf_best = RandomForestClassifier(
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
    **{k: v for k, v in best_params.items()},
)
full_pipe = Pipeline([
    ("prep",  make_preprocessor()),
    ("model", rf_best),
])
full_pipe.fit(X_train, y_train)
print(f"  Refit done.  ({elapsed(t0)})")


# ── 5. Calibration holdout split (from train) ─────────────────────────────────

step("Splitting train → fit_set (80%) + cal_holdout (20%) for calibration...")
t0 = time.time()

X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_train, y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=42,
)

# Fit a fresh RF (same best params) on the fit_set only, then calibrate on cal_holdout
rf_for_cal = RandomForestClassifier(
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
    **{k: v for k, v in best_params.items()},
)
pipe_for_cal = Pipeline([
    ("prep",  make_preprocessor()),
    ("model", rf_for_cal),
])
pipe_for_cal.fit(X_fit, y_fit)
print(f"  RF fitted on fit_set ({len(X_fit):,} rows).  ({elapsed(t0)})")


# ── 6. Compare calibration methods on validation ──────────────────────────────

step("Comparing sigmoid vs isotonic calibration (cv='prefit') on val PR-AUC...")
t0 = time.time()

calibration_results = {}
best_cal_method, best_cal_prauc, best_cal_pipe = None, -np.inf, None

for method in ("sigmoid", "isotonic"):
    cal_pipe = CalibratedClassifierCV(pipe_for_cal, method=method, cv="prefit")
    cal_pipe.fit(X_cal, y_cal)

    val_proba = cal_pipe.predict_proba(X_val)[:, 1]
    prauc_val = float(average_precision_score(y_val, val_proba))
    print(f"  {method:10s}  val PR-AUC = {prauc_val:.4f}")
    calibration_results[method] = {"val_prauc": round(prauc_val, 4)}

    if prauc_val > best_cal_prauc:
        best_cal_prauc  = prauc_val
        best_cal_method = method
        best_cal_pipe   = cal_pipe

print(f"  Best calibration: {best_cal_method}  ({elapsed(t0)})")


# ── 7. Threshold tuning on validation ─────────────────────────────────────────

step("Tuning decision threshold on validation (500 points, optimise F2)...")
t0 = time.time()

val_proba  = best_cal_pipe.predict_proba(X_val)[:, 1]
threshold, val_f2 = tune_threshold(y_val, val_proba)
print(f"  Optimal threshold: {threshold:.3f}  val F2={val_f2:.4f}  ({elapsed(t0)})")


# ── 8. Evaluate on validation and test ────────────────────────────────────────

step("Evaluating on validation and test sets...")

val_pred  = (val_proba >= threshold).astype(int)
val_m     = report(y_val, val_pred, val_proba, "val")

test_proba = best_cal_pipe.predict_proba(X_test)[:, 1]
test_pred  = (test_proba >= threshold).astype(int)
test_m     = report(y_test, test_pred, test_proba, "test")

# Delta vs V1
print(f"\n  Δ vs V1 baseline:")
print(f"  val  F2    {V1_VAL_F2:.3f} → {val_m['f2']:.3f}  "
      f"({val_m['f2'] - V1_VAL_F2:+.3f})")
print(f"  val  PR-AUC {V1_VAL_PRAUC:.3f} → {val_m['pr_auc']:.3f}  "
      f"({val_m['pr_auc'] - V1_VAL_PRAUC:+.3f})")
print(f"  test F2    {V1_TEST_F2:.3f} → {test_m['f2']:.3f}  "
      f"({test_m['f2'] - V1_TEST_F2:+.3f})")


# ── 9. Save outputs ────────────────────────────────────────────────────────────

step("Saving outputs...")

pd.DataFrame({
    "Year":           test_df["Year"].values,
    "Magnitude":      test_df["Magnitude"].values,
    "Dangerous_true": y_test.values,
    "proba":          test_proba,
    "Dangerous_pred": test_pred,
}).to_csv(OUT_PREDS, index=False)

joblib.dump(best_cal_pipe, OUT_MODEL)

metrics_out = {
    "split": {"train": "<=2018", "val": "2019-2022", "test": "2023-2025"},
    "features": FEATURES,
    "search": {
        "model": "RandomForestClassifier",
        "n_iter": 5, "cv": 2,
        "subsample_rows": len(X_search),
        "best_cv_prauc": round(float(search.best_score_), 4),
        "best_params": {k: (int(v) if hasattr(v, "item") else v)
                        for k, v in best_params.items()},
    },
    "calibration": {
        "methods_compared": calibration_results,
        "best_method": best_cal_method,
    },
    "threshold": round(float(threshold), 4),
    "val":  val_m,
    "test": test_m,
    "delta_vs_v1": {
        "val_f2":    round(val_m["f2"]    - V1_VAL_F2,    4),
        "val_prauc": round(val_m["pr_auc"] - V1_VAL_PRAUC, 4),
        "test_f2":   round(test_m["f2"]   - V1_TEST_F2,   4),
    },
}
with open(OUT_METRICS, "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"  Metrics → {OUT_METRICS}")
print(f"  Preds   → {OUT_PREDS}")
print(f"  Model   → {OUT_MODEL}")

print(f"\n{'=' * 60}")
print(f"DONE  —  total elapsed: {elapsed(t0_total)}")
print(f"{'=' * 60}")
