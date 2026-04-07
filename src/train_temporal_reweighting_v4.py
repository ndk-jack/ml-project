"""
train_temporal_reweighting_v4.py — V4 temporal sample weighting experiment.

Hypothesis: the class-distribution shift (train 3.1% dangerous → val/test 0.9%)
is caused by improved seismic monitoring, not real changes in earthquake hazard.
Old training data (1900–1960) has a 36–73% "dangerous" rate purely because small
quakes were not detected. Downweighting those years should shift the model toward
the modern distribution seen in val/test.

Weighting scheme (piecewise, applied to training set only):
  1900–1989 : 1.0  (pre-broadband, extremely biased detection)
  1990–2004 : 1.5  (early broadband / FDSN era)
  2005–2012 : 2.0  (modern global network)
  2013–2018 : 3.0  (most recent, closest to val/test distribution)

Classifier: exact V1 settings — RandomForest, n_estimators=200,
            balanced_subsample, random_state=42.

Setups compared:
  A) baseline_v1       — no temporal weighting
  B) temporal_weighted — piecewise bin weighting
"""

import json
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PATH   = "data/database_updated.csv"
OUT_METRICS = "outputs/metrics/metrics_temporal_reweighting_v4.json"
OUT_MODEL   = "outputs/models/best_classifier_v4.joblib"

# ── V1 feature set (identical to train_time_split.py) ─────────────────────────

NUMERIC_FEATURES = [
    "Latitude", "Longitude", "Depth",
    "Azimuthal Gap", "Horizontal Distance", "Root Mean Square",
    "Horizontal Error", "depthError", "Month",
]
CAT_FEATURES = ["Magnitude Type"]
ALL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES

# ── Temporal weight bins (training set only) ──────────────────────────────────

WEIGHT_BINS = [
    (1900, 1989, 1.0),
    (1990, 2004, 1.5),
    (2005, 2012, 2.0),
    (2013, 2018, 3.0),
]

# ── V1 reference metrics ───────────────────────────────────────────────────────

V1_REF = {"val_prauc": 0.3479, "val_f2": 0.5460,
          "test_prauc": 0.3504, "test_f2": 0.5200}

# ── Helpers ────────────────────────────────────────────────────────────────────

def ts():
    return datetime.now().strftime("%H:%M:%S")

def step(msg):
    print(f"\n[{ts()}] {msg}")

def elapsed(t0):
    return f"{time.time() - t0:.1f}s"

def fbeta(p, r, beta=2.0):
    b2 = beta ** 2
    d = b2 * p + r
    return (1 + b2) * p * r / d if d > 0 else 0.0

def tune_threshold(y_true, y_proba, n=500, max_t=0.60):
    best_t, best_f2 = 0.5, 0.0
    for t in np.linspace(0.01, max_t, n):
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
    print(f"    {label:6s}  PR-AUC={prauc:.4f}  P={p:.3f}  R={r:.3f}  "
          f"F1={f1:.3f}  F2={f2:.3f}")
    return {"pr_auc": round(prauc,4), "precision": round(p,4), "recall": round(r,4),
            "f1": round(f1,4), "f2": round(f2,4), "confusion_matrix": cm}

def make_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT_FEATURES),
    ], remainder="drop")

def make_rf():
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

def compute_temporal_weights(years):
    """Assign piecewise bin weights based on Year. Returns np.ndarray."""
    w = np.ones(len(years), dtype=float)
    for lo, hi, weight in WEIGHT_BINS:
        mask = (years >= lo) & (years <= hi)
        w[mask] = weight
    return w


# ── 1. Load & prepare data ────────────────────────────────────────────────────

t0_total = time.time()
step("Loading data...")
t0 = time.time()

df = pd.read_csv(DATA_PATH)
df = df[df["Type"].str.strip().str.lower() == "earthquake"].copy()
df["_dt"]    = pd.to_datetime(df["Date"], errors="coerce")
df["Year"]   = df["_dt"].dt.year.astype("Int64")
df["Month"]  = df["_dt"].dt.month.astype("Int64")
df           = df.dropna(subset=["Year", "Magnitude"])
df["Year"]   = df["Year"].astype(int)
df["Month"]  = df["Month"].astype(int)
df["target"] = (df["Magnitude"] >= 6.0).astype(int)

print(f"  Loaded: {len(df):,} rows  ({elapsed(t0)})")


# ── 2. Time split ─────────────────────────────────────────────────────────────

step("Time split: train <=2018 | val 2019-2022 | test 2023-2025")

train_df = df[df["Year"] <= 2018]
val_df   = df[(df["Year"] >= 2019) & (df["Year"] <= 2022)]
test_df  = df[(df["Year"] >= 2023) & (df["Year"] <= 2025)]

X_train = train_df[ALL_FEATURES];  y_train = train_df["target"]
X_val   = val_df[ALL_FEATURES];    y_val   = val_df["target"]
X_test  = test_df[ALL_FEATURES];   y_test  = test_df["target"]

for label, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"  {label:5s}: {len(split):,}  dangerous {split['target'].mean()*100:.2f}%")


# ── 3. Compute and inspect temporal weights ───────────────────────────────────

step("Computing temporal sample weights...")

sample_weights = compute_temporal_weights(train_df["Year"].values)

print(f"  Weight distribution in training set:")
for lo, hi, w in WEIGHT_BINS:
    mask = (train_df["Year"] >= lo) & (train_df["Year"] <= hi)
    n = mask.sum()
    dan_rate = train_df.loc[mask, "target"].mean() * 100
    print(f"    {lo}–{hi}  weight={w:.1f}  n={n:>7,}  dangerous={dan_rate:.2f}%")

# Effective dangerous rate under the weighting
eff_rate = np.average(y_train.values, weights=sample_weights)
print(f"\n  Raw dangerous rate in train  : {y_train.mean()*100:.2f}%")
print(f"  Weighted dangerous rate      : {eff_rate*100:.2f}%  "
      f"(target: val={y_val.mean()*100:.2f}%  test={y_test.mean()*100:.2f}%)")


# ── 4. Train both setups ──────────────────────────────────────────────────────

setups = [
    ("baseline_v1",       None),
    ("temporal_weighted", sample_weights),
]

all_results  = {}
best_name, best_val_f2, best_pipe, best_threshold_final = None, -np.inf, None, 0.5

for name, weights in setups:
    step(f"Setup [{name}]")
    t0 = time.time()

    prep = make_preprocessor()
    rf   = make_rf()

    # Fit preprocessor, then pass sample_weight only to the RF fit step
    X_tr_prep = prep.fit_transform(X_train, y_train)
    X_v_prep  = prep.transform(X_val)
    X_te_prep = prep.transform(X_test)

    fit_kwargs = {"sample_weight": weights} if weights is not None else {}
    rf.fit(X_tr_prep, y_train, **fit_kwargs)
    print(f"  Fitted in {elapsed(t0)}")

    val_proba   = rf.predict_proba(X_v_prep)[:, 1]
    threshold, _= tune_threshold(y_val, val_proba)
    print(f"  Threshold (val F2): {threshold:.3f}")

    val_pred   = (val_proba  >= threshold).astype(int)
    test_proba = rf.predict_proba(X_te_prep)[:, 1]
    test_pred  = (test_proba >= threshold).astype(int)

    val_m  = report(y_val,  val_pred,  val_proba,  "val")
    test_m = report(y_test, test_pred, test_proba, "test")

    dv = {
        "val_prauc":  round(val_m["pr_auc"]  - V1_REF["val_prauc"],  4),
        "val_f2":     round(val_m["f2"]      - V1_REF["val_f2"],     4),
        "test_prauc": round(test_m["pr_auc"] - V1_REF["test_prauc"], 4),
        "test_f2":    round(test_m["f2"]     - V1_REF["test_f2"],    4),
    }
    both_up = dv["val_prauc"] > 0 and dv["test_prauc"] > 0
    verdict = "IMPROVEMENT" if both_up else (
        "NO IMPROVEMENT" if dv["val_prauc"] <= 0 and dv["test_prauc"] <= 0
        else "MIXED"
    )
    print(f"  Δ vs V1  val  PR-AUC={dv['val_prauc']:+.4f}  F2={dv['val_f2']:+.4f}  "
          f"| test PR-AUC={dv['test_prauc']:+.4f}  F2={dv['test_f2']:+.4f}  → {verdict}")

    # Store prep + rf together for saving
    full_pipe = Pipeline([("prep", prep), ("model", rf)])
    all_results[name] = {
        "threshold": round(float(threshold), 4),
        "val":       val_m,
        "test":      test_m,
        "delta_vs_v1": dv,
        "verdict":   verdict,
    }

    if val_m["f2"] > best_val_f2:
        best_val_f2           = val_m["f2"]
        best_name             = name
        best_pipe             = full_pipe
        best_threshold_final  = threshold


# ── 5. Summary ────────────────────────────────────────────────────────────────

step("Summary")
print(f"\n  {'Setup':20s}  {'val PR-AUC':>11}  {'val F2':>7}  "
      f"{'test PR-AUC':>12}  {'test F2':>8}  Verdict")
print(f"  {'V1 reference':20s}  {V1_REF['val_prauc']:>11.4f}  "
      f"{V1_REF['val_f2']:>7.4f}  {V1_REF['test_prauc']:>12.4f}  "
      f"{V1_REF['test_f2']:>8.4f}  (reference)")
for n, r in all_results.items():
    vm, tm = r["val"], r["test"]
    print(f"  {n:20s}  {vm['pr_auc']:>11.4f}  {vm['f2']:>7.4f}  "
          f"{tm['pr_auc']:>12.4f}  {tm['f2']:>8.4f}  {r['verdict']}")

eng_verdict = all_results["temporal_weighted"]["verdict"]
print(f"\n  Temporal reweighting: {eng_verdict}")
print(f"  Best setup by val F2: {best_name}  (F2={best_val_f2:.4f})")


# ── 6. Save ───────────────────────────────────────────────────────────────────

step("Saving outputs...")

joblib.dump(best_pipe, OUT_MODEL)

metrics_out = {
    "split": {"train": "<=2018", "val": "2019-2022", "test": "2023-2025"},
    "classifier": {"n_estimators": 200, "class_weight": "balanced_subsample",
                   "random_state": 42},
    "weighting_scheme": {
        "method": "piecewise_bins",
        "bins": [{"from": lo, "to": hi, "weight": w} for lo, hi, w in WEIGHT_BINS],
        "raw_train_dangerous_pct":      round(float(y_train.mean() * 100), 4),
        "weighted_train_dangerous_pct": round(float(eff_rate * 100), 4),
        "val_dangerous_pct":            round(float(y_val.mean()   * 100), 4),
        "test_dangerous_pct":           round(float(y_test.mean()  * 100), 4),
    },
    "best_setup":     best_name,
    "best_threshold": round(float(best_threshold_final), 4),
    "v1_reference":   V1_REF,
    "results":        all_results,
    "verdict":        eng_verdict,
}
with open(OUT_METRICS, "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"  Metrics → {OUT_METRICS}")
print(f"  Model   → {OUT_MODEL}")
print(f"\n{'=' * 60}")
print(f"DONE  —  total elapsed: {elapsed(t0_total)}")
print(f"{'=' * 60}")
