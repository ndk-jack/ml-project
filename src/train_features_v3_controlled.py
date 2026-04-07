"""
train_features_v3_controlled.py — Controlled feature engineering comparison.

Apples-to-apples comparison using the exact V1 classifier settings:
  - RandomForestClassifier(n_estimators=200, balanced_subsample, random_state=42)
  - Same preprocessing, same split, same threshold tuning logic

Setups compared:
  A) baseline_v1  : V1 feature set (10 features)
  B) engineered   : V1 + dist_fault_km, IsCoastal, log_depth,
                    Depth_category, 4 missingness flags (18 features)
                    (Season excluded — shown to be noise in V3 experiment)

Merge strategy for engineered features:
  clean_updated.csv deduplicated on (Date, Lat, Lon, Depth),
  then left-joined to database_updated.csv on exact floats.
  Unmatched rows (~0.15%) are imputed by the pipeline.
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

RAW_DATA    = "data/database_updated.csv"
CLEAN_DATA  = "data/clean_updated.csv"
OUT_METRICS = "outputs/metrics/metrics_feature_v3_controlled.json"
OUT_MODEL   = "outputs/models/best_classifier_v3_controlled.joblib"

# ── V1 feature set (identical to train_time_split.py) ─────────────────────────

V1_NUMERIC = [
    "Latitude", "Longitude", "Depth",
    "Azimuthal Gap", "Horizontal Distance", "Root Mean Square",
    "Horizontal Error", "depthError", "Month",
]
V1_CAT = ["Magnitude Type"]

# Engineered additions (Season excluded — noise in V3)
ENG_NUMERIC_ADD = ["dist_fault_km", "IsCoastal", "log_depth",
                   "missing_azimuthal_gap", "missing_horizontal_distance",
                   "missing_horizontal_error", "missing_deptherror"]
ENG_CAT_ADD     = ["Depth_category"]

# V1 reference from metrics_time_split.json
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

def make_pipeline(numeric_cols, cat_cols):
    prep = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ], remainder="drop")
    return Pipeline([
        ("prep",  prep),
        # Exact V1 settings from metrics_time_split.json best classifier
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ── 1. Load raw data ──────────────────────────────────────────────────────────

t0_total = time.time()
step("Loading database_updated.csv...")
t0 = time.time()

df = pd.read_csv(RAW_DATA)
df = df[df["Type"].str.strip().str.lower() == "earthquake"].copy()

df["_dt"]    = pd.to_datetime(df["Date"], errors="coerce")
df["Year"]   = df["_dt"].dt.year.astype("Int64")
df["Month"]  = df["_dt"].dt.month.astype("Int64")
df           = df.dropna(subset=["Year", "Magnitude"])
df["Year"]   = df["Year"].astype(int)
df["Month"]  = df["Month"].astype(int)
df["target"] = (df["Magnitude"] >= 6.0).astype(int)
df["_norm_date"] = df["_dt"].dt.strftime("%m/%d/%Y")

print(f"  Loaded: {len(df):,} rows  ({elapsed(t0)})")


# ── 2. Merge pre-computed features from clean_updated.csv ─────────────────────

step("Merging engineered features from clean_updated.csv...")
t0 = time.time()

MERGE_KEY_CLEAN = ["Date", "Latitude", "Longitude", "Depth"]
MERGE_COLS      = MERGE_KEY_CLEAN + ["dist_fault_km", "IsCoastal", "Depth_category"]

clean = pd.read_csv(CLEAN_DATA, usecols=MERGE_COLS)
clean_dedup = clean.drop_duplicates(subset=MERGE_KEY_CLEAN, keep="first")

n_dupes_removed = len(clean) - len(clean_dedup)
if n_dupes_removed:
    print(f"  Removed {n_dupes_removed:,} duplicate keys from clean_updated "
          f"before merge (keeping first)")

df = df.merge(
    clean_dedup,
    left_on=["_norm_date", "Latitude", "Longitude", "Depth"],
    right_on=MERGE_KEY_CLEAN,
    how="left",
)

n_matched   = int(df["dist_fault_km"].notna().sum())
n_unmatched = int(df["dist_fault_km"].isna().sum())
match_pct   = n_matched / len(df) * 100
print(f"  Matched   : {n_matched:,} rows ({match_pct:.2f}%)")
print(f"  Unmatched : {n_unmatched:,} rows ({100-match_pct:.2f}%) → will be median-imputed")
print(f"  ({elapsed(t0)})")


# ── 3. Compute cheap engineered features in-memory ────────────────────────────

step("Computing in-memory features...")

df["log_depth"] = np.log1p(df["Depth"].clip(lower=0))

for col, flag in [
    ("Azimuthal Gap",       "missing_azimuthal_gap"),
    ("Horizontal Distance", "missing_horizontal_distance"),
    ("Horizontal Error",    "missing_horizontal_error"),
    ("depthError",          "missing_deptherror"),
]:
    df[flag] = df[col].isna().astype(int) if col in df.columns else 0

print(f"  log_depth + 4 missingness flags computed")


# ── 4. Time split ─────────────────────────────────────────────────────────────

step("Time split: train <=2018 | val 2019-2022 | test 2023-2025")

train_df = df[df["Year"] <= 2018]
val_df   = df[(df["Year"] >= 2019) & (df["Year"] <= 2022)]
test_df  = df[(df["Year"] >= 2023) & (df["Year"] <= 2025)]

for label, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"  {label:5s}: {len(split):,} rows  |  "
          f"dangerous {split['target'].mean()*100:.1f}%")


# ── 5. Define both setups ─────────────────────────────────────────────────────

ENG_NUMERIC = V1_NUMERIC + ENG_NUMERIC_ADD
ENG_CAT     = V1_CAT     + ENG_CAT_ADD

setups = [
    ("baseline_v1", V1_NUMERIC,  V1_CAT),
    ("engineered",  ENG_NUMERIC, ENG_CAT),
]

print(f"\n  baseline_v1 : {len(V1_NUMERIC)+len(V1_CAT)} features")
print(f"  engineered  : {len(ENG_NUMERIC)+len(ENG_CAT)} features  "
      f"(+{len(ENG_NUMERIC_ADD)+len(ENG_CAT_ADD)} engineered)")
print(f"  Added: {ENG_NUMERIC_ADD + ENG_CAT_ADD}")


# ── 6. Train, tune, evaluate ──────────────────────────────────────────────────

all_results  = {}
best_name, best_val_f2 = None, -np.inf
best_pipe, best_threshold_final = None, 0.5

for name, num_cols, cat_cols in setups:
    step(f"Setup [{name}]  —  {len(num_cols)+len(cat_cols)} features")
    t0 = time.time()

    all_cols = num_cols + cat_cols
    X_tr = train_df[all_cols];  y_tr = train_df["target"]
    X_v  = val_df[all_cols];    y_v  = val_df["target"]
    X_te = test_df[all_cols];   y_te = test_df["target"]

    pipe = make_pipeline(num_cols, cat_cols)
    pipe.fit(X_tr, y_tr)
    print(f"  Fitted in {elapsed(t0)}")

    val_proba   = pipe.predict_proba(X_v)[:, 1]
    threshold, _= tune_threshold(y_v, val_proba)
    print(f"  Threshold (val F2): {threshold:.3f}")

    val_pred   = (val_proba >= threshold).astype(int)
    test_proba = pipe.predict_proba(X_te)[:, 1]
    test_pred  = (test_proba >= threshold).astype(int)

    val_m  = report(y_v,  val_pred,  val_proba,  "val")
    test_m = report(y_te, test_pred, test_proba, "test")

    dv = {
        "val_prauc":  round(val_m["pr_auc"]  - V1_REF["val_prauc"],  4),
        "val_f2":     round(val_m["f2"]      - V1_REF["val_f2"],     4),
        "test_prauc": round(test_m["pr_auc"] - V1_REF["test_prauc"], 4),
        "test_f2":    round(test_m["f2"]     - V1_REF["test_f2"],    4),
    }
    verdict = ("IMPROVEMENT" if dv["val_prauc"] > 0 and dv["test_prauc"] > 0
               else "NO IMPROVEMENT" if dv["val_prauc"] <= 0 and dv["test_prauc"] <= 0
               else "MIXED")
    print(f"  Δ vs V1 — val PR-AUC={dv['val_prauc']:+.4f}  F2={dv['val_f2']:+.4f}  "
          f"| test PR-AUC={dv['test_prauc']:+.4f}  F2={dv['test_f2']:+.4f}  → {verdict}")

    all_results[name] = {
        "n_features": len(num_cols) + len(cat_cols),
        "threshold":  round(float(threshold), 4),
        "val":        val_m,
        "test":       test_m,
        "delta_vs_v1": dv,
        "verdict":    verdict,
    }

    if val_m["f2"] > best_val_f2:
        best_val_f2           = val_m["f2"]
        best_name             = name
        best_pipe             = pipe
        best_threshold_final  = threshold


# ── 7. Summary table ──────────────────────────────────────────────────────────

step("Summary")
print(f"\n  {'Setup':15s}  {'N feat':>6}  {'val PR-AUC':>11}  {'val F2':>7}  "
      f"{'test PR-AUC':>12}  {'test F2':>8}  {'Verdict'}")
print(f"  {'V1 reference':15s}  {'10':>6}  {V1_REF['val_prauc']:>11.4f}  "
      f"{V1_REF['val_f2']:>7.4f}  {V1_REF['test_prauc']:>12.4f}  "
      f"{V1_REF['test_f2']:>8.4f}  (reference)")
for n, r in all_results.items():
    vm, tm = r["val"], r["test"]
    print(f"  {n:15s}  {r['n_features']:>6}  {vm['pr_auc']:>11.4f}  "
          f"{vm['f2']:>7.4f}  {tm['pr_auc']:>12.4f}  {tm['f2']:>8.4f}  "
          f"{r['verdict']}")

print(f"\n  Best setup by val F2: {best_name}  (F2={best_val_f2:.4f})")

keep_alive = (
    all_results["engineered"]["delta_vs_v1"]["val_prauc"] > 0 and
    all_results["engineered"]["delta_vs_v1"]["test_prauc"] > 0
)
print(f"  Feature-engineering path: "
      f"{'KEEP ALIVE ✓' if keep_alive else 'DEAD END — no consistent improvement'}")


# ── 8. Save ───────────────────────────────────────────────────────────────────

step("Saving outputs...")

joblib.dump(best_pipe, OUT_MODEL)

metrics_out = {
    "split": {"train": "<=2018", "val": "2019-2022", "test": "2023-2025"},
    "classifier": {
        "n_estimators": 200,
        "class_weight": "balanced_subsample",
        "random_state": 42,
    },
    "merge": {
        "strategy": "dedup clean_updated on (Date,Lat,Lon,Depth) then left-join on exact floats",
        "matched": n_matched,
        "unmatched": n_unmatched,
        "match_pct": round(match_pct, 4),
    },
    "best_setup":     best_name,
    "best_threshold": round(float(best_threshold_final), 4),
    "v1_reference":   V1_REF,
    "results":        all_results,
    "feature_engineering_verdict":
        "keep_alive" if keep_alive else "dead_end",
}
with open(OUT_METRICS, "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"  Metrics → {OUT_METRICS}")
print(f"  Model   → {OUT_MODEL}")
print(f"\n{'=' * 60}")
print(f"DONE  —  total elapsed: {elapsed(t0_total)}")
print(f"{'=' * 60}")
