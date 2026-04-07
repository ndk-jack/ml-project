"""
train_feature_ablation_v5.py — Feature ablation study.

Isolates which engineered feature(s) are responsible for the +0.020 test F2
gain seen in the V3.1 controlled experiment.

Setups (all using the same RF / preprocessing / merge logic):
  1. baseline_v1                          (10 features)
  2. baseline_v1 + dist_fault_km          (11 features)
  3. baseline_v1 + IsCoastal              (11 features)
  4. baseline_v1 + dist_fault_km + IsCoastal  (12 features)
  5. baseline_v1 + full V3.1 set          (18 features)

Merge: identical to train_features_v3_controlled.py
  - dedup clean_updated on (Date, Lat, Lon, Depth)
  - left-join on exact floats, no Magnitude in key
  - unmatched rows imputed by pipeline

Verdict logic: IMPROVEMENT requires both val AND test PR-AUC > V1 reference.
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
OUT_METRICS = "outputs/metrics/metrics_feature_ablation_v5.json"
OUT_MODEL   = "outputs/models/best_classifier_v5.joblib"

# ── Feature sets ───────────────────────────────────────────────────────────────

V1_NUMERIC = [
    "Latitude", "Longitude", "Depth",
    "Azimuthal Gap", "Horizontal Distance", "Root Mean Square",
    "Horizontal Error", "depthError", "Month",
]
V1_CAT = ["Magnitude Type"]

FULL_ENG_NUMERIC_ADD = [
    "dist_fault_km", "IsCoastal", "log_depth",
    "missing_azimuthal_gap", "missing_horizontal_distance",
    "missing_horizontal_error", "missing_deptherror",
]
FULL_ENG_CAT_ADD = ["Depth_category"]

# The 5 ablation setups: (name, extra_numeric, extra_cat)
SETUPS = [
    ("baseline_v1",              [],                          []),
    ("+dist_fault_km",           ["dist_fault_km"],           []),
    ("+IsCoastal",               ["IsCoastal"],               []),
    ("+dist_fault_km+IsCoastal", ["dist_fault_km","IsCoastal"],[]),
    ("+full_v3_engineered",      FULL_ENG_NUMERIC_ADD,        FULL_ENG_CAT_ADD),
]

# V1 reference (from metrics_feature_v3_controlled.json baseline_v1 run)
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
        ("prep", prep),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

def verdict(dv):
    if dv["val_prauc"] > 0 and dv["test_prauc"] > 0:
        return "IMPROVEMENT"
    if dv["val_prauc"] <= 0 and dv["test_prauc"] <= 0:
        return "NO IMPROVEMENT"
    return "MIXED"


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

MERGE_KEY  = ["Date", "Latitude", "Longitude", "Depth"]
MERGE_COLS = MERGE_KEY + ["dist_fault_km", "IsCoastal", "Depth_category"]

clean       = pd.read_csv(CLEAN_DATA, usecols=MERGE_COLS)
clean_dedup = clean.drop_duplicates(subset=MERGE_KEY, keep="first")

n_dupes = len(clean) - len(clean_dedup)
if n_dupes:
    print(f"  Removed {n_dupes:,} duplicate keys from clean_updated (keeping first)")

df = df.merge(clean_dedup, left_on=["_norm_date","Latitude","Longitude","Depth"],
              right_on=MERGE_KEY, how="left")

n_matched   = int(df["dist_fault_km"].notna().sum())
n_unmatched = int(df["dist_fault_km"].isna().sum())
match_pct   = n_matched / len(df) * 100
print(f"  Matched   : {n_matched:,} ({match_pct:.2f}%)")
print(f"  Unmatched : {n_unmatched:,} ({100-match_pct:.2f}%) → median-imputed by pipeline")
print(f"  ({elapsed(t0)})")


# ── 3. Compute cheap in-memory features ───────────────────────────────────────

step("Computing in-memory features...")

df["log_depth"] = np.log1p(df["Depth"].clip(lower=0))

for col, flag in [("Azimuthal Gap",       "missing_azimuthal_gap"),
                  ("Horizontal Distance", "missing_horizontal_distance"),
                  ("Horizontal Error",    "missing_horizontal_error"),
                  ("depthError",          "missing_deptherror")]:
    df[flag] = df[col].isna().astype(int) if col in df.columns else 0

print(f"  log_depth + 4 missingness flags computed")


# ── 4. Time split ─────────────────────────────────────────────────────────────

step("Time split: train <=2018 | val 2019-2022 | test 2023-2025")

train_df = df[df["Year"] <= 2018]
val_df   = df[(df["Year"] >= 2019) & (df["Year"] <= 2022)]
test_df  = df[(df["Year"] >= 2023) & (df["Year"] <= 2025)]

for lbl, s in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"  {lbl:5s}: {len(s):,}  dangerous {s['target'].mean()*100:.2f}%")


# ── 5. Ablation loop ──────────────────────────────────────────────────────────

all_results  = {}
best_name, best_val_f2, best_pipe, best_threshold_final = None, -np.inf, None, 0.5

for name, extra_num, extra_cat in SETUPS:
    num_cols = V1_NUMERIC + extra_num
    cat_cols = V1_CAT + extra_cat
    n_feat   = len(num_cols) + len(cat_cols)

    step(f"[{name}]  —  {n_feat} features")
    t0 = time.time()

    X_tr = train_df[num_cols + cat_cols];  y_tr = train_df["target"]
    X_v  = val_df[num_cols + cat_cols];    y_v  = val_df["target"]
    X_te = test_df[num_cols + cat_cols];   y_te = test_df["target"]

    pipe = make_pipeline(num_cols, cat_cols)
    pipe.fit(X_tr, y_tr)
    print(f"  Fitted in {elapsed(t0)}")

    val_proba   = pipe.predict_proba(X_v)[:, 1]
    threshold, _= tune_threshold(y_v, val_proba)
    print(f"  Threshold (val F2): {threshold:.3f}")

    val_pred   = (val_proba  >= threshold).astype(int)
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
    verd = verdict(dv)
    print(f"  Δ vs V1  val PR-AUC={dv['val_prauc']:+.4f}  F2={dv['val_f2']:+.4f}  "
          f"| test PR-AUC={dv['test_prauc']:+.4f}  F2={dv['test_f2']:+.4f}  → {verd}")

    all_results[name] = {
        "n_features": n_feat,
        "added":      extra_num + extra_cat,
        "threshold":  round(float(threshold), 4),
        "val":        val_m,
        "test":       test_m,
        "delta_vs_v1": dv,
        "verdict":    verd,
    }

    if val_m["f2"] > best_val_f2:
        best_val_f2           = val_m["f2"]
        best_name             = name
        best_pipe             = pipe
        best_threshold_final  = threshold


# ── 6. Summary table ──────────────────────────────────────────────────────────

step("Ablation summary")

W = 28
print(f"\n  {'Setup':{W}s}  {'N':>4}  {'val AUC':>8}  {'val F2':>7}  "
      f"{'tst AUC':>8}  {'tst F2':>7}  Verdict")
print(f"  {'V1 reference':{W}s}  {'10':>4}  "
      f"{V1_REF['val_prauc']:>8.4f}  {V1_REF['val_f2']:>7.4f}  "
      f"{V1_REF['test_prauc']:>8.4f}  {V1_REF['test_f2']:>7.4f}  (reference)")
for n, r in all_results.items():
    vm, tm = r["val"], r["test"]
    print(f"  {n:{W}s}  {r['n_features']:>4}  "
          f"{vm['pr_auc']:>8.4f}  {vm['f2']:>7.4f}  "
          f"{tm['pr_auc']:>8.4f}  {tm['f2']:>7.4f}  {r['verdict']}")

# Ablation verdict
improvements = [n for n, r in all_results.items()
                if r["verdict"] == "IMPROVEMENT" and n != "baseline_v1"]
mixed        = [n for n, r in all_results.items()
                if r["verdict"] == "MIXED" and n != "baseline_v1"]

print(f"\n  IMPROVEMENT setups : {improvements if improvements else 'none'}")
print(f"  MIXED setups       : {mixed if mixed else 'none'}")

if improvements:
    ablation_verdict = f"keep_alive — consistent gain in: {improvements}"
elif mixed:
    ablation_verdict = f"weak_signal — mixed results in: {mixed}"
else:
    ablation_verdict = "dead_end — no consistent improvement from any feature addition"

print(f"  Ablation verdict   : {ablation_verdict}")
print(f"  Best by val F2     : {best_name}  (F2={best_val_f2:.4f})")


# ── 7. Save ───────────────────────────────────────────────────────────────────

step("Saving outputs...")

joblib.dump(best_pipe, OUT_MODEL)

metrics_out = {
    "split":      {"train": "<=2018", "val": "2019-2022", "test": "2023-2025"},
    "classifier": {"n_estimators": 200, "class_weight": "balanced_subsample",
                   "random_state": 42},
    "merge":      {"matched": n_matched, "unmatched": n_unmatched,
                   "match_pct": round(match_pct, 4)},
    "best_setup":      best_name,
    "best_threshold":  round(float(best_threshold_final), 4),
    "v1_reference":    V1_REF,
    "results":         all_results,
    "ablation_verdict": ablation_verdict,
}
with open(OUT_METRICS, "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"  Metrics → {OUT_METRICS}")
print(f"  Model   → {OUT_MODEL}")
print(f"\n{'=' * 60}")
print(f"DONE  —  total elapsed: {elapsed(t0_total)}")
print(f"{'=' * 60}")
