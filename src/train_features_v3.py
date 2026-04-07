"""
train_features_v3.py — V3 feature engineering experiment.

Compares two setups on the same RandomForestClassifier:
  A) baseline_v1  : same features as train_time_split.py (V1)
  B) engineered   : V1 + dist_fault_km, IsCoastal, log1p(Depth),
                    Depth_category, Season, missingness flags

dist_fault_km and IsCoastal are read from data/clean_updated.csv
(already computed via GeoJSON + cKDTree in preprocess_updated.py).
All other engineered features are computed cheaply in-memory.

Same split: train <=2018, val 2019-2022, test 2023-2025.
Test set is never used for model selection.
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

RAW_DATA        = "data/database_updated.csv"
CLEAN_DATA      = "data/clean_updated.csv"
OUT_METRICS     = "outputs/metrics/metrics_feature_v3.json"
OUT_PREDS       = "outputs/metrics/predictions_feature_v3_test.csv"
OUT_MODEL       = "outputs/models/best_classifier_v3.joblib"

# ── V1 baseline feature sets (identical to train_time_split.py) ───────────────

V1_NUMERIC = [
    "Latitude", "Longitude", "Depth",
    "Azimuthal Gap", "Horizontal Distance", "Root Mean Square",
    "Horizontal Error", "depthError", "Month",
]
V1_CAT = ["Magnitude Type"]

# ── V1 reference metrics (from metrics_time_split.json) ───────────────────────

V1_REF = {"val_prauc": 0.3479, "val_f2": 0.546,
           "test_prauc": 0.3504, "test_f2": 0.520}

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
        ("model", RandomForestClassifier(
            n_estimators=100,
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
if "Type" in df.columns:
    df = df[df["Type"].str.strip().str.lower() == "earthquake"].copy()

df["_dt"]    = pd.to_datetime(df["Date"], errors="coerce")
df["Year"]   = df["_dt"].dt.year.astype("Int64")
df["Month"]  = df["_dt"].dt.month.astype("Int64")
df           = df.dropna(subset=["Year", "Magnitude"])
df["Year"]   = df["Year"].astype(int)
df["Month"]  = df["Month"].astype(int)
df["target"] = (df["Magnitude"] >= 6.0).astype(int)
df["_norm_date"] = df["_dt"].dt.strftime("%m/%d/%Y")

print(f"  Raw (filtered): {len(df):,} rows  ({elapsed(t0)})")


# ── 2. Merge pre-computed features from clean_updated.csv ─────────────────────

step("Merging dist_fault_km and IsCoastal from clean_updated.csv...")
t0 = time.time()

clean = pd.read_csv(CLEAN_DATA, usecols=["Date", "Latitude", "Longitude",
                                          "Magnitude", "dist_fault_km", "IsCoastal"])
clean = clean.rename(columns={"Date": "_norm_date"})

df = df.merge(
    clean[["_norm_date", "Latitude", "Longitude", "Magnitude",
           "dist_fault_km", "IsCoastal"]],
    on=["_norm_date", "Latitude", "Longitude", "Magnitude"],
    how="left",
)

n_merged  = df["dist_fault_km"].notna().sum()
n_missing = df["dist_fault_km"].isna().sum()
print(f"  Merged: {n_merged:,} rows  |  unmatched (will impute): {n_missing:,}  ({elapsed(t0)})")


# ── 3. Compute cheap engineered features in-memory ────────────────────────────

step("Computing in-memory engineered features...")

# log1p(Depth) — reduces right-skew
df["log_depth"] = np.log1p(df["Depth"].clip(lower=0))

# Depth_category (seismological classification)
df["depth_cat"] = pd.cut(
    df["Depth"],
    bins=[-np.inf, 70, 300, np.inf],
    labels=["superficiel", "intermediaire", "profond"],
).astype(str)

# Season from Month
season_map = {1:"Hiver",2:"Hiver",3:"Printemps",4:"Printemps",5:"Printemps",
              6:"Ete",7:"Ete",8:"Ete",9:"Automne",10:"Automne",11:"Automne",12:"Hiver"}
df["season"] = df["Month"].map(season_map)

# Missingness flags for sparse measurement-quality columns
SPARSE_COLS = ["Azimuthal Gap", "Horizontal Distance", "Horizontal Error", "depthError"]
for col in SPARSE_COLS:
    if col in df.columns:
        flag = f"missing_{col.lower().replace(' ', '_')}"
        df[flag] = df[col].isna().astype(int)

missing_flags = [f"missing_{c.lower().replace(' ', '_')}"
                 for c in SPARSE_COLS if c in df.columns]
print(f"  Engineered: log_depth, depth_cat, season, flags={missing_flags}")


# ── 4. Time split ─────────────────────────────────────────────────────────────

step("Applying time split (train <=2018, val 2019-2022, test 2023-2025)...")

train_df = df[df["Year"] <= 2018]
val_df   = df[(df["Year"] >= 2019) & (df["Year"] <= 2022)]
test_df  = df[(df["Year"] >= 2023) & (df["Year"] <= 2025)]

print(f"  Train {len(train_df):,} ({train_df['target'].mean()*100:.1f}% dangerous) | "
      f"Val {len(val_df):,} ({val_df['target'].mean()*100:.1f}%) | "
      f"Test {len(test_df):,} ({test_df['target'].mean()*100:.1f}%)")


# ── 5. Define both feature sets ───────────────────────────────────────────────

ENG_NUMERIC = V1_NUMERIC + ["dist_fault_km", "IsCoastal", "log_depth"] + missing_flags
ENG_CAT     = V1_CAT + ["depth_cat", "season"]

setups = [
    ("baseline_v1",  V1_NUMERIC,  V1_CAT),
    ("engineered",   ENG_NUMERIC, ENG_CAT),
]

print(f"\n  baseline_v1 : {len(V1_NUMERIC)} numeric + {len(V1_CAT)} cat = "
      f"{len(V1_NUMERIC)+len(V1_CAT)} features")
print(f"  engineered  : {len(ENG_NUMERIC)} numeric + {len(ENG_CAT)} cat = "
      f"{len(ENG_NUMERIC)+len(ENG_CAT)} features")
print(f"  New features: dist_fault_km, IsCoastal, log_depth, depth_cat, season, "
      f"{len(missing_flags)} missingness flags")


# ── 6. Train, tune threshold, evaluate ───────────────────────────────────────

all_results = {}
best_setup_name, best_val_f2, best_pipe = None, -np.inf, None
best_threshold_final = 0.5

for name, num_cols, cat_cols in setups:
    step(f"Setup: {name}  ({len(num_cols)+len(cat_cols)} features)")
    t0 = time.time()

    all_cols  = num_cols + cat_cols
    X_tr  = train_df[all_cols]
    X_v   = val_df[all_cols]
    X_te  = test_df[all_cols]
    y_tr  = train_df["target"]
    y_v   = val_df["target"]
    y_te  = test_df["target"]

    pipe = make_pipeline(num_cols, cat_cols)
    pipe.fit(X_tr, y_tr)
    print(f"  Fitted in {elapsed(t0)}")

    val_proba  = pipe.predict_proba(X_v)[:, 1]
    threshold, val_f2 = tune_threshold(y_v, val_proba)
    print(f"  Threshold (val F2): {threshold:.3f}")

    val_pred  = (val_proba  >= threshold).astype(int)
    test_proba = pipe.predict_proba(X_te)[:, 1]
    test_pred  = (test_proba >= threshold).astype(int)

    val_m  = report(y_v,  val_pred,  val_proba,  "val")
    test_m = report(y_te, test_pred, test_proba, "test")

    dv = {
        "val_prauc": round(val_m["pr_auc"] - V1_REF["val_prauc"],  4),
        "val_f2":    round(val_m["f2"]     - V1_REF["val_f2"],     4),
        "test_prauc":round(test_m["pr_auc"]- V1_REF["test_prauc"], 4),
        "test_f2":   round(test_m["f2"]    - V1_REF["test_f2"],    4),
    }
    print(f"  Δ vs V1 — val PR-AUC={dv['val_prauc']:+.3f}  F2={dv['val_f2']:+.3f}  "
          f"| test PR-AUC={dv['test_prauc']:+.3f}  F2={dv['test_f2']:+.3f}")

    all_results[name] = {
        "features": {"numeric": num_cols, "categorical": cat_cols,
                     "total": len(num_cols) + len(cat_cols)},
        "threshold": round(float(threshold), 4),
        "val":       val_m,
        "test":      test_m,
        "delta_vs_v1": dv,
    }

    if val_m["f2"] > best_val_f2:
        best_val_f2        = val_m["f2"]
        best_setup_name    = name
        best_pipe          = pipe
        best_threshold_final = threshold
        best_test_proba    = test_proba
        best_test_pred     = test_pred
        best_X_te          = X_te


# ── 7. Summary ────────────────────────────────────────────────────────────────

step("Summary")
print(f"  Best setup (val F2): {best_setup_name}  F2={best_val_f2:.4f}")
print(f"\n  {'Setup':15s}  {'val PR-AUC':>11}  {'val F2':>7}  "
      f"{'test PR-AUC':>12}  {'test F2':>8}")
print(f"  {'V1 reference':15s}  {V1_REF['val_prauc']:>11.4f}  "
      f"{V1_REF['val_f2']:>7.4f}  {V1_REF['test_prauc']:>12.4f}  "
      f"{V1_REF['test_f2']:>8.4f}")
for name, res in all_results.items():
    vm, tm = res["val"], res["test"]
    print(f"  {name:15s}  {vm['pr_auc']:>11.4f}  {vm['f2']:>7.4f}  "
          f"{tm['pr_auc']:>12.4f}  {tm['f2']:>8.4f}")


# ── 8. Save outputs ────────────────────────────────────────────────────────────

step("Saving outputs...")

pd.DataFrame({
    "Year":           test_df["Year"].values,
    "Magnitude":      test_df["Magnitude"].values,
    "Dangerous_true": test_df["target"].values,
    "proba":          best_test_proba,
    "Dangerous_pred": best_test_pred,
    "setup":          best_setup_name,
}).to_csv(OUT_PREDS, index=False)

joblib.dump(best_pipe, OUT_MODEL)

metrics_out = {
    "split":       {"train": "<=2018", "val": "2019-2022", "test": "2023-2025"},
    "best_setup":  best_setup_name,
    "best_threshold": round(float(best_threshold_final), 4),
    "v1_reference": V1_REF,
    "results":     all_results,
}
with open(OUT_METRICS, "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"  Metrics → {OUT_METRICS}")
print(f"  Preds   → {OUT_PREDS}")
print(f"  Model   → {OUT_MODEL}")

print(f"\n{'=' * 60}")
print(f"DONE  —  total elapsed: {elapsed(t0_total)}")
print(f"{'=' * 60}")
