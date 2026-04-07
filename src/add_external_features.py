"""
add_external_features.py — Enrich dataset with 3 external sources + retrain.

New features added:
  GEM Active Faults  → dist_to_nearest_fault_km, fault_slip_type_enc
  WSM2016            → stress_regime_enc, shmax_azimuth, wsm_quality_enc
  Background rate    → background_rate_yr, normalized_rate_30d

Usage : python3 src/add_external_features.py
Output:
  data/features/dataset_v3.csv
  models/lgbm_v3_label7d.txt
  reports/lgbm_v3_classification_report.txt
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from sklearn.neighbors import BallTree
from pathlib import Path
import warnings, subprocess, sys

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_IN   = Path("data/features/dataset_v2.csv")
DATASET_OUT  = Path("data/features/dataset_v3.csv")
MODEL_OUT    = Path("models/lgbm_v3_label7d.txt")
REPORT_OUT   = Path("reports/lgbm_v3_classification_report.txt")

GEM_FAULTS   = Path("data/external/gem_active_faults.geojson")
WSM_CSV      = Path("data/external/wsm2016.csv")

TARGET       = "label_7d"
TEST_YEAR    = 2010
RADIUS_KM    = 200.0
EARTH_R      = 6371.0
EPS          = 1e-9
RANDOM_STATE = 42


# ── 1. GEM Active Faults ──────────────────────────────────────────────────────

SLIP_TYPE_MAP = {
    "Normal":        1,
    "Reverse":       2,
    "Thrust":        2,
    "Strike-Slip":   3,
    "Sinistral":     3,
    "Dextral":       3,
    "Oblique":       4,
    "Blind Thrust":  2,
}

def add_gem_fault_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── GEM Active Faults ───────────────────────────────────────────")
    faults = gpd.read_file(GEM_FAULTS).to_crs("EPSG:4326")
    print(f"   Loaded {len(faults):,} fault segments")

    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["ref_lon"], df["ref_lat"]),
        crs="EPSG:4326",
    )
    points_m  = points.to_crs("EPSG:3857")
    faults_m  = faults.to_crs("EPSG:3857")
    merged    = unary_union(faults_m.geometry)

    print(f"   Computing distances for {len(points_m):,} events …")
    dist_km = points_m.geometry.apply(lambda p: p.distance(merged)) / 1000.0
    df["dist_to_nearest_fault_km"] = dist_km.values

    # Slip type: find nearest fault and read its slip_type attribute
    print("   Extracting slip types (nearest fault) …")
    # Build centroids for fault segments to find nearest
    fault_lats = faults_m.geometry.centroid.y.values
    fault_lons = faults_m.geometry.centroid.x.values
    fault_coords = np.column_stack([fault_lats, fault_lons])

    ev_x = points_m.geometry.x.values
    ev_y = points_m.geometry.y.values
    ev_coords = np.column_stack([ev_y, ev_x])   # (y, x) for KDTree in proj space

    from sklearn.neighbors import KDTree
    kt = KDTree(fault_coords)
    _, idx = kt.query(ev_coords, k=1)
    idx = idx.flatten()

    # Try common slip type column names in GEM GeoJSON
    slip_col = None
    for col in ["slip_type", "slip_type_1", "rake", "type"]:
        if col in faults.columns:
            slip_col = col
            break

    if slip_col:
        raw_types = faults[slip_col].iloc[idx].values
        enc = np.array([
            SLIP_TYPE_MAP.get(str(t).strip().title(), 0) for t in raw_types
        ])
    else:
        enc = np.zeros(len(df), dtype=int)

    df["fault_slip_type_enc"] = enc
    print(f"   dist_to_nearest_fault_km  mean={df['dist_to_nearest_fault_km'].mean():.1f} km")
    print(f"   fault_slip_type_enc       counts: {pd.Series(enc).value_counts().to_dict()}")
    return df


# ── 2. WSM2016 ────────────────────────────────────────────────────────────────

REGIME_MAP   = {"NF": 0, "TF": 1, "SS": 2, "NS": 3, "TS": 3, "U": 4}
QUALITY_MAP  = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}

def add_wsm_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── World Stress Map 2016 ────────────────────────────────────────")
    wsm = pd.read_csv(WSM_CSV, low_memory=False, encoding="latin-1")
    wsm.columns = wsm.columns.str.strip()

    # Keep only quality A/B/C and valid lat/lon/AZI
    wsm = wsm[wsm["QUALITY"].isin(["A", "B", "C"])].copy()
    wsm = wsm.dropna(subset=["LAT", "LON", "AZI"])
    print(f"   {len(wsm):,} WSM records after quality filter (A/B/C)")

    # Build BallTree on WSM coordinates
    wsm_rad  = np.radians(wsm[["LAT", "LON"]].values)
    ev_rad   = np.radians(df[["ref_lat", "ref_lon"]].values)
    tree     = BallTree(wsm_rad, metric="haversine")

    # For each event: find nearest WSM record within 500 km
    max_dist_rad = 500.0 / EARTH_R
    print(f"   Querying nearest WSM within 500 km …")
    dist_arr, idx_arr = tree.query(ev_rad, k=1)
    dist_arr = dist_arr.flatten() * EARTH_R   # in km
    idx_arr  = idx_arr.flatten()

    # Mask events with no WSM within 500 km → fill with defaults
    no_wsm = dist_arr > 500.0

    regimes    = wsm["REGIME"].iloc[idx_arr].values
    azis       = wsm["AZI"].iloc[idx_arr].values.astype(float)
    qualities  = wsm["QUALITY"].iloc[idx_arr].values

    regime_enc  = np.array([REGIME_MAP.get(str(r).strip(), 4) for r in regimes])
    quality_enc = np.array([QUALITY_MAP.get(str(q).strip(), 0) for q in qualities])

    # Convert azimuth to sin/cos so 0° and 360° are the same
    azi_sin = np.sin(np.radians(azis))
    azi_cos = np.cos(np.radians(azis))

    # Apply mask for no WSM
    regime_enc[no_wsm]  = 4      # unknown
    quality_enc[no_wsm] = 0
    azi_sin[no_wsm]     = 0.0
    azi_cos[no_wsm]     = 0.0

    df["stress_regime_enc"] = regime_enc
    df["shmax_sin"]         = azi_sin
    df["shmax_cos"]         = azi_cos
    df["wsm_quality_enc"]   = quality_enc
    df["wsm_dist_km"]       = dist_arr

    print(f"   stress_regime_enc counts: {pd.Series(regime_enc).value_counts().to_dict()}")
    print(f"   Events without WSM within 500 km: {no_wsm.sum():,} ({no_wsm.mean():.1%})")
    return df


# ── 3. Background seismicity rate ─────────────────────────────────────────────

def add_background_rate(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Background seismicity rate ───────────────────────────────────")

    # Use pre-2010 catalog as historical reference (avoid test-set leakage)
    times  = pd.to_datetime(df["datetime"])
    is_ref = times.dt.year < TEST_YEAR

    lat_rad  = np.radians(df["ref_lat"].values)
    lon_rad  = np.radians(df["ref_lon"].values)
    coords   = np.column_stack([lat_rad, lon_rad])
    tree     = BallTree(coords, metric="haversine")

    # Total reference period (1900 → 2009)
    t_min  = times[is_ref].min()
    t_max  = times[is_ref].max()
    span_yr = (t_max - t_min).days / 365.25
    print(f"   Reference period: {t_min.date()} → {t_max.date()} ({span_yr:.1f} yr)")

    # Only count M≥3 events in reference period
    ref_mask = is_ref.values & (df["magnitude"].values >= 3.0)
    ref_coords = coords[ref_mask]

    # For each event, count how many reference events are within 200 km
    radius_rad = RADIUS_KM / EARTH_R
    print(f"   Computing background counts for {len(df):,} events …")
    ref_tree   = BallTree(ref_coords, metric="haversine")
    bg_counts  = ref_tree.query_radius(coords, r=radius_rad, count_only=True)

    background_rate = bg_counts / span_yr   # events per year
    df["background_rate_yr"] = background_rate

    # Normalized recent rate: how anomalous is count_30d vs background?
    expected_30d = background_rate * (30.0 / 365.25) + EPS
    df["normalized_rate_30d"] = df["count_30d"] / expected_30d

    # Clip extremes
    df["normalized_rate_30d"] = df["normalized_rate_30d"].clip(0, 200)

    print(f"   background_rate_yr   mean={background_rate.mean():.1f}  "
          f"median={np.median(background_rate):.1f}")
    print(f"   normalized_rate_30d  mean={df['normalized_rate_30d'].mean():.2f}  "
          f"std={df['normalized_rate_30d'].std():.2f}")
    return df


# ── 4. Train LightGBM & compare ───────────────────────────────────────────────

def train_and_compare(df: pd.DataFrame, new_features: list):
    import lightgbm as lgb
    from sklearn.metrics import (
        roc_auc_score, classification_report,
        average_precision_score,
    )

    drop_cols = ["datetime", "ref_lat", "ref_lon", "latitude", "longitude",
                 "label_7d", "label_30d", "label_365d"]

    df["_year"]   = pd.to_datetime(df["datetime"]).dt.year
    train_mask    = df["_year"] < TEST_YEAR
    test_mask     = df["_year"] >= TEST_YEAR

    def run(cols, label):
        X_tr = df.loc[train_mask, cols]
        y_tr = df.loc[train_mask, TARGET]
        X_te = df.loc[test_mask,  cols]
        y_te = df.loc[test_mask,  TARGET]

        neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        model = lgb.LGBMClassifier(
            objective="binary", metric="auc",
            n_estimators=1000, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, scale_pos_weight=neg/pos,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(100)])

        yp   = model.predict_proba(X_te)[:, 1]
        roc  = roc_auc_score(y_te, yp)
        prec = average_precision_score(y_te, yp)
        rep  = classification_report(y_te, model.predict(X_te),
                                     target_names=["No event","Event ≥M5"])
        print(f"\n   [{label}]")
        print(f"   ROC-AUC : {roc:.4f}   Avg Precision: {prec:.4f}   "
              f"Best iter: {model.best_iteration_}")
        print(rep)
        return model, roc, prec, rep

    feat_v2 = [c for c in df.columns if c not in drop_cols + new_features + ["_year"]]
    feat_v3 = feat_v2 + new_features

    print("\n── Training ────────────────────────────────────────────────────")
    print("   v2 baseline (55 features) …")
    model_v2, roc_v2, prec_v2, rep_v2 = run(feat_v2, "v2 baseline — 55 features")

    print("   v3 + external features …")
    model_v3, roc_v3, prec_v3, rep_v3 = run(feat_v3, f"v3 — {len(feat_v3)} features (+external)")

    # Delta
    d_roc  = roc_v3 - roc_v2
    d_prec = prec_v3 - prec_v2
    print("\n── Results ─────────────────────────────────────────────────────")
    print(f"   ROC-AUC       v2={roc_v2:.4f}  v3={roc_v3:.4f}  "
          f"{'▲' if d_roc>0 else '▼'} {abs(d_roc):.4f}")
    print(f"   Avg Precision v2={prec_v2:.4f}  v3={prec_v3:.4f}  "
          f"{'▲' if d_prec>0 else '▼'} {abs(d_prec):.4f}")

    # New feature ranks
    imp = pd.Series(model_v3.feature_importances_, index=feat_v3).sort_values(ascending=False)
    print("\n   New feature ranks in v3:")
    for f in new_features:
        if f in imp.index:
            rank = list(imp.index).index(f) + 1
            print(f"   #{rank:<3} {f:<35} {imp[f]:>8.0f}")

    # Save
    model_v3.booster_.save_model(str(MODEL_OUT))
    REPORT_OUT.write_text(
        f"v2 ROC-AUC: {roc_v2:.4f}  →  v3 ROC-AUC: {roc_v3:.4f}  "
        f"({'▲' if d_roc>0 else '▼'}{abs(d_roc):.4f})\n\n{rep_v3}\n\n"
        f"Top 20:\n{imp.head(20).to_string()}\n"
    )
    print(f"\n   Model  → {MODEL_OUT}")
    print(f"   Report → {REPORT_OUT}")
    return roc_v3


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    print("── Loading dataset_v2 ──────────────────────────────────────────")
    df = pd.read_csv(DATASET_IN, low_memory=False)
    print(f"   {df.shape[0]:,} rows × {df.shape[1]} cols")

    new_features = []

    if GEM_FAULTS.exists():
        df = add_gem_fault_features(df)
        new_features += ["dist_to_nearest_fault_km", "fault_slip_type_enc"]
    else:
        print(f"\n   SKIP: {GEM_FAULTS} not found")

    if WSM_CSV.exists():
        df = add_wsm_features(df)
        new_features += ["stress_regime_enc", "shmax_sin", "shmax_cos",
                         "wsm_quality_enc", "wsm_dist_km"]
    else:
        print(f"\n   SKIP: {WSM_CSV} not found")

    df = add_background_rate(df)
    new_features += ["background_rate_yr", "normalized_rate_30d"]

    print(f"\n── Saving dataset_v3 ({len(new_features)} new features) ────────")
    df.to_csv(DATASET_OUT, index=False)
    print(f"   Saved → {DATASET_OUT}  ({DATASET_OUT.stat().st_size/1e6:.0f} MB)")
    print(f"   New features: {new_features}")

    train_and_compare(df, new_features)


if __name__ == "__main__":
    main()
