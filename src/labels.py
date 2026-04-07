"""
labels.py — Binary label generation for earthquake sequence prediction.

For each earthquake (reference point, date), labels whether a significant
follow-up event (M >= mag_threshold) occurs within a spatial radius and
time horizon.

Usage : python3 src/labels.py
Output: data/features/labels.csv
        columns: ref_lat, ref_lon, ref_date, label_7d, label_30d, label_365d

Algorithm:
  - Unit-sphere cKDTree for spatial neighbor lookup (chord ≈ great-circle)
  - Batched queries (BATCH_SIZE events at a time) to bound peak memory
  - Single spatial pass computes all three horizons simultaneously
  - Forward-only labeling: only events strictly AFTER the reference date count

Runtime: ~1–5 min on 540 k events (depends on hardware / seismic density).
"""

import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_DATA   = "data/database_updated.csv"
OUT_DIR    = Path("data/features")
OUT_FILE   = OUT_DIR / "labels.csv"

R_EARTH_KM = 6371.0
RADIUS_KM  = 200.0
MAG_THRESH = 5.0
HORIZONS   = [7, 30, 365]   # days — all three always computed
BATCH_SIZE = 50_000          # events per spatial-query batch (bounds peak memory)


# ── Helpers ────────────────────────────────────────────────────────────────────
def ts():
    return datetime.now().strftime("%H:%M:%S")

def step(msg):
    print(f"\n[{ts()}] {msg}")

def elapsed(t0):
    return f"{time.time() - t0:.1f}s"


# ── Core function ──────────────────────────────────────────────────────────────
def make_labels(df, radius_km=200, horizon_days=30, mag_threshold=5.0):
    """
    Compute binary predictive labels for each earthquake.

    For each event, label_Xd = 1 if a M >= mag_threshold earthquake occurs
    within radius_km km in the strictly next X days (X ∈ {7, 30, 365}).

    Parameters
    ----------
    df            : DataFrame with columns Latitude, Longitude, _date, Magnitude.
                    _date must be parseable as datetime64[D].
    radius_km     : spatial search radius in km (default 200).
    horizon_days  : not used — all three standard horizons (7 / 30 / 365 d) are
                    always computed and returned.
    mag_threshold : minimum magnitude for the follow-up event (default 5.0).

    Returns
    -------
    DataFrame: ref_lat, ref_lon, ref_date, label_7d, label_30d, label_365d.
    """
    n = len(df)

    lats  = df["Latitude"].values.astype(np.float64)
    lons  = df["Longitude"].values.astype(np.float64)
    mags  = df["Magnitude"].values.astype(np.float32)
    dates = pd.to_datetime(df["_date"]).values.astype("datetime64[D]")

    # ── Spatial index ────────────────────────────────────────────────────────
    # Project to 3D unit sphere; Euclidean chord distance approximates
    # great-circle distance with negligible error at ≤200 km.
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    xyz = np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ])

    # Chord length for radius_km on a sphere of radius R_EARTH_KM
    chord = 2.0 * np.sin(radius_km / (2.0 * R_EARTH_KM))
    max_h = max(HORIZONS)

    print(f"  Building KD-tree on {n:,} points...")
    t_tree = time.time()
    tree   = cKDTree(xyz)
    print(f"  KD-tree ready  ({elapsed(t_tree)})")

    # ── Label arrays ─────────────────────────────────────────────────────────
    out = {h: np.zeros(n, dtype=np.int8) for h in HORIZONS}

    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Labeling {n:,} events in {n_batches} batches of ≤{BATCH_SIZE:,}...")
    t_loop = time.time()

    for b in range(n_batches):
        lo = b * BATCH_SIZE
        hi = min(lo + BATCH_SIZE, n)

        # Spatial neighbors for this batch — all times, all magnitudes
        # workers=-1 uses all CPU cores (requires scipy >= 1.6)
        nbr_lists = tree.query_ball_point(xyz[lo:hi], chord, workers=-1)

        for local_i, nbrs in enumerate(nbr_lists):
            if len(nbrs) <= 1:
                continue  # no neighbors other than self

            i = lo + local_i
            j = np.array(nbrs, dtype=np.int32)
            j = j[j != i]
            if len(j) == 0:
                continue

            # Signed delta in days: positive = j is in the future relative to i
            delta = (
                (dates[j] - dates[i]) / np.timedelta64(1, "D")
            ).astype(np.int32)

            # Keep strictly future events within max horizon above mag threshold
            mask = (delta > 0) & (delta <= max_h) & (mags[j] >= mag_threshold)
            if not np.any(mask):
                continue

            d = delta[mask]
            for h in HORIZONS:
                if np.any(d <= h):
                    out[h][i] = 1

        # Progress every 2 batches
        if (b + 1) % 2 == 0 or b == n_batches - 1:
            done = hi
            rate = done / max(time.time() - t_loop, 1e-9)
            eta  = (n - done) / rate
            print(f"    batch {b+1:>3}/{n_batches}  "
                  f"{done:>7,}/{n:,}  "
                  f"({elapsed(t_loop)}  ETA {eta:.0f}s)")

    return pd.DataFrame({
        "ref_lat":    lats,
        "ref_lon":    lons,
        "ref_date":   pd.to_datetime(dates),
        "label_7d":   out[7],
        "label_30d":  out[30],
        "label_365d": out[365],
    }, index=df.index)


# ── 1. Load data ───────────────────────────────────────────────────────────────
t0_total = time.time()
step("Loading data/database_updated.csv...")
t0 = time.time()

df = pd.read_csv(RAW_DATA)
df = df[df["Type"].str.strip().str.lower() == "earthquake"].copy()
df["_dt"]   = pd.to_datetime(df["Date"], errors="coerce")
df["_date"] = df["_dt"].dt.normalize()          # midnight datetime64[ns], day-resolution
df = df.dropna(subset=["_date", "Latitude", "Longitude", "Magnitude"])
df = df.reset_index(drop=True)

print(f"  {len(df):,} earthquakes  |  "
      f"date range: {df['_date'].min().date()} → {df['_date'].max().date()}  "
      f"({elapsed(t0)})")


# ── 2. Compute labels ──────────────────────────────────────────────────────────
step(f"Computing labels  (radius={RADIUS_KM} km, M≥{MAG_THRESH}, "
     f"horizons={HORIZONS} days)...")
t0 = time.time()

labels = make_labels(
    df,
    radius_km=RADIUS_KM,
    horizon_days=365,
    mag_threshold=MAG_THRESH,
)

print(f"\n  Done  ({elapsed(t0)})")


# ── 3. Label distribution ──────────────────────────────────────────────────────
step("Label distribution")

n_total = len(labels)
print(f"\n  {'Horizon':10s}  {'Positive':>9s}  {'%':>6s}  {'Negative':>9s}  {'%':>6s}")
print(f"  {'-'*10}  {'-'*9}  {'-'*6}  {'-'*9}  {'-'*6}")
for col in ["label_7d", "label_30d", "label_365d"]:
    n_pos = int(labels[col].sum())
    n_neg = n_total - n_pos
    pct_p = n_pos / n_total * 100
    pct_n = n_neg / n_total * 100
    print(f"  {col:10s}  {n_pos:>9,}  {pct_p:>6.2f}%  {n_neg:>9,}  {pct_n:>6.2f}%")


# ── 4. Save ────────────────────────────────────────────────────────────────────
step("Saving...")

OUT_DIR.mkdir(parents=True, exist_ok=True)
labels.to_csv(OUT_FILE, index=False)

print(f"  {len(labels):,} rows → {OUT_FILE}")
print(f"  Columns: {list(labels.columns)}")
print(f"  File size: {OUT_FILE.stat().st_size / 1e6:.1f} MB")

print(f"\n{'='*60}")
print(f"DONE  —  total elapsed: {elapsed(t0_total)}")
print(f"{'='*60}")
