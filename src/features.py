"""
features.py — Feature engineering for earthquake sequence prediction.

Generates ~46 features per event:
  - Reference event features  (magnitude, depth, gap, rms, …)
  - Temporal window features  (count, mag stats, energy, b-value, …)
    over 4 time windows (1d, 7d, 30d, 90d) within a 200 km radius
  - Elapsed time since the previous event
  - Distance to the nearest tectonic plate boundary

Usage : python3 src/features.py
Output: data/features/features.csv
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from sklearn.neighbors import BallTree
from shapely.ops import unary_union
import warnings

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
RADIUS_KM     = 200.0
EARTH_R_KM    = 6371.0
WINDOWS_DAYS  = [1, 7, 30, 90]


# ── Helpers ───────────────────────────────────────────────────────────────────

def seismic_energy(mags: np.ndarray) -> float:
    """Proxy cumulative energy: sum(10^(1.5 * M))."""
    return float(np.sum(10 ** (1.5 * mags)))


def seismic_moment(mags: np.ndarray) -> float:
    """Cumulative seismic moment: sum(10^(1.5*M + 9.1))."""
    return float(np.sum(10 ** (1.5 * mags + 9.1)))


def b_value_mle(mags: np.ndarray) -> float:
    """
    Maximum-likelihood Gutenberg-Richter b-value.
    Requires at least 5 events; returns NaN otherwise.
    """
    if len(mags) < 5:
        return np.nan
    mc = mags.min()
    mean_m = mags.mean()
    denom = mean_m - mc
    if denom <= 0:
        return np.nan
    return float(np.log10(np.e) / denom)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Parse datetime from separate Date (MM/DD/YYYY) + Time (HH:MM:SS) columns
    df["datetime"] = pd.to_datetime(
        df["Date"].str.strip() + " " + df["Time"].str.strip(),
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime", "Latitude", "Longitude", "Magnitude"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Normalise column names
    rename = {
        "Latitude":              "latitude",
        "Longitude":             "longitude",
        "Depth":                 "depth",
        "Magnitude":             "magnitude",
        "Magnitude Type":        "mag_type",
        "Azimuthal Gap":         "gap",
        "Horizontal Distance":   "dmin",
        "Root Mean Square":      "rms",
        "Horizontal Error":      "h_error",
        "depthError":            "depth_error",
        "Magnitude Error":       "mag_error",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


# ── Plate boundary distance ───────────────────────────────────────────────────

def plate_boundary_distances(df: pd.DataFrame, geojson_path: Path) -> pd.Series:
    """Distance (km) from each event to the nearest tectonic plate boundary."""
    print("  Loading plate boundaries …")
    boundaries = gpd.read_file(geojson_path).to_crs("EPSG:4326")

    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    # Project both to a metre-based CRS
    points_m     = points.to_crs("EPSG:3857")
    boundaries_m = boundaries.to_crs("EPSG:3857")
    merged_boundary = unary_union(boundaries_m.geometry)

    print(f"  Computing distances for {len(points_m):,} points …")
    dist_m = points_m.geometry.apply(lambda p: p.distance(merged_boundary))
    return (dist_m / 1000.0).rename("dist_to_plate_boundary_km")


# ── Window feature computation ────────────────────────────────────────────────

def compute_all_window_features(
    df: pd.DataFrame,
    all_neighbors: list,          # pre-computed by BallTree.query_radius
    windows_days: list,
) -> pd.DataFrame:
    """
    For every event i, look at its spatial neighbours (pre-computed) that
    occurred BEFORE event i and within each time window, and compute stats.
    """
    times    = df["datetime"].values          # numpy datetime64
    mags     = df["magnitude"].values
    depths   = df["depth"].values

    rows = []
    n_events = len(df)

    for i in range(n_events):
        if i % 20_000 == 0:
            print(f"    {i:>7,} / {n_events:,}  ({100*i/n_events:.0f}%)")

        t_ref   = times[i]
        row     = {}

        # Neighbours (spatial) — exclude self
        nbr_idx = all_neighbors[i]
        nbr_idx = nbr_idx[nbr_idx != i]

        # ── Elapsed since last event (any magnitude) ──────────────────────
        if len(nbr_idx) > 0:
            past_mask  = times[nbr_idx] < t_ref
            past_idx   = nbr_idx[past_mask]
            if len(past_idx) > 0:
                last_t = times[past_idx].max()
                row["elapsed_since_last_s"] = float(
                    (t_ref - last_t) / np.timedelta64(1, "s")
                )
            else:
                row["elapsed_since_last_s"] = np.nan
        else:
            row["elapsed_since_last_s"] = np.nan

        # ── Per-window features ───────────────────────────────────────────
        for w in windows_days:
            horizon = np.timedelta64(w, "D")
            if len(nbr_idx) > 0:
                win_mask = (times[nbr_idx] < t_ref) & \
                           (times[nbr_idx] >= t_ref - horizon)
                win_idx  = nbr_idx[win_mask]
            else:
                win_idx = np.array([], dtype=int)

            n = len(win_idx)
            row[f"count_{w}d"]  = n
            row[f"rate_{w}d"]   = n / w          # events per day

            if n > 0:
                m = mags[win_idx]
                d = depths[win_idx]
                d_valid = d[~np.isnan(d)]

                row[f"mag_max_{w}d"]   = float(m.max())
                row[f"mag_mean_{w}d"]  = float(m.mean())
                row[f"mag_std_{w}d"]   = float(m.std()) if n > 1 else 0.0
                row[f"energy_{w}d"]    = seismic_energy(m)
                row[f"moment_{w}d"]    = seismic_moment(m)
                row[f"b_value_{w}d"]   = b_value_mle(m)
                row[f"depth_mean_{w}d"]= float(d_valid.mean()) if len(d_valid) > 0 else np.nan
                row[f"depth_std_{w}d"] = float(d_valid.std())  if len(d_valid) > 1 else 0.0
            else:
                for feat in ["mag_max", "mag_mean", "mag_std",
                             "b_value", "depth_mean", "depth_std"]:
                    row[f"{feat}_{w}d"] = np.nan
                row[f"energy_{w}d"]  = 0.0
                row[f"moment_{w}d"]  = 0.0

        rows.append(row)

    return pd.DataFrame(rows, index=df.index)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    catalog_path    = Path("data/database_updated.csv")
    boundaries_path = Path("data/external/PB2002_boundaries.json")
    output_dir      = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load catalog
    print("── 1. Loading catalog ──────────────────────────────────────────")
    df = load_catalog(catalog_path)
    print(f"   {len(df):,} events  |  "
          f"{df['datetime'].min().date()} → {df['datetime'].max().date()}")

    # 2. Reference-event features
    print("\n── 2. Reference-event features ─────────────────────────────────")
    mag_type_map = {t: i for i, t in
                    enumerate(df["mag_type"].fillna("unknown").unique())}
    df["mag_type_enc"] = df["mag_type"].fillna("unknown").map(mag_type_map)

    ref_cols = ["latitude", "longitude", "depth", "magnitude",
                "mag_type_enc", "gap", "dmin", "rms",
                "h_error", "depth_error", "mag_error"]
    ref_cols = [c for c in ref_cols if c in df.columns]
    ref_features = df[ref_cols].copy()
    print(f"   {len(ref_cols)} reference features: {ref_cols}")

    # 3. Build BallTree (spatial index) — computed once
    print("\n── 3. Building BallTree spatial index ──────────────────────────")
    lat_rad   = np.radians(df["latitude"].values)
    lon_rad   = np.radians(df["longitude"].values)
    coords_r  = np.column_stack([lat_rad, lon_rad])
    tree      = BallTree(coords_r, metric="haversine")

    # Pre-compute ALL spatial neighbours at once (much faster than per-event)
    print(f"   Querying {len(df):,} events within {RADIUS_KM} km …")
    radius_rad   = RADIUS_KM / EARTH_R_KM
    all_neighbors = tree.query_radius(coords_r, r=radius_rad)
    print(f"   Done.  Avg neighbours per event: "
          f"{np.mean([len(n) for n in all_neighbors]):.1f}")

    # 4. Window features
    print("\n── 4. Temporal window features ─────────────────────────────────")
    print(f"   Windows: {WINDOWS_DAYS} days  |  Radius: {RADIUS_KM} km")
    window_df = compute_all_window_features(df, all_neighbors, WINDOWS_DAYS)
    print(f"   Generated {window_df.shape[1]} window features.")

    # 5. Plate boundary distance
    print("\n── 5. Plate boundary distance ──────────────────────────────────")
    if boundaries_path.exists():
        plate_dist = plate_boundary_distances(df, boundaries_path)
    else:
        print(f"   WARNING: {boundaries_path} not found — feature set to NaN")
        plate_dist = pd.Series(np.nan, index=df.index,
                               name="dist_to_plate_boundary_km")

    # 6. Assemble
    print("\n── 6. Assembling feature matrix ────────────────────────────────")
    features = pd.concat([ref_features, window_df, plate_dist], axis=1)

    # Keep datetime + coords for joining with labels.csv
    features.insert(0, "datetime",      df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    features.insert(1, "ref_lat",       df["latitude"])
    features.insert(2, "ref_lon",       df["longitude"])

    # 7. Summary
    print(f"\n   Shape: {features.shape[0]:,} rows × {features.shape[1]} columns")
    nan_summary = features.isna().mean().sort_values(ascending=False)
    high_nan = nan_summary[nan_summary > 0.1]
    if len(high_nan) > 0:
        print("\n   ⚠ Columns with >10% NaN:")
        for col, pct in high_nan.items():
            print(f"     {col:<40} {pct:.1%}")

    # 8. Save
    out_path = output_dir / "features.csv"
    features.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n── Saved → {out_path}  ({size_mb:.1f} MB)")
    print(features.head(3).to_string())


if __name__ == "__main__":
    main()
