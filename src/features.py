"""
features.py — Feature engineering for earthquake sequence prediction.

Two catalogs are loaded:
  - data/database_updated.csv        → target events (output rows)
  - data/raw/catalog_m2_m4.csv       → contextual M≥2 neighbours (post-1999)

Feature sets produced per target event:
  A) Reference features        (magnitude, depth, gap, rms, …)
  B) Coherent window features  — neighbours filtered to M≥4.0 (combined catalog)
       count_Nd, rate_Nd, mag_max_Nd, mag_mean_Nd, mag_std_Nd,
       energy_Nd, moment_Nd, b_value_Nd, depth_mean_Nd, depth_std_Nd
       elapsed_since_last_s
       Windows: 1d, 7d, 30d, 90d
  C) M2-enriched window features — neighbours from catalog_m2_m4 only
       count_Nd_m2, b_value_Nd_m2, energy_Nd_m2
       Windows: 7d, 30d, 90d  (NaN for target events before 2000-01-01)
  D) Distance to nearest tectonic plate boundary

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

def compute_coherent_window_features(
    df_all: pd.DataFrame,
    target_indices: np.ndarray,
    all_neighbors: list,
    windows_days: list,
) -> pd.DataFrame:
    """
    Coherent windows (feature set B).

    Neighbours are filtered to M≥4.0 events from the combined catalog.
    Feature names are unchanged: count_Nd, rate_Nd, mag_max_Nd, …
    """
    times   = df_all["datetime"].values
    mags    = df_all["magnitude"].values
    depths  = df_all["depth"].values
    m4_mask = mags >= 4.0                    # boolean index over df_all rows

    rows     = []
    n_target = len(target_indices)

    for k, i in enumerate(target_indices):
        if k % 20_000 == 0:
            print(f"    {k:>7,} / {n_target:,}  ({100*k/n_target:.0f}%)")

        t_ref   = times[i]
        row     = {}

        # Spatial neighbours → keep only M≥4.0, exclude self
        nbr_all = all_neighbors[k]
        nbr_idx = nbr_all[m4_mask[nbr_all]]
        nbr_idx = nbr_idx[nbr_idx != i]

        # ── Elapsed since last M≥4.0 event ───────────────────────────────
        if len(nbr_idx) > 0:
            past_idx = nbr_idx[times[nbr_idx] < t_ref]
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
            row[f"count_{w}d"] = n
            row[f"rate_{w}d"]  = n / w

            if n > 0:
                m = mags[win_idx]
                d = depths[win_idx]
                d_valid = d[~np.isnan(d)]

                row[f"mag_max_{w}d"]    = float(m.max())
                row[f"mag_mean_{w}d"]   = float(m.mean())
                row[f"mag_std_{w}d"]    = float(m.std()) if n > 1 else 0.0
                row[f"energy_{w}d"]     = seismic_energy(m)
                row[f"moment_{w}d"]     = seismic_moment(m)
                row[f"b_value_{w}d"]    = b_value_mle(m)
                row[f"depth_mean_{w}d"] = float(d_valid.mean()) if len(d_valid) > 0 else np.nan
                row[f"depth_std_{w}d"]  = float(d_valid.std())  if len(d_valid) > 1 else 0.0
            else:
                for feat in ["mag_max", "mag_mean", "mag_std",
                             "b_value", "depth_mean", "depth_std"]:
                    row[f"{feat}_{w}d"] = np.nan
                row[f"energy_{w}d"] = 0.0
                row[f"moment_{w}d"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows, index=df_all.index[target_indices])


def compute_m2_window_features(
    df_all: pd.DataFrame,
    target_indices: np.ndarray,
    all_neighbors: list,
) -> pd.DataFrame:
    """
    M2-enriched windows (feature set C).

    Neighbours are filtered to rows sourced from catalog_m2_m4 (is_m2_ctx=True).
    Only windows 7d, 30d, 90d are computed.
    Features produced: count_Nd_m2, b_value_Nd_m2, energy_Nd_m2.
    All features are NaN for target events whose datetime < 2000-01-01,
    because catalog_m2_m4 only covers 2000+.
    """
    WINDOWS    = [7, 30, 90]
    CUTOFF     = np.datetime64("2000-01-01")

    times      = df_all["datetime"].values
    mags       = df_all["magnitude"].values
    m2_mask    = df_all["is_m2_ctx"].values       # boolean index over df_all rows

    rows     = []
    n_target = len(target_indices)

    for k, i in enumerate(target_indices):
        t_ref = times[i]
        row   = {}

        # Target events before the catalog coverage → all NaN
        if t_ref < CUTOFF:
            for w in WINDOWS:
                row[f"count_{w}d_m2"]   = np.nan
                row[f"b_value_{w}d_m2"] = np.nan
                row[f"energy_{w}d_m2"]  = np.nan
            rows.append(row)
            continue

        # Spatial neighbours → keep only catalog_m2_m4 rows, exclude self
        nbr_all = all_neighbors[k]
        nbr_idx = nbr_all[m2_mask[nbr_all]]
        nbr_idx = nbr_idx[nbr_idx != i]

        for w in WINDOWS:
            horizon = np.timedelta64(w, "D")
            if len(nbr_idx) > 0:
                win_mask = (times[nbr_idx] < t_ref) & \
                           (times[nbr_idx] >= t_ref - horizon)
                win_idx  = nbr_idx[win_mask]
            else:
                win_idx = np.array([], dtype=int)

            n = len(win_idx)
            row[f"count_{w}d_m2"] = n

            if n > 0:
                m = mags[win_idx]
                row[f"b_value_{w}d_m2"] = b_value_mle(m)
                row[f"energy_{w}d_m2"]  = seismic_energy(m)
            else:
                row[f"b_value_{w}d_m2"] = np.nan
                row[f"energy_{w}d_m2"]  = 0.0

        rows.append(row)

    return pd.DataFrame(rows, index=df_all.index[target_indices])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    catalog_path    = Path("data/database_updated.csv")
    context_path    = Path("data/raw/catalog_m2_m4.csv")
    boundaries_path = Path("data/external/PB2002_boundaries.json")
    output_dir      = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load catalogs
    print("── 1. Loading catalogs ─────────────────────────────────────────")
    df_main = load_catalog(catalog_path)
    df_main["is_target"] = True
    print(f"   Target  (database_updated): {len(df_main):,} events  |  "
          f"{df_main['datetime'].min().date()} → {df_main['datetime'].max().date()}")

    df_ctx = load_catalog(context_path)
    df_ctx["is_target"] = False
    print(f"   Context (catalog_m2_m4):    {len(df_ctx):,} events  |  "
          f"{df_ctx['datetime'].min().date()} → {df_ctx['datetime'].max().date()}")

    # Merge: target events take precedence when deduplicating on (datetime, lat, lon)
    # is_m2_ctx marks rows that come from catalog_m2_m4 (even if also in target)
    df_ctx["is_m2_ctx"] = True
    df_main["is_m2_ctx"] = False

    df_all = pd.concat([df_main, df_ctx], ignore_index=True)
    before = len(df_all)
    df_all = df_all.sort_values("datetime", kind="stable")
    # Drop exact duplicates: target version (is_target=True) comes first → kept
    df_all = df_all.drop_duplicates(
        subset=["datetime", "latitude", "longitude"], keep="first"
    ).reset_index(drop=True)
    print(f"\n   Combined: {len(df_all):,} events after dedup "
          f"(removed {before - len(df_all):,} duplicates)")
    n_target_in_all = df_all["is_target"].sum()
    n_m2_in_all     = df_all["is_m2_ctx"].sum()
    print(f"   Target events  (database_updated): {n_target_in_all:,}")
    print(f"   M2-ctx events  (catalog_m2_m4):    {n_m2_in_all:,}")

    # Indices into df_all of target events (output rows)
    target_indices = np.where(df_all["is_target"].values)[0]

    # 2. Reference-event features (target events only)
    print("\n── 2. Reference-event features ─────────────────────────────────")
    df_target = df_all.iloc[target_indices]

    mag_type_map = {t: i for i, t in
                    enumerate(df_target["mag_type"].fillna("unknown").unique())}
    # Encode on df_all so index alignment is preserved
    df_all["mag_type_enc"] = df_all["mag_type"].fillna("unknown").map(mag_type_map)

    ref_cols = ["latitude", "longitude", "depth", "magnitude",
                "mag_type_enc", "gap", "dmin", "rms",
                "h_error", "depth_error", "mag_error"]
    ref_cols = [c for c in ref_cols if c in df_all.columns]
    ref_features = df_all.loc[df_all.index[target_indices], ref_cols].copy()
    print(f"   {len(ref_cols)} reference features: {ref_cols}")

    # 3. Build BallTree on the COMBINED catalog (spatial index)
    print("\n── 3. Building BallTree spatial index (combined catalog) ────────")
    lat_rad_all  = np.radians(df_all["latitude"].values)
    lon_rad_all  = np.radians(df_all["longitude"].values)
    coords_all   = np.column_stack([lat_rad_all, lon_rad_all])
    tree         = BallTree(coords_all, metric="haversine")

    # Query neighbours for TARGET event positions only
    coords_target = coords_all[target_indices]
    radius_rad    = RADIUS_KM / EARTH_R_KM
    print(f"   Querying {len(target_indices):,} target positions "
          f"within {RADIUS_KM} km of {len(df_all):,} combined events …")
    all_neighbors = tree.query_radius(coords_target, r=radius_rad)
    print(f"   Done.  Avg neighbours per target event: "
          f"{np.mean([len(n) for n in all_neighbors]):.1f}")

    # 4a. Coherent window features — M≥4.0 events only
    print("\n── 4a. Coherent window features (M≥4.0) ────────────────────────")
    print(f"   Windows: {WINDOWS_DAYS} days  |  Radius: {RADIUS_KM} km")
    coherent_df = compute_coherent_window_features(
        df_all, target_indices, all_neighbors, WINDOWS_DAYS
    )
    print(f"   Generated {coherent_df.shape[1]} coherent window features.")

    # 4b. M2-enriched window features — catalog_m2_m4 events only
    print("\n── 4b. M2-enriched window features (catalog_m2_m4, post-1999) ──")
    m2_df = compute_m2_window_features(df_all, target_indices, all_neighbors)
    print(f"   Generated {m2_df.shape[1]} M2-enriched features.")
    n_nan = m2_df["count_7d_m2"].isna().sum()
    print(f"   NaN rows (pre-2000 target events): {n_nan:,}")

    # 5. Plate boundary distance (target events only)
    print("\n── 5. Plate boundary distance ──────────────────────────────────")
    target_df_for_dist = df_all.iloc[target_indices].reset_index(drop=True)
    if boundaries_path.exists():
        plate_dist_vals = plate_boundary_distances(target_df_for_dist, boundaries_path)
        plate_dist = pd.Series(
            plate_dist_vals.values,
            index=df_all.index[target_indices],
            name="dist_to_plate_boundary_km",
        )
    else:
        print(f"   WARNING: {boundaries_path} not found — feature set to NaN")
        plate_dist = pd.Series(
            np.nan,
            index=df_all.index[target_indices],
            name="dist_to_plate_boundary_km",
        )

    # 6. Assemble
    print("\n── 6. Assembling feature matrix ────────────────────────────────")
    features = pd.concat([ref_features, coherent_df, m2_df, plate_dist], axis=1)

    # Keep datetime + coords for joining with labels.csv
    dt_vals  = df_all["datetime"].iloc[target_indices].dt.strftime("%Y-%m-%d %H:%M:%S")
    lat_vals = df_all["latitude"].iloc[target_indices]
    lon_vals = df_all["longitude"].iloc[target_indices]

    features.insert(0, "datetime", dt_vals.values)
    features.insert(1, "ref_lat",  lat_vals.values)
    features.insert(2, "ref_lon",  lon_vals.values)

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
