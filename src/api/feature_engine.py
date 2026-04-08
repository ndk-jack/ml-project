"""
feature_engine.py — Compute all model features for a single seismic event.

Mirrors the logic of features.py and add_ratio_features.py but operates on
one event at a time using the rolling catalog from catalog_manager.

Feature groups:
  1. Coherent window features (M≥4, 1d/7d/30d/90d) — 40 features (10 per window)
  2. M≥2 enriched window features (post-1999)         —  9 features
  3. Acceleration / anomaly features                   —  3 features
  4. External features (GEM, WSM, background rate)     —  8 features
  5. Base event features                               —  4 features

NOTE: FEATURE_COLS is the full set we compute. scorer.py uses model.feature_name()
to select the exact subset expected by each trained model, filling NaN for
any feature not in this list.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from catalog_manager import catalog

logger = logging.getLogger(__name__)

EPS = 1e-9

# ── Feature column order (must match training dataset_v3, wsm_quality_enc dropped)
FEATURE_COLS = [
    # Base
    "magnitude", "depth", "elapsed_since_last_s", "dist_to_plate_boundary_km",
    # Window features — 1d  (10 per window: +mag_std vs original 9)
    "count_1d", "rate_1d", "energy_1d", "moment_1d",
    "b_value_1d", "mag_mean_1d", "mag_max_1d", "mag_std_1d", "depth_mean_1d", "depth_std_1d",
    # Window features — 7d
    "count_7d", "rate_7d", "energy_7d", "moment_7d",
    "b_value_7d", "mag_mean_7d", "mag_max_7d", "mag_std_7d", "depth_mean_7d", "depth_std_7d",
    # Window features — 30d
    "count_30d", "rate_30d", "energy_30d", "moment_30d",
    "b_value_30d", "mag_mean_30d", "mag_max_30d", "mag_std_30d", "depth_mean_30d", "depth_std_30d",
    # Window features — 90d
    "count_90d", "rate_90d", "energy_90d", "moment_90d",
    "b_value_90d", "mag_mean_90d", "mag_max_90d", "mag_std_90d", "depth_mean_90d", "depth_std_90d",
    # M≥2 enriched (post-1999)
    "count_7d_m2", "energy_7d_m2", "b_value_7d_m2",
    "count_30d_m2", "energy_30d_m2", "b_value_30d_m2",
    "count_90d_m2", "energy_90d_m2", "b_value_90d_m2",
    # Acceleration
    "accel_count", "accel_energy", "mag_excess",
    # External
    "dist_to_nearest_fault_km", "fault_slip_type_enc",
    "stress_regime_enc", "shmax_sin", "shmax_cos", "wsm_dist_km",
    "background_rate_yr", "normalized_rate_30d",
]


# ── Seismological helpers ─────────────────────────────────────────────────────

def _seismic_energy(magnitudes: np.ndarray) -> float:
    """Seismic energy proxy: sum(10^(1.5*M))."""
    if len(magnitudes) == 0:
        return 0.0
    return float(np.sum(10 ** (1.5 * magnitudes)))


def _seismic_moment(magnitudes: np.ndarray) -> float:
    """Seismic moment proxy: sum(10^(1.5*M + 9.1))."""
    if len(magnitudes) == 0:
        return 0.0
    return float(np.sum(10 ** (1.5 * magnitudes + 9.1)))


def _b_value_mle(magnitudes: np.ndarray, mc: float = 2.0) -> Optional[float]:
    """Maximum likelihood b-value estimate. Returns None if < 5 events."""
    mags = magnitudes[magnitudes >= mc]
    if len(mags) < 5:
        return None
    mean_m = mags.mean()
    if mean_m <= mc:
        return None
    return float(1.0 / ((mean_m - mc) * np.log(10)))


def _window_features(neighbors: pd.DataFrame, window_days: int,
                     min_mag: float = 2.0, suffix: str = "") -> dict:
    """
    Compute count/rate/energy/moment/b_value/mag_mean/mag_max/depth stats
    for the given neighbor slice. suffix e.g. "_7d" or "_7d_m2".
    """
    df = neighbors[neighbors["magnitude"] >= min_mag]
    n  = len(df)
    mags   = df["magnitude"].values if n > 0 else np.array([])
    depths = df["depth"].values     if n > 0 else np.array([])

    rate    = n / max(window_days, 1)
    energy  = _seismic_energy(mags)
    moment  = _seismic_moment(mags)
    bval    = _b_value_mle(mags)

    return {
        f"count{suffix}":      n,
        f"rate{suffix}":       rate,
        f"energy{suffix}":     energy,
        f"moment{suffix}":     moment,
        f"b_value{suffix}":    bval if bval is not None else np.nan,
        f"mag_mean{suffix}":   float(mags.mean())   if n > 0 else np.nan,
        f"mag_max{suffix}":    float(mags.max())    if n > 0 else np.nan,
        f"mag_std{suffix}":    float(mags.std())    if n > 1 else 0.0,
        f"depth_mean{suffix}": float(depths.mean()) if n > 0 else np.nan,
        f"depth_std{suffix}":  float(depths.std())  if n > 1 else 0.0,
    }


# ── Main feature computation ──────────────────────────────────────────────────

def compute_features(
    lat: float,
    lon: float,
    depth: float,
    magnitude: float,
    event_dt: datetime,
    event_id: Optional[str] = None,
) -> dict:
    """
    Compute all 60 model features for a single event.
    Returns a dict with all FEATURE_COLS keys.
    All NaN values will be imputed by scorer.py using training medians.
    """
    feats: dict = {}

    # ── 1. Base features ──────────────────────────────────────────────────────
    feats["magnitude"] = magnitude
    feats["depth"]     = depth if depth is not None else np.nan

    # Elapsed since last M≥4 event in 200km
    neighbors_90d_m4 = catalog.query_rolling(lat, lon, event_dt,
                                             days=90, min_mag=4.0)
    if not neighbors_90d_m4.empty:
        last_dt = neighbors_90d_m4["datetime"].max()
        feats["elapsed_since_last_s"] = (event_dt - last_dt).total_seconds()
    else:
        feats["elapsed_since_last_s"] = np.nan

    feats["dist_to_plate_boundary_km"] = catalog.nearest_plate_boundary(lat, lon)

    # ── 2. Coherent window features (M≥4) ─────────────────────────────────────
    for days, label in [(1, "_1d"), (7, "_7d"), (30, "_30d"), (90, "_90d")]:
        neighbors = catalog.query_rolling(lat, lon, event_dt,
                                          days=days, min_mag=4.0)
        wf = _window_features(neighbors, days, min_mag=4.0, suffix=label)
        feats.update(wf)

    # ── 3. M≥2 enriched features (only for post-1999 events) ──────────────────
    is_post_1999 = event_dt.year >= 2000
    for days, label in [(7, "_7d_m2"), (30, "_30d_m2"), (90, "_90d_m2")]:
        if is_post_1999:
            neighbors = catalog.query_rolling(lat, lon, event_dt,
                                              days=days, min_mag=2.0)
            wf = _window_features(neighbors, days, min_mag=2.0, suffix=label)
            # Only keep count, energy, b_value for M2
            feats[f"count{label}"]   = wf[f"count{label}"]
            feats[f"energy{label}"]  = wf[f"energy{label}"]
            feats[f"b_value{label}"] = wf[f"b_value{label}"]
        else:
            feats[f"count{label}"]   = np.nan
            feats[f"energy{label}"]  = np.nan
            feats[f"b_value{label}"] = np.nan

    # ── 4. Acceleration features ───────────────────────────────────────────────
    c7, c90 = feats.get("count_7d", 0), feats.get("count_90d", 0)
    e7, e90 = feats.get("energy_7d", 0), feats.get("energy_90d", 0)
    bg_count_7d  = (c90 / 90.0) * 7.0
    bg_energy_7d = (e90 / 90.0) * 7.0
    feats["accel_count"]  = min((c7 / (bg_count_7d + EPS)),  500.0)
    feats["accel_energy"] = min((e7 / (bg_energy_7d + EPS)), 500.0)

    mag_mean_90d = feats.get("mag_mean_90d", np.nan)
    feats["mag_excess"] = (
        magnitude - mag_mean_90d if not np.isnan(mag_mean_90d) else 0.0
    )

    # ── 5. External features ──────────────────────────────────────────────────
    dist_fault, slip_type = catalog.nearest_fault(lat, lon)
    feats["dist_to_nearest_fault_km"] = dist_fault
    feats["fault_slip_type_enc"]      = slip_type

    regime, shmax_sin, shmax_cos, wsm_dist = catalog.nearest_wsm(lat, lon)
    feats["stress_regime_enc"] = regime
    feats["shmax_sin"]         = shmax_sin
    feats["shmax_cos"]         = shmax_cos
    feats["wsm_dist_km"]       = wsm_dist

    bg_count = catalog.background_count(lat, lon)
    bg_rate  = bg_count / max(catalog.hist_span_yr, 1.0)
    feats["background_rate_yr"] = bg_rate

    expected_30d = bg_rate * (30.0 / 365.25) + EPS
    feats["normalized_rate_30d"] = min(
        feats.get("count_30d", 0) / expected_30d, 200.0
    )

    return feats


def features_to_series(feats: dict) -> pd.Series:
    """Return a pd.Series with exactly FEATURE_COLS, NaN for missing."""
    return pd.Series({col: feats.get(col, np.nan) for col in FEATURE_COLS})
