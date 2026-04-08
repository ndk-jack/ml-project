"""
catalog_manager.py — Rolling in-memory catalog of recent seismic events.

Responsibilities:
  - Load the historical pre-2010 catalog at startup (for background_rate_yr)
  - Load and cache external data (GEM faults, WSM2016) at startup
  - Maintain a rolling 90-day window of recent M≥4 events (refreshed from USGS)
  - Expose a thread-safe interface for feature_engine.py to query neighbors

The historical catalog is never modified at runtime. The rolling catalog is
updated every POLL_INTERVAL_MINUTES by the background scheduler in main.py.
"""

import threading
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)

# ── Paths (relative to project root) ─────────────────────────────────────────
DATA_DIR         = Path("data")
RAW_DIR          = DATA_DIR / "raw"
EXTERNAL_DIR     = DATA_DIR / "external"

PRIMARY_CATALOG  = RAW_DIR / "database_updated.csv"
GEM_FAULTS       = EXTERNAL_DIR / "gem_active_faults.geojson"
WSM_CSV          = EXTERNAL_DIR / "wsm2016.csv"
PB_JSON          = EXTERNAL_DIR / "PB2002_boundaries.json"

# ── Constants ─────────────────────────────────────────────────────────────────
EARTH_R          = 6371.0           # km
RADIUS_KM        = 200.0
ROLLING_DAYS     = 92               # keep 92 days to cover the 90d window
HISTORICAL_YEAR  = 2010             # pre-2010 = reference for background rate
MIN_MAG_ROLLING  = 2.0              # keep M≥2 in rolling catalog
MIN_MAG_HISTORY  = 3.0              # M≥3 for background rate computation
USGS_API         = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
    "?format=geojson&minmagnitude=2.0"
    "&orderby=time&limit=1000"
)
SLIP_TYPE_MAP = {
    "Normal": 1, "Reverse": 2, "Thrust": 2,
    "Strike-Slip": 3, "Sinistral": 3, "Dextral": 3,
    "Oblique": 4, "Blind Thrust": 2,
}
REGIME_MAP  = {"NF": 0, "TF": 1, "SS": 2, "NS": 3, "TS": 3, "U": 4}
QUALITY_MAP = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}


class CatalogManager:
    """Thread-safe catalog manager. Initialised once at app startup."""

    def __init__(self):
        self._lock           = threading.RLock()

        # Historical catalog (pre-2010, M≥3) — used for background_rate_yr
        self.hist_coords_rad: np.ndarray | None = None   # shape (N, 2)
        self.hist_tree:       BallTree | None    = None
        self.hist_span_yr:    float              = 110.0

        # Rolling catalog (last 92 days, M≥2)
        self.rolling_df:      pd.DataFrame       = pd.DataFrame()
        self.rolling_tree:    BallTree | None    = None

        # External data
        self.gem_fault_union = None        # shapely MultiLineString (EPSG:3857)
        self.fault_lats_m    = None        # fault centroid Y in EPSG:3857
        self.fault_lons_m    = None        # fault centroid X in EPSG:3857
        self.fault_types     = None        # encoded slip types
        self.fault_kt        = None        # KDTree for slip type lookup

        self.wsm_df:          pd.DataFrame = pd.DataFrame()
        self.wsm_tree:        BallTree | None = None

        self.pb_coords_rad:   np.ndarray | None = None
        self.pb_tree:         BallTree | None   = None

        self._initialized = False

    # ── Public init ───────────────────────────────────────────────────────────

    def initialize(self):
        """Load all static data. Call once at app startup."""
        logger.info("CatalogManager: initializing…")
        self._load_historical_catalog()
        self._load_gem_faults()
        self._load_wsm()
        self._load_plate_boundaries()
        self._initialized = True
        logger.info("CatalogManager: ready.")

    def refresh_rolling(self):
        """
        Fetch the last ROLLING_DAYS of M≥2 events from USGS and rebuild
        the rolling BallTree. Called by the background scheduler.
        """
        start = (datetime.now(timezone.utc) - timedelta(days=ROLLING_DAYS)
                 ).strftime("%Y-%m-%dT%H:%M:%S")
        url = USGS_API + f"&starttime={start}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            if not features:
                logger.warning("USGS returned 0 events.")
                return

            rows = []
            for f in features:
                p = f["properties"]
                c = f["geometry"]["coordinates"]
                rows.append({
                    "event_id": f["id"],
                    "datetime": pd.to_datetime(p["time"], unit="ms", utc=True),
                    "latitude":  c[1],
                    "longitude": c[0],
                    "depth":     c[2] if c[2] is not None else np.nan,
                    "magnitude": p["mag"] if p["mag"] is not None else np.nan,
                })

            df = pd.DataFrame(rows).dropna(subset=["magnitude", "latitude", "longitude"])
            df = df[df["magnitude"] >= MIN_MAG_ROLLING].copy()
            df.sort_values("datetime", inplace=True)

            coords_rad = np.radians(df[["latitude", "longitude"]].values)
            tree = BallTree(coords_rad, metric="haversine")

            with self._lock:
                self.rolling_df   = df.reset_index(drop=True)
                self.rolling_tree = tree

            logger.info(f"Rolling catalog refreshed: {len(df):,} events "
                        f"({df['datetime'].min().date()} → {df['datetime'].max().date()})")

        except Exception as e:
            logger.error(f"USGS refresh failed: {e}")

    # ── Spatial queries ───────────────────────────────────────────────────────

    def query_rolling(self, lat: float, lon: float, before_dt: datetime,
                      days: int, min_mag: float = 2.0) -> pd.DataFrame:
        """
        Return events from the rolling catalog within RADIUS_KM and `days`
        before `before_dt`, with magnitude ≥ min_mag.
        """
        with self._lock:
            if self.rolling_tree is None or self.rolling_df.empty:
                return pd.DataFrame()
            df   = self.rolling_df
            tree = self.rolling_tree

        q_rad    = np.radians([[lat, lon]])
        r_rad    = RADIUS_KM / EARTH_R
        idx_list = tree.query_radius(q_rad, r=r_rad)[0]

        if len(idx_list) == 0:
            return pd.DataFrame()

        sub = df.iloc[idx_list].copy()
        cutoff = before_dt - timedelta(days=days)
        sub = sub[(sub["datetime"] >= cutoff) &
                  (sub["datetime"] < before_dt) &
                  (sub["magnitude"] >= min_mag)]
        return sub

    def background_count(self, lat: float, lon: float) -> float:
        """Return pre-2010 M≥3 event count within RADIUS_KM."""
        if self.hist_tree is None:
            return 0.0
        q_rad = np.radians([[lat, lon]])
        r_rad = RADIUS_KM / EARTH_R
        count = self.hist_tree.query_radius(q_rad, r=r_rad, count_only=True)[0]
        return float(count)

    def nearest_fault(self, lat: float, lon: float):
        """Return (dist_km, slip_type_enc) for nearest GEM fault."""
        if self.gem_fault_union is None:
            return np.nan, 0
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            pt_m = gpd.GeoSeries(
                [Point(lon, lat)], crs="EPSG:4326"
            ).to_crs("EPSG:3857").iloc[0]
            dist_km = pt_m.distance(self.gem_fault_union) / 1000.0

            # Nearest fault segment for slip type
            ev_coord = np.array([[pt_m.y, pt_m.x]])
            _, idx = self.fault_kt.query(ev_coord, k=1)
            slip_enc = int(self.fault_types[idx.flatten()[0]])
            return float(dist_km), slip_enc
        except Exception as e:
            logger.warning(f"Fault lookup failed: {e}")
            return np.nan, 0

    def nearest_wsm(self, lat: float, lon: float):
        """Return (regime_enc, shmax_sin, shmax_cos, wsm_dist_km)."""
        if self.wsm_tree is None:
            return 4, 0.0, 0.0, np.nan
        q_rad = np.radians([[lat, lon]])
        dist_arr, idx_arr = self.wsm_tree.query(q_rad, k=1)
        dist_km = float(dist_arr[0, 0] * EARTH_R)
        if dist_km > 500:
            return 4, 0.0, 0.0, dist_km
        row = self.wsm_df.iloc[idx_arr[0, 0]]
        regime = REGIME_MAP.get(str(row.get("REGIME", "U")).strip(), 4)
        azi    = float(row.get("AZI", 0))
        return (regime,
                float(np.sin(np.radians(azi))),
                float(np.cos(np.radians(azi))),
                dist_km)

    def nearest_plate_boundary(self, lat: float, lon: float) -> float:
        """Return distance in km to nearest plate boundary."""
        if self.pb_tree is None:
            return np.nan
        q_rad    = np.radians([[lat, lon]])
        dist_arr, _ = self.pb_tree.query(q_rad, k=1)
        return float(dist_arr[0, 0] * EARTH_R)

    # ── Private loaders ───────────────────────────────────────────────────────

    def _load_historical_catalog(self):
        if not PRIMARY_CATALOG.exists():
            logger.warning(f"Historical catalog not found: {PRIMARY_CATALOG}")
            return

        df = pd.read_csv(PRIMARY_CATALOG, low_memory=False)
        df.columns = df.columns.str.strip()

        # Normalise column names (Kaggle vs API format)
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("lat", "latitude"):       rename[c] = "latitude"
            elif cl in ("lon", "longitude"):     rename[c] = "longitude"
            elif cl in ("mag", "magnitude"):     rename[c] = "magnitude"
            elif cl in ("depth",):               rename[c] = "depth"
            elif cl in ("time", "date", "datetime"): rename[c] = "datetime"
        df.rename(columns=rename, inplace=True)

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime", "latitude", "longitude", "magnitude"])
        df = df[
            (df["datetime"].dt.year < HISTORICAL_YEAR) &
            (df["magnitude"] >= MIN_MAG_HISTORY)
        ].copy()

        if df.empty:
            logger.warning("Historical catalog empty after filtering.")
            return

        t_min, t_max  = df["datetime"].min(), df["datetime"].max()
        self.hist_span_yr = (t_max - t_min).days / 365.25

        coords_rad = np.radians(df[["latitude", "longitude"]].values)
        self.hist_coords_rad = coords_rad
        self.hist_tree       = BallTree(coords_rad, metric="haversine")
        logger.info(f"Historical catalog: {len(df):,} M≥{MIN_MAG_HISTORY} events "
                    f"({t_min.year}–{t_max.year}, {self.hist_span_yr:.1f} yr)")

    def _load_gem_faults(self):
        if not GEM_FAULTS.exists():
            logger.warning(f"GEM faults not found: {GEM_FAULTS}")
            return
        try:
            import geopandas as gpd
            from shapely.ops import unary_union
            from sklearn.neighbors import KDTree

            faults   = gpd.read_file(GEM_FAULTS).to_crs("EPSG:4326")
            faults_m = faults.to_crs("EPSG:3857")
            self.gem_fault_union = unary_union(faults_m.geometry)

            # For slip type lookup via KDTree
            self.fault_lats_m = faults_m.geometry.centroid.y.values
            self.fault_lons_m = faults_m.geometry.centroid.x.values
            fault_coords = np.column_stack([self.fault_lats_m, self.fault_lons_m])
            self.fault_kt = KDTree(fault_coords)

            slip_col = next((c for c in ["slip_type", "slip_type_1", "rake", "type"]
                             if c in faults.columns), None)
            if slip_col:
                self.fault_types = np.array([
                    SLIP_TYPE_MAP.get(str(t).strip().title(), 0)
                    for t in faults[slip_col].values
                ])
            else:
                self.fault_types = np.zeros(len(faults), dtype=int)

            logger.info(f"GEM faults loaded: {len(faults):,} segments")
        except Exception as e:
            logger.error(f"GEM faults load failed: {e}")

    def _load_wsm(self):
        if not WSM_CSV.exists():
            logger.warning(f"WSM not found: {WSM_CSV}")
            return
        try:
            wsm = pd.read_csv(WSM_CSV, encoding="latin-1", low_memory=False)
            wsm.columns = wsm.columns.str.strip()
            wsm = wsm[wsm["QUALITY"].isin(["A", "B", "C"])].dropna(
                subset=["LAT", "LON", "AZI"]
            ).copy()
            self.wsm_df   = wsm.reset_index(drop=True)
            wsm_rad       = np.radians(wsm[["LAT", "LON"]].values)
            self.wsm_tree = BallTree(wsm_rad, metric="haversine")
            logger.info(f"WSM loaded: {len(wsm):,} records (A/B/C quality)")
        except Exception as e:
            logger.error(f"WSM load failed: {e}")

    def _load_plate_boundaries(self):
        if not PB_JSON.exists():
            logger.warning(f"Plate boundaries not found: {PB_JSON}")
            return
        try:
            import json
            with open(PB_JSON) as f:
                pb = json.load(f)
            pts = []
            for feature in pb.get("features", []):
                geom = feature.get("geometry", {})
                if geom.get("type") == "LineString":
                    pts.extend(geom["coordinates"])
                elif geom.get("type") == "MultiLineString":
                    for line in geom["coordinates"]:
                        pts.extend(line)
            coords = np.array([[p[1], p[0]] for p in pts])  # lat, lon
            coords_rad      = np.radians(coords)
            self.pb_coords_rad = coords_rad
            self.pb_tree       = BallTree(coords_rad, metric="haversine")
            logger.info(f"Plate boundaries loaded: {len(coords):,} points")
        except Exception as e:
            logger.error(f"Plate boundaries load failed: {e}")


# Singleton — imported by feature_engine and main
catalog = CatalogManager()
