"""
catalog_manager.py — Rolling in-memory catalog of recent seismic events.

Responsibilities:
  - Load the historical pre-2010 catalog at startup (for background_rate_yr)
  - Load and cache external data (GEM faults, WSM2016) at startup
  - Maintain a rolling 92-day window of recent M≥2 events (refreshed from USGS)
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

PROJECT_ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR         = PROJECT_ROOT / "data"
RAW_DIR          = DATA_DIR / "raw"
API_DIR          = DATA_DIR / "api"
EXTERNAL_DIR     = DATA_DIR / "external"

PRIMARY_CATALOG  = API_DIR / "historical_catalog_pre2010_m3.csv.gz"
FALLBACK_CATALOGS = (
    DATA_DIR / "database_updated.csv",
    RAW_DIR / "database_updated.csv",
)

GEM_FAULTS       = EXTERNAL_DIR / "gem_active_faults.geojson"
WSM_CSV          = EXTERNAL_DIR / "wsm2016.csv"
PB_JSON          = EXTERNAL_DIR / "PB2002_boundaries.json"

# ── Constants ─────────────────────────────────────────────────────────────────

EARTH_R          = 6371.0
RADIUS_KM        = 200.0
ROLLING_DAYS        = 92
ROLLING_CHUNK_HOURS = 24
HISTORICAL_YEAR     = 2010
MIN_MAG_ROLLING     = 2.0
MIN_MAG_HISTORY     = 3.0
USGS_TIMEOUT        = 30
USGS_API            = "https://earthquake.usgs.gov/fdsnws/event/1/query"
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
        self.hist_coords_rad: np.ndarray | None = None
        self.hist_tree:       BallTree | None   = None
        self.hist_span_yr:    float             = 110.0
        self.hist_df:         pd.DataFrame      = pd.DataFrame()
        self.hist_source:     str | None        = None
        self.hist_events:     int               = 0
        self.hist_loaded:     bool              = False

        # Rolling catalog (last 92 days, M≥2)
        self.rolling_df:              pd.DataFrame     = pd.DataFrame()
        self.rolling_tree:            BallTree | None  = None
        self.rolling_events:          int              = 0
        self.rolling_window_start:    datetime | None  = None
        self.rolling_window_end:      datetime | None  = None
        self.rolling_last_refresh_at: datetime | None  = None

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

    def _fetch_usgs_chunk(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        params = {
            "format": "geojson",
            "minmagnitude": str(MIN_MAG_ROLLING),
            "orderby": "time",
            "starttime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "limit": 20000,
        }

        resp = requests.get(USGS_API, params=params, timeout=USGS_TIMEOUT)
        resp.raise_for_status()

        features = resp.json().get("features", [])
        if not features:
            return pd.DataFrame(columns=["event_id", "datetime", "latitude", "longitude", "depth", "magnitude"])

        rows = []
        for f in features:
            props = f.get("properties", {})
            geom = f.get("geometry", {})
            coords = geom.get("coordinates", [None, None, None])

            rows.append({
                "event_id": f.get("id"),
                "datetime_ms": props.get("time"),
                "latitude": coords[1] if len(coords) > 1 else None,
                "longitude": coords[0] if len(coords) > 0 else None,
                "depth": coords[2] if len(coords) > 2 else np.nan,
                "magnitude": props.get("mag"),
            })

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime_ms"], unit="ms", utc=True, errors="coerce")
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
        df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")

        df = df.drop(columns=["datetime_ms"])
        df = df.dropna(subset=["event_id", "datetime", "latitude", "longitude", "magnitude"])
        df = df[df["magnitude"] >= MIN_MAG_ROLLING].copy()

        return df

    def refresh_rolling(self):
        """
        Fetch the last ROLLING_DAYS of M≥2 events from USGS in chunks and rebuild
        the rolling BallTree. Called by the background scheduler.
        """
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=ROLLING_DAYS)

        chunk_start = start_dt
        chunks = []
        chunk_count = 0
        failed_chunks = 0

        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(hours=ROLLING_CHUNK_HOURS), end_dt)
            chunk_count += 1

            try:
                df_chunk = self._fetch_usgs_chunk(chunk_start, chunk_end)
                if not df_chunk.empty:
                    chunks.append(df_chunk)
            except Exception as e:
                failed_chunks += 1
                logger.error(
                    f"USGS chunk refresh failed for {chunk_start} → {chunk_end}: {e}"
                )

            chunk_start = chunk_end

        if not chunks:
            logger.warning("USGS returned 0 events across all rolling chunks.")
            return

        df = pd.concat(chunks, ignore_index=True)
        df = df.drop_duplicates(subset=["event_id"], keep="last").copy()
        df = df[
            (df["datetime"] >= start_dt) &
            (df["datetime"] <= end_dt) &
            (df["magnitude"] >= MIN_MAG_ROLLING)
        ].copy()
        df.sort_values("datetime", inplace=True)

        if df.empty:
            logger.warning("Rolling catalog empty after filtering and deduplication.")
            return

        coords_rad = np.radians(df[["latitude", "longitude"]].values)
        tree = BallTree(coords_rad, metric="haversine")

        with self._lock:
            self.rolling_df = df.reset_index(drop=True)
            self.rolling_tree = tree
            self.rolling_events = len(df)
            self.rolling_window_start = df["datetime"].min()
            self.rolling_window_end = df["datetime"].max()
            self.rolling_last_refresh_at = datetime.now(timezone.utc)

        logger.info(
            f"Rolling catalog refreshed: {len(df):,} events "
            f"({df['datetime'].min()} → {df['datetime'].max()}), "
            f"chunks={chunk_count}, failed_chunks={failed_chunks}"
        )

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
        source_used: Path | None = None

        # 1) Preferred path: small API artifact
        if PRIMARY_CATALOG.exists():
            logger.info(f"Loading historical catalog from API artifact: {PRIMARY_CATALOG}")
            df = pd.read_csv(PRIMARY_CATALOG, compression="infer")
            df.columns = df.columns.str.strip()
            df.rename(columns={c: c.strip().lower() for c in df.columns}, inplace=True)

            required = {"datetime", "latitude", "longitude", "magnitude"}
            missing = required - set(df.columns)
            if missing:
                logger.error(
                    f"Historical API artifact missing required columns: {sorted(missing)}"
                )
                return

            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
            source_used = PRIMARY_CATALOG

        else:
            # 2) Fallback path(s): raw historical catalog
            fallback = next((p for p in FALLBACK_CATALOGS if p.exists()), None)
            if fallback is None:
                logger.warning(
                    "Historical catalog not found. Looked for: "
                    f"{PRIMARY_CATALOG}, {FALLBACK_CATALOGS[0]}, {FALLBACK_CATALOGS[1]}"
                )
                self.hist_loaded = False
                self.hist_source = None
                self.hist_events = 0
                self.hist_df = pd.DataFrame()
                self.hist_tree = None
                self.hist_coords_rad = None
                return

            logger.info(f"Loading historical catalog from fallback raw file: {fallback}")

            header = pd.read_csv(fallback, nrows=0, low_memory=False)
            header.columns = header.columns.str.strip()
            header_cols = set(header.columns)

            # Kaggle-style schema: Date + Time separated
            if {"Date", "Time", "Latitude", "Longitude", "Magnitude"}.issubset(header_cols):
                df = pd.read_csv(
                    fallback,
                    usecols=["Date", "Time", "Latitude", "Longitude", "Magnitude"],
                    low_memory=False,
                )
                df = df.rename(columns={
                    "Date": "date",
                    "Time": "time",
                    "Latitude": "latitude",
                    "Longitude": "longitude",
                    "Magnitude": "magnitude",
                })

                df["datetime"] = pd.to_datetime(
                    df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
                    format="%m/%d/%Y %H:%M:%S",
                    errors="coerce",
                    utc=True,
                )
                df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
                df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
                df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
                source_used = fallback

            else:
                df = pd.read_csv(fallback, low_memory=False)
                df.columns = df.columns.str.strip()

                rename = {}
                for c in df.columns:
                    cl = c.lower().strip()
                    if cl == "latitude":
                        rename[c] = "latitude"
                    elif cl == "longitude":
                        rename[c] = "longitude"
                    elif cl in ("magnitude", "mag"):
                        rename[c] = "magnitude"
                    elif cl == "date":
                        rename[c] = "date"
                    elif cl == "time":
                        rename[c] = "time"
                    elif cl == "datetime":
                        rename[c] = "datetime"

                df.rename(columns=rename, inplace=True)

                if "datetime" not in df.columns and {"date", "time"}.issubset(df.columns):
                    df["datetime"] = pd.to_datetime(
                        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
                        errors="coerce",
                        utc=True,
                    )
                elif "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
                else:
                    logger.error(
                        f"Historical fallback file has no usable datetime columns: {fallback}"
                    )
                    return

                if "latitude" not in df.columns or "longitude" not in df.columns or "magnitude" not in df.columns:
                    logger.error(
                        f"Historical fallback file missing required spatial/magnitude columns: {fallback}"
                    )
                    return

                df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
                df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
                df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
                source_used = fallback

        df = df.dropna(subset=["datetime", "latitude", "longitude", "magnitude"])
        df = df[
            (df["datetime"].dt.year < HISTORICAL_YEAR) &
            (df["magnitude"] >= MIN_MAG_HISTORY)
        ].copy()

        if df.empty:
            logger.warning("Historical catalog empty after filtering.")
            self.hist_loaded = False
            self.hist_source = str(source_used) if source_used else None
            self.hist_events = 0
            self.hist_df = pd.DataFrame()
            self.hist_tree = None
            self.hist_coords_rad = None
            return

        df.sort_values("datetime", inplace=True)

        t_min, t_max = df["datetime"].min(), df["datetime"].max()
        self.hist_span_yr = max((t_max - t_min).days / 365.25, 1.0)

        coords_rad = np.radians(df[["latitude", "longitude"]].values)

        self.hist_df = df.reset_index(drop=True)
        self.hist_coords_rad = coords_rad
        self.hist_tree = BallTree(coords_rad, metric="haversine")
        self.hist_source = str(source_used) if source_used else None
        self.hist_events = len(df)
        self.hist_loaded = True

        logger.info(
            f"Historical catalog loaded: {len(df):,} M≥{MIN_MAG_HISTORY} events "
            f"({t_min} → {t_max}, span={self.hist_span_yr:.1f}y, source={self.hist_source})"
        )

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
