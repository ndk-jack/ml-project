"""
main.py — FastAPI application for real-time earthquake sequence scoring.

Startup sequence:
  1. CatalogManager.initialize() — load historical catalog + external data
  2. CatalogManager.refresh_rolling() — fetch last 92 days from USGS
  3. Scorer.initialize() — load LightGBM models + medians
  4. APScheduler starts background polling every POLL_INTERVAL minutes

Endpoints:
  GET  /health               — service status
  GET  /score/latest?n=10    — score the N most recent M≥4 USGS events
  POST /score                — score a single event (manual input)
  GET  /events               — list recently scored events (in-memory cache)

Run:
  cd ~/ml-project
  python3.9 -m uvicorn src.api.main:app --reload --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow imports from src/api when running as a module
sys.path.insert(0, str(__file__).replace("/main.py", ""))

from catalog_manager import catalog
from feature_engine import compute_features, features_to_series
from scorer import scorer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

# ── Config ────────────────────────────────────────────────────────────────────
POLL_INTERVAL_MINUTES = 5
MAX_CACHE_SIZE        = 500    # keep last N scored events in memory
USGS_LATEST_URL       = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
    "?format=geojson&minmagnitude=4.0&orderby=time"
)

# In-memory cache of recently scored events
_scored_cache: list[dict] = []
_scheduler: Optional[BackgroundScheduler] = None


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler

    logger.info("=== Earthquake Scoring API — startup ===")

    # 1. Load static data
    catalog.initialize()

    # 2. Fetch initial rolling catalog
    catalog.refresh_rolling()

    # 3. Load models
    scorer.initialize()

    # 4. Start background scheduler
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        catalog.refresh_rolling,
        "interval",
        minutes=POLL_INTERVAL_MINUTES,
        id="usgs_poll",
        max_instances=1,
    )
    _scheduler.start()
    logger.info(f"USGS polling started (every {POLL_INTERVAL_MINUTES} min)")

    logger.info("=== API ready ===")
    yield

    # Shutdown
    if _scheduler:
        _scheduler.shutdown(wait=False)
    logger.info("=== API shutdown ===")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Earthquake Sequence Forecasting API",
    description=(
        "Estimates the probability of a M≥5.0 follow-up earthquake "
        "within 200 km over 7, 30, and 365 days after a reference event."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class EventInput(BaseModel):
    latitude:  float = Field(..., ge=-90,  le=90,  description="Latitude (°)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (°)")
    depth:     float = Field(..., ge=0,    le=800, description="Depth (km)")
    magnitude: float = Field(..., ge=0,    le=10,  description="Magnitude")
    datetime:  Optional[str] = Field(
        None,
        description="ISO-8601 UTC datetime. Defaults to now if omitted.",
        examples=["2024-01-17T12:34:56Z"],
    )
    event_id:  Optional[str] = Field(None, description="Optional event identifier")


class ScoreResponse(BaseModel):
    event_id:      Optional[str]
    latitude:      float
    longitude:     float
    depth:         float
    magnitude:     float
    datetime:      str
    prob_7d:       Optional[float]
    prob_30d:      Optional[float]
    prob_365d:     Optional[float]
    risk_7d:       str
    risk_30d:      str
    risk_365d:     str
    features_used: int
    scored_at:     str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(dt_str: Optional[str]) -> datetime:
    if dt_str is None:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid datetime format: '{dt_str}'. Use ISO-8601 UTC."
        )


def _score_event(
    lat: float, lon: float, depth: float, magnitude: float,
    event_dt: datetime, event_id: Optional[str] = None,
) -> dict:
    """Compute features and score. Returns a full response dict."""
    feats_dict = compute_features(lat, lon, depth, magnitude,
                                  event_dt, event_id)
    feats_s    = features_to_series(feats_dict)
    scores     = scorer.score(feats_s)

    result = {
        "event_id":      event_id,
        "latitude":      lat,
        "longitude":     lon,
        "depth":         depth,
        "magnitude":     magnitude,
        "datetime":      event_dt.isoformat(),
        "scored_at":     datetime.now(timezone.utc).isoformat(),
        **scores,
    }

    # Add to cache
    _scored_cache.append(result)
    if len(_scored_cache) > MAX_CACHE_SIZE:
        _scored_cache.pop(0)

    return result


def _fetch_usgs_latest(n: int) -> list[dict]:
    """Fetch the N most recent M≥4 events from USGS."""
    url = USGS_LATEST_URL + f"&limit={min(n, 100)}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except Exception as e:
        raise HTTPException(status_code=503,
                            detail=f"USGS API unreachable: {e}")

    events = []
    for f in features:
        p = f["properties"]
        c = f["geometry"]["coordinates"]
        if p.get("mag") is None:
            continue
        events.append({
            "event_id":  f["id"],
            "latitude":  c[1],
            "longitude": c[0],
            "depth":     c[2] if c[2] is not None else 33.0,
            "magnitude": p["mag"],
            "datetime":  datetime.fromtimestamp(
                p["time"] / 1000, tz=timezone.utc
            ).isoformat(),
        })
    return events


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Service status and component availability."""
    rolling_ok = not catalog.rolling_df.empty
    rolling_n  = len(catalog.rolling_df) if rolling_ok else 0
    rolling_latest = (
        catalog.rolling_df["datetime"].max().isoformat()
        if rolling_ok else None
    )
    return {
        "status":          "ok" if scorer.ready else "degraded",
        "models_loaded":   list(scorer.models.keys()),
        "rolling_catalog": {
            "ok":    rolling_ok,
            "events": rolling_n,
            "latest": rolling_latest,
        },
        "external_data": {
            "gem_faults": catalog.gem_fault_union is not None,
            "wsm":        not catalog.wsm_df.empty,
            "plate_boundaries": catalog.pb_tree is not None,
            "historical_catalog": catalog.hist_tree is not None,
        },
        "poll_interval_minutes": POLL_INTERVAL_MINUTES,
    }


@app.get(
    "/score/latest",
    response_model=list[ScoreResponse],
    tags=["Scoring"],
    summary="Score the N most recent M≥4 earthquakes from USGS",
)
def score_latest(
    n: int = Query(10, ge=1, le=50,
                   description="Number of recent events to score"),
):
    """
    Fetch and score the N most recent M≥4.0 earthquakes from the USGS
    real-time feed. Results are also added to the /events cache.
    """
    if not scorer.ready:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    events = _fetch_usgs_latest(n)
    results = []
    for ev in events:
        dt = _parse_dt(ev["datetime"])
        try:
            r = _score_event(
                ev["latitude"], ev["longitude"],
                ev["depth"], ev["magnitude"], dt, ev["event_id"]
            )
            results.append(r)
        except Exception as e:
            logger.error(f"Scoring failed for {ev['event_id']}: {e}")

    logger.info(f"Scored {len(results)}/{len(events)} latest events.")
    return results


@app.post(
    "/score",
    response_model=ScoreResponse,
    tags=["Scoring"],
    summary="Score a single earthquake event",
)
def score_single(event: EventInput):
    """
    Score a single earthquake event provided manually.
    The `datetime` field defaults to now (UTC) if omitted.
    """
    if not scorer.ready:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    event_dt = _parse_dt(event.datetime)
    result   = _score_event(
        event.latitude, event.longitude,
        event.depth, event.magnitude,
        event_dt, event.event_id,
    )
    return result


@app.get(
    "/events",
    response_model=list[ScoreResponse],
    tags=["History"],
    summary="List recently scored events",
)
def list_events(
    limit: int = Query(50, ge=1, le=MAX_CACHE_SIZE,
                       description="Max events to return"),
    min_prob_7d: Optional[float] = Query(
        None, ge=0.0, le=1.0,
        description="Filter: only events with prob_7d ≥ this value"
    ),
):
    """
    Returns the most recently scored events from the in-memory cache,
    ordered from most recent to oldest.
    """
    events = list(reversed(_scored_cache))
    if min_prob_7d is not None:
        events = [e for e in events
                  if e.get("prob_7d") is not None
                  and e["prob_7d"] >= min_prob_7d]
    return events[:limit]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
