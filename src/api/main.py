"""
main.py — FastAPI application for real-time earthquake sequence scoring.

Startup sequence:
1. CatalogManager.initialize() — load historical catalog + external data
2. Scorer.initialize() — load LightGBM models + runtime medians
3. APScheduler starts background polling every POLL_INTERVAL minutes
4. Initial rolling refresh is triggered in a background thread

Endpoints:
  GET  /health               — service status
  GET  /score/latest?n=10    — score the N most recent M≥4 USGS events
  POST /score                — score a single event (manual input)
  GET  /events               — list recently scored events (in-memory cache)
  GET  /api/v1/model-accuracy — latest model evaluation snapshot

Run:
  cd ~/ml-project
  source .venv/bin/activate
  python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import threading
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

from .catalog_manager import catalog
from .feature_engine import compute_features, features_to_series
from .scorer import scorer
from .router_public import router as public_router
from . import database

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

# ── Config ────────────────────────────────────────────────────────────────────

POLL_INTERVAL_MINUTES = 5
AUTO_SCORE_EVENTS_N   = 20
MAX_CACHE_SIZE        = 500
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

    catalog.initialize()
    scorer.initialize()

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        catalog.refresh_rolling,
        "interval",
        minutes=POLL_INTERVAL_MINUTES,
        id="usgs_poll",
        max_instances=1,
    )
    _scheduler.add_job(
        _auto_score_job,
        "interval",
        minutes=POLL_INTERVAL_MINUTES,
        id="auto_score",
        max_instances=1,
    )
    _scheduler.start()
    logger.info(f"USGS polling and auto-scoring started (every {POLL_INTERVAL_MINUTES} min)")

    threading.Thread(
        target=catalog.refresh_rolling,
        daemon=True,
        name="initial-rolling-refresh",
    ).start()

    logger.info("=== API ready ===")
    yield

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

app.include_router(public_router)

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

class ModelAccuracyResponse(BaseModel):
    model_version:  str
    horizon:        str
    sample_size:    int
    roc_auc:        Optional[float]
    brier_score:    Optional[float]
    positive_rate:  Optional[float]
    evaluated_at:   str
    notes:          Optional[str]

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

    _scored_cache.append(result)
    if len(_scored_cache) > MAX_CACHE_SIZE:
        _scored_cache.pop(0)

    database.insert_scored_event(result)
    database.insert_prediction_log(result)
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

def _auto_score_job() -> None:
    """Scheduled job: fetch latest USGS events, score, and persist to Supabase."""
    if not scorer.ready:
        logger.warning("auto_score_job: scorer not ready, skipping.")
        return

    from .serving_metadata import get_serving_metadata
    model_version = get_serving_metadata()["model_version"]

    try:
        url = USGS_LATEST_URL + f"&limit={AUTO_SCORE_EVENTS_N}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except Exception as e:
        logger.error(f"auto_score_job: USGS fetch failed: {e}")
        return

    scored = 0
    for f in features:
        p = f["properties"]
        c = f["geometry"]["coordinates"]
        if p.get("mag") is None:
            continue

        ev = {
            "event_id":  f["id"],
            "latitude":  c[1],
            "longitude": c[0],
            "depth":     c[2] if c[2] is not None else 33.0,
            "magnitude": p["mag"],
            "datetime":  datetime.fromtimestamp(
                p["time"] / 1000, tz=timezone.utc
            ).isoformat(),
        }

        try:
            if database.prediction_log_exists_for_event_model(ev["event_id"], model_version):
                continue
            dt = _parse_dt(ev["datetime"])
            _score_event(
                ev["latitude"], ev["longitude"],
                ev["depth"], ev["magnitude"], dt, ev["event_id"],
            )
            scored += 1
        except Exception as e:
            logger.error(f"auto_score_job: scoring failed for {ev['event_id']}: {e}")

    logger.info(f"auto_score_job: scored and persisted {scored}/{len(features)} events.")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Service status and component availability."""
    rolling_ok    = not catalog.rolling_df.empty and catalog.rolling_tree is not None
    historical_ok = catalog.hist_tree is not None and getattr(catalog, "hist_loaded", False)

    rolling_min = (
        catalog.rolling_df["datetime"].min().isoformat() if rolling_ok else None
    )
    rolling_max = (
        catalog.rolling_df["datetime"].max().isoformat() if rolling_ok else None
    )
    historical_min = (
        catalog.hist_df["datetime"].min().isoformat()
        if historical_ok and not catalog.hist_df.empty else None
    )
    historical_max = (
        catalog.hist_df["datetime"].max().isoformat()
        if historical_ok and not catalog.hist_df.empty else None
    )
    external_ok = (
        catalog.gem_fault_union is not None and
        not catalog.wsm_df.empty and
        catalog.pb_tree is not None
    )
    supabase_ok = database.is_ready()
    overall_ok  = scorer.ready and rolling_ok and historical_ok and external_ok

    return {
        "status": "ok" if overall_ok else "degraded",
        "startup_mode": "normal" if overall_ok else "degraded",
        "models_loaded": list(scorer.models.keys()),
        "model_count": len(scorer.models),
        "rolling_catalog": {
            "ok": rolling_ok,
            "events": getattr(catalog, "rolling_events", len(catalog.rolling_df)),
            "unique_event_ids": (
                int(catalog.rolling_df["event_id"].nunique())
                if rolling_ok and "event_id" in catalog.rolling_df.columns else 0
            ),
            "window_start": (
                catalog.rolling_window_start.isoformat()
                if getattr(catalog, "rolling_window_start", None) else rolling_min
            ),
            "window_end": (
                catalog.rolling_window_end.isoformat()
                if getattr(catalog, "rolling_window_end", None) else rolling_max
            ),
            "last_refresh_at": (
                catalog.rolling_last_refresh_at.isoformat()
                if getattr(catalog, "rolling_last_refresh_at", None) else None
            ),
            "min_datetime": rolling_min,
            "max_datetime": rolling_max,
        },
        "historical_catalog": {
            "ok": historical_ok,
            "events": getattr(catalog, "hist_events", 0),
            "source": getattr(catalog, "hist_source", None),
            "span_years": round(getattr(catalog, "hist_span_yr", 0.0), 2),
            "min_datetime": historical_min,
            "max_datetime": historical_max,
        },
        "external_data": {
            "gem_faults": catalog.gem_fault_union is not None,
            "wsm": not catalog.wsm_df.empty,
            "plate_boundaries": catalog.pb_tree is not None,
        },
        "scorer": {
            "ready": scorer.ready,
            "using_fallback_medians": getattr(scorer, "using_fallback_medians", None),
        },
        "poll_interval_minutes": POLL_INTERVAL_MINUTES,
        "supabase": supabase_ok,
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
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

    events  = _fetch_usgs_latest(n)
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


@app.get(
    "/api/v1/model-accuracy/history",
    response_model=list[ModelAccuracyResponse],
    tags=["Model Evaluation"],
    summary="Model evaluation history",
)
def model_accuracy_history(
    horizon: str = Query(default="7d", pattern="^(7d|30d)$"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Returns evaluation snapshots sorted by date descending."""
    return database.get_eval_snapshot_history(horizon=horizon, limit=limit)


@app.get(
    "/api/v1/model-accuracy",
    response_model=ModelAccuracyResponse,
    tags=["Model Evaluation"],
    summary="Latest model evaluation snapshot",
)
def model_accuracy(
    horizon: str = Query(default="7d", pattern="^(7d|30d)$"),
):
    """
    Returns the most recent evaluation snapshot from model_eval_snapshots.
    Includes ROC-AUC, Brier score, sample size and evaluation window.
    Updated automatically as new prediction outcomes are resolved.
    """
    snap = database.get_latest_eval_snapshot(horizon=horizon)
    if snap is None:
        raise HTTPException(
            status_code=404,
            detail="No evaluation snapshot available yet."
        )
    return snap


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )