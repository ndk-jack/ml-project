"""
database.py — Supabase persistence layer for scored earthquake events.

Responsibilities:
  - Create Supabase client from environment variables
  - insert_scored_event(result_dict) → persist a scored event (fire-and-forget safe)
  - get_recent_events(limit, min_prob_7d) → query Supabase for the /events endpoint

Environment variables required:
  SUPABASE_URL               — e.g. https://sfykwnhynwwuvientblh.supabase.co
  SUPABASE_SERVICE_KEY       — preferred service_role key name
  SUPABASE_SERVICE_ROLE_KEY  — accepted fallback name for compatibility

Optional:
  SUPABASE_ENABLED           — set to "false" to disable persistence (local dev)

Table expected in Supabase (run migrations/001_scored_events.sql first):
  scored_events (see migrations/ for full schema)
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from .serving_metadata import get_serving_metadata

logger = logging.getLogger(__name__)

# ── Lazy client — only imported if Supabase is enabled ────────────────────────

_client = None
_enabled: Optional[bool] = None


def _get_supabase_credentials() -> tuple[Optional[str], Optional[str]]:
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )
    return url, key


def _is_enabled() -> bool:
    global _enabled
    if _enabled is not None:
        return _enabled
    if os.getenv("SUPABASE_ENABLED", "true").lower() == "false":
        logger.info("Supabase persistence disabled (SUPABASE_ENABLED=false).")
        _enabled = False
        return False
    url, key = _get_supabase_credentials()
    if not url or not key:
        logger.warning(
            "Supabase credentials not found. "
            "Set SUPABASE_URL and SUPABASE_SERVICE_KEY "
            "(or SUPABASE_SERVICE_ROLE_KEY) to enable persistence."
        )
        _enabled = False
        return False
    _enabled = True
    return True


def _get_client():
    """Return Supabase client, creating it on first call."""
    global _client
    if _client is not None:
        return _client

    try:
        from supabase import create_client  # type: ignore
    except ImportError:
        raise ImportError(
            "supabase package not installed in the active environment. "
            "Install it in the project venv with: pip install supabase"
        )

    url, key = _get_supabase_credentials()
    _client = create_client(url, key)
    logger.info("Supabase client created.")
    return _client


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_features_used(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def insert_scored_event(result: dict) -> bool:
    """
    Insert a scored event into the Supabase scored_events table.
    Returns True on success, False on failure (never raises — safe to call
    in background paths without disrupting the API response).

    Parameters
    ----------
    result : dict
        The full response dict from _score_event() in main.py.
        Expected keys: event_id, latitude, longitude, depth, magnitude,
                       datetime, prob_7d, prob_30d, prob_365d,
                       risk_7d, risk_30d, risk_365d, features_used, scored_at.
        Serving metadata (model_version, benchmark_id, feature_set_version,
        dataset_version, mlflow_run_id) is injected automatically from
        serving_metadata.py — not read from result.
    """
    if not _is_enabled():
        return False

    try:
        meta = get_serving_metadata()
        row = {
            "event_id":            result.get("event_id"),
            "latitude":            result.get("latitude"),
            "longitude":           result.get("longitude"),
            "depth":               result.get("depth"),
            "magnitude":           result.get("magnitude"),
            "event_datetime":      result.get("datetime"),
            "prob_7d":             result.get("prob_7d"),
            "prob_30d":            result.get("prob_30d"),
            "prob_365d":           result.get("prob_365d"),
            "risk_7d":             result.get("risk_7d"),
            "risk_30d":            result.get("risk_30d"),
            "risk_365d":           result.get("risk_365d"),
            "features_used":       _normalize_features_used(result.get("features_used")),
            "scored_at":           result.get("scored_at"),
            "model_version":       meta["model_version"],
            "benchmark_id":        meta["benchmark_id"],
            "feature_set_version": meta["feature_set_version"],
            "dataset_version":     meta["dataset_version"],
            "mlflow_run_id":       meta["mlflow_run_id"],
        }

        client = _get_client()
        # upsert: if same event_id scored again, update rather than duplicate
        response = (
            client.table("scored_events")
            .upsert(row, on_conflict="event_id")
            .execute()
        )

        if hasattr(response, "data") and response.data:
            logger.debug(f"Persisted event {row['event_id']} to Supabase.")
            return True
        else:
            logger.warning(f"Supabase upsert returned no data: {response}")
            return False

    except Exception as e:
        logger.error(f"Failed to persist event to Supabase: {e}")
        return False


def insert_prediction_log(result: dict) -> str | None:
    """
    Append-only log of served predictions.
    Returns prediction_id on success, None on failure.
    """
    if not _is_enabled():
        return None

    try:
        client = _get_client()
        meta = get_serving_metadata()

        row = {
            "event_id": result.get("event_id"),
            "event_datetime": result.get("datetime"),
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
            "depth": result.get("depth"),
            "magnitude": result.get("magnitude"),
            "prob_7d": result.get("prob_7d"),
            "prob_30d": result.get("prob_30d"),
            "risk_7d": result.get("risk_7d"),
            "risk_30d": result.get("risk_30d"),
            "features_used": _normalize_features_used(result.get("features_used")),
            "scored_at": result.get("scored_at"),
            "model_version": meta["model_version"],
            "benchmark_id": meta["benchmark_id"],
            "feature_set_version": meta["feature_set_version"],
            "dataset_version": meta["dataset_version"],
            "mlflow_run_id": meta["mlflow_run_id"],
        }

        response = client.table("prediction_log").insert(row).execute()

        if hasattr(response, "data") and response.data:
            prediction_id = response.data[0].get("prediction_id")
            logger.info(f"Prediction log inserted for event {row['event_id']}.")
            return prediction_id

        logger.warning(f"Prediction log insert returned no data: {response}")
        return None

    except Exception as e:
        logger.error(f"Failed to insert prediction_log: {e}")
        return None


def insert_prediction_outcome_stub(prediction_id: str, result: dict) -> bool:
    """Deprecated: delayed outcomes are now evaluated in ml-data-plane."""
    logger.warning(
        "insert_prediction_outcome_stub is deprecated: "
        "delayed outcomes now live in ml-data-plane"
    )
    return False


def get_recent_events(
    limit: int = 50,
    min_prob_7d: Optional[float] = None,
) -> list[dict]:
    """
    Query the most recent scored events from Supabase.
    Falls back to an empty list if Supabase is unavailable.

    Parameters
    ----------
    limit       : max rows to return (default 50)
    min_prob_7d : optional filter — only events with prob_7d >= this value
    """
    if not _is_enabled():
        return []

    try:
        client = _get_client()
        query = (
            client.table("scored_events")
            .select("*")
            .order("scored_at", desc=True)
            .limit(limit)
        )
        if min_prob_7d is not None:
            query = query.gte("prob_7d", min_prob_7d)

        response = query.execute()
        rows = response.data if hasattr(response, "data") else []

        # Normalise column names to match ScoreResponse schema
        # (event_time → datetime, so the API response stays consistent)
        for row in rows:
            if "event_datetime" in row and "datetime" not in row:
                row["datetime"] = row.pop("event_datetime")

        return rows

    except Exception as e:
        logger.error(f"Failed to query Supabase: {e}")
        return []


def prediction_log_exists_for_event_model(event_id: str, model_version: str) -> bool:
    """Return True if prediction_log already has an entry for (event_id, model_version)."""
    if not _is_enabled():
        return False

    try:
        client = _get_client()
        response = (
            client.table("prediction_log")
            .select("prediction_id")
            .eq("event_id", event_id)
            .eq("model_version", model_version)
            .limit(1)
            .execute()
        )
        rows = response.data if hasattr(response, "data") and response.data else []
        return len(rows) > 0
    except Exception as e:
        logger.error(f"Failed to check prediction_log for {event_id}/{model_version}: {e}")
        return False


def is_ready() -> bool:
    """Return True if Supabase is configured and reachable."""
    if not _is_enabled():
        return False
    try:
        _get_client()
        return True
    except Exception:
        return False


PUBLIC_SCORED_EVENT_COLUMNS = ",".join([
    "event_id",
    "event_datetime",
    "latitude",
    "longitude",
    "depth",
    "magnitude",
    "prob_7d",
    "prob_30d",
    "prob_365d",
    "risk_7d",
    "risk_30d",
    "risk_365d",
    "scored_at",
    "model_version",
    "benchmark_id",
    "feature_set_version",
    "dataset_version",
])


_PUBLIC_REQUIRED_COLS = ("event_id", "event_datetime", "scored_at")


def list_scored_events_public(limit: int = 50) -> list[dict]:
    """
    Return scored events for the public API feed.

    Required fields for a row to be considered valid:
      - event_id        (primary identifier)
      - event_datetime  (event time)
      - scored_at       (when the prediction was made)

    Rows missing any of these are filtered at the DB level.
    """
    if not _is_enabled():
        return []

    try:
        client = _get_client()
        query = (
            client.table("scored_events")
            .select(PUBLIC_SCORED_EVENT_COLUMNS)
            .order("scored_at", desc=True)
            .limit(limit)
        )
        for col in _PUBLIC_REQUIRED_COLS:
            query = query.not_.is_(col, "null")

        response = query.execute()
        return response.data if hasattr(response, "data") and response.data else []
    except Exception as e:
        logger.error(f"Failed to list scored_events public: {e}")
        return []


def get_scored_event_public(event_id: str) -> Optional[dict]:
    if not _is_enabled():
        return None

    try:
        client = _get_client()
        response = (
            client.table("scored_events")
            .select(PUBLIC_SCORED_EVENT_COLUMNS)
            .eq("event_id", event_id)
            .limit(1)
            .execute()
        )
        rows = response.data if hasattr(response, "data") and response.data else []
        return rows[0] if rows else None
    except Exception as e:
        logger.error(f"Failed to get scored_event public for {event_id}: {e}")
        return None
