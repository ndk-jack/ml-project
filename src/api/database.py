"""
database.py — Supabase persistence layer for scored earthquake events.

Responsibilities:
  - Create Supabase client from environment variables
  - insert_scored_event(result_dict) → persist a scored event (fire-and-forget safe)
  - get_recent_events(limit, min_prob_7d) → query Supabase for the /events endpoint

Environment variables required:
  SUPABASE_URL          — e.g. https://sfykwnhynwwuvientblh.supabase.co
  SUPABASE_SERVICE_KEY  — service_role key (secret — never expose in frontend)

Optional:
  SUPABASE_ENABLED      — set to "false" to disable persistence (local dev)

Table expected in Supabase (run migrations/001_scored_events.sql first):
  scored_events (see migrations/ for full schema)
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy client — only imported if Supabase is enabled ────────────────────────

_client = None
_enabled: Optional[bool] = None


def _is_enabled() -> bool:
    global _enabled
    if _enabled is not None:
        return _enabled
    if os.getenv("SUPABASE_ENABLED", "true").lower() == "false":
        logger.info("Supabase persistence disabled (SUPABASE_ENABLED=false).")
        _enabled = False
        return False
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        logger.warning(
            "Supabase credentials not found. "
            "Set SUPABASE_URL and SUPABASE_SERVICE_KEY to enable persistence."
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
            "supabase package not installed. "
            "Run: pip3.9 install supabase --break-system-packages"
        )

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    _client = create_client(url, key)
    logger.info("Supabase client created.")
    return _client


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
                       risk_7d, risk_30d, risk_365d, features_used, scored_at
    """
    if not _is_enabled():
        return False

    try:
        row = {
            "event_id":      result.get("event_id"),
            "latitude":      result.get("latitude"),
            "longitude":     result.get("longitude"),
            "depth":         result.get("depth"),
            "magnitude":     result.get("magnitude"),
            "event_datetime": result.get("datetime"),
            "prob_7d":       result.get("prob_7d"),
            "prob_30d":      result.get("prob_30d"),
            "prob_365d":     result.get("prob_365d"),
            "risk_7d":       result.get("risk_7d"),
            "risk_30d":      result.get("risk_30d"),
            "risk_365d":     result.get("risk_365d"),
            "features_used": result.get("features_used"),
            "scored_at":     result.get("scored_at"),
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


def is_ready() -> bool:
    """Return True if Supabase is configured and reachable."""
    if not _is_enabled():
        return False
    try:
        _get_client()
        return True
    except Exception:
        return False
