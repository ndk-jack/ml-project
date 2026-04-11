import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError

from . import database
from .schemas_public import (
    HealthV1Response,
    ScoredEventDetailResponse,
    ScoredEventPublic,
    ScoredEventsListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["public"])


@router.get("/health", response_model=HealthV1Response)
def api_v1_health():
    return {"status": "ok"}


@router.get("/scored-events", response_model=ScoredEventsListResponse)
def api_v1_list_scored_events(limit: int = Query(50, ge=1, le=200)):
    raw_rows = database.list_scored_events_public(limit=limit)

    valid_rows = []
    rejected = 0
    for row in raw_rows:
        try:
            valid_rows.append(ScoredEventPublic.model_validate(row))
        except ValidationError:
            rejected += 1

    if rejected:
        logger.warning(
            "list_scored_events: rejected %d/%d rows that failed schema validation",
            rejected,
            len(raw_rows),
        )

    return {
        "data": [row.model_dump(mode="json") for row in valid_rows],
        "meta": {
            "count": len(valid_rows),
            "limit": limit,
            "rejected": rejected,
        },
    }


@router.get("/scored-events/{event_id}", response_model=ScoredEventDetailResponse)
def api_v1_get_scored_event(event_id: str):
    row = database.get_scored_event_public(event_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    try:
        validated = ScoredEventPublic.model_validate(row)
    except ValidationError as e:
        logger.error(
            "Invalid scored_event payload for event_id=%s validation_error=%s",
            event_id,
            e,
        )
        raise HTTPException(status_code=500, detail="Invalid scored event payload")

    return {"data": validated.model_dump(mode="json")}