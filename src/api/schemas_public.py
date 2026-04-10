from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class HealthV1Response(BaseModel):
    status: str


class ScoredEventPublic(BaseModel):
    event_id: str
    event_datetime: datetime
    latitude: float
    longitude: float
    depth: float
    magnitude: float
    prob_7d: float
    prob_30d: float
    prob_365d: Optional[float] = None
    risk_7d: str
    risk_30d: str
    risk_365d: Optional[str] = None
    scored_at: datetime
    model_version: Optional[str] = None
    benchmark_id: Optional[str] = None
    feature_set_version: Optional[str] = None
    dataset_version: Optional[str] = None


class ScoredEventsListMeta(BaseModel):
    count: int
    limit: int


class ScoredEventsListResponse(BaseModel):
    data: list[ScoredEventPublic]
    meta: ScoredEventsListMeta


class ScoredEventDetailResponse(BaseModel):
    data: ScoredEventPublic
