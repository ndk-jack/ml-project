from datetime import datetime
from typing import Optional

from pydantic import BaseModel, model_validator


def _prob_to_risk_code(prob: Optional[float]) -> Optional[str]:
    """
    Convert a probability to a stable risk code.

    Thresholds (same bands as getRiskLabel in the frontend):
      prob >= 0.70  → very_high
      prob >= 0.50  → high
      prob >= 0.35  → moderate
      prob >= 0.20  → low
      prob <  0.20  → very_low
    """
    if prob is None:
        return None
    if prob >= 0.70:
        return "very_high"
    if prob >= 0.50:
        return "high"
    if prob >= 0.35:
        return "moderate"
    if prob >= 0.20:
        return "low"
    return "very_low"


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
    # Human-readable labels (kept for UI compat — do not parse in code)
    risk_7d: str
    risk_30d: str
    risk_365d: Optional[str] = None
    # Stable machine-readable codes — use these for filters/logic/i18n
    risk_7d_code: Optional[str] = None
    risk_30d_code: Optional[str] = None
    risk_365d_code: Optional[str] = None
    scored_at: datetime
    model_version: Optional[str] = None
    benchmark_id: Optional[str] = None
    feature_set_version: Optional[str] = None
    dataset_version: Optional[str] = None

    @model_validator(mode="after")
    def populate_risk_codes(self) -> "ScoredEventPublic":
        if self.risk_7d_code is None:
            self.risk_7d_code = _prob_to_risk_code(self.prob_7d)
        if self.risk_30d_code is None:
            self.risk_30d_code = _prob_to_risk_code(self.prob_30d)
        if self.risk_365d_code is None:
            self.risk_365d_code = _prob_to_risk_code(self.prob_365d)
        return self


class ScoredEventsListMeta(BaseModel):
    count: int
    limit: int
    rejected: int = 0


class ScoredEventsListResponse(BaseModel):
    data: list[ScoredEventPublic]
    meta: ScoredEventsListMeta


class ScoredEventDetailResponse(BaseModel):
    data: ScoredEventPublic
