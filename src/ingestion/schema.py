from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ArticleIn(BaseModel):
    text: str = Field(min_length=10, max_length=10000)
    source: str = "unknown"
    published_at: datetime | None = None


class ClassificationResult(BaseModel):
    label: Literal["World", "Sports", "Business", "Sci/Tech"]
    confidence: float = Field(ge=0.0, le=1.0)
    model: Literal["distilbert", "baseline"]


class UrgencyResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    level: Literal["low", "medium", "high", "critical"]
    features_used: list[str]


class RiskBrief(BaseModel):
    summary: str
    risk_level: Literal["low", "medium", "high", "critical"]
    key_entities: list[str] = Field(max_length=5)
    recommended_action: str
    generated_by: Literal["llm", "fallback"]


class ArticleOut(BaseModel):
    classification: ClassificationResult
    urgency: UrgencyResult
    risk_brief: RiskBrief
    processing_ms: float
