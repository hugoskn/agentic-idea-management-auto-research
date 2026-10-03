from enum import Enum

from pydantic import BaseModel, Field


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IdeaRanking(BaseModel):
    idea_id: str
    score: int = Field(ge=0, le=100)
    reasoning: str
    expected_impact: str
    confidence: Confidence
    key_assumptions: list[str]
    main_risks: list[str]


class RankingResult(BaseModel):
    calibration_notes: str = Field(description="Why the score distribution is spread the way it is.")
    rankings: list[IdeaRanking]
