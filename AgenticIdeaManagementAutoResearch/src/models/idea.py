from enum import Enum

from pydantic import BaseModel, Field


class IdeaStatus(str, Enum):
    GENERATED = "generated"
    CLUSTERED = "clustered"
    RANKED = "ranked"
    SELECTED = "selected"
    IMPLEMENTED = "implemented"
    AUDITED = "audited"
    EVIDENCE_COLLECTED = "evidence_collected"
    REFINED = "refined"
    DISCARDED = "discarded"


class IdeaOrigin(str, Enum):
    INITIAL = "initial"
    REFINEMENT = "refinement"
    COMBINATION = "combination"
    FIX = "fix"
    NEW_DIRECTION = "new_direction"


class IdeaDraft(BaseModel):
    title: str
    description: str = Field(description="What the idea is, concrete enough for another agent to implement or test it.")
    how_it_addresses_problem: str
    rationale: str = Field(description="Why this idea has a high probability of solving or significantly contributing to the problem.")
    assumptions: list[str]
    risks: list[str] = Field(description="Main risks or reasons it might fail.")
    origin: IdeaOrigin = IdeaOrigin.INITIAL
    parent_ids: list[str] = Field(default_factory=list, description="Ids of existing ideas this one refines, fixes or combines.")


class IdeaBatch(BaseModel):
    problem_understanding: str
    ideas: list[IdeaDraft]


class Idea(IdeaDraft):
    id: str
    iteration: int
    status: IdeaStatus = IdeaStatus.GENERATED
