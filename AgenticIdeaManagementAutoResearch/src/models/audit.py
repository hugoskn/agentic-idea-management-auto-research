from enum import Enum

from pydantic import BaseModel, Field

from models.idea import IdeaContent


class AuditFlag(str, Enum):
    TRIVIAL = "trivial"
    TASK_MISMATCH = "task_mismatch"
    IDEA_MISMATCH = "idea_mismatch"
    REWARD_HACKING = "reward_hacking"


DISCARD_FLAGS = {AuditFlag.TRIVIAL, AuditFlag.TASK_MISMATCH, AuditFlag.REWARD_HACKING}


class AuditVerdict(BaseModel):
    flags: list[AuditFlag] = Field(description="Failure modes that apply; empty when the solution is legitimate.")
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    reconstructed_idea: IdeaContent | None = Field(
        default=None,
        description="REQUIRED iff flags contains idea_mismatch: the idea the solution actually implements, in the same shape as the assigned idea.",
    )
    task_solved: bool
    score: float = Field(ge=0, le=100, description="How well the experiment, as verified, solves the original problem. Replaced by the evaluator's score when an evaluator is configured.")
    evidence: list[str]
    discrepancies: list[str]
    lessons_learned: list[str]


class AuditResult(AuditVerdict):
    experiment_id: str
    idea_id: str = Field(description="The idea the result is attributed to: the reconstructed idea when the only flag is idea_mismatch.")

    @property
    def legit(self) -> bool:
        return not self.flags

    @property
    def trusted(self) -> bool:
        return not DISCARD_FLAGS.intersection(self.flags)
