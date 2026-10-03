from pydantic import BaseModel, Field


class AuditVerdict(BaseModel):
    valid: bool = Field(description="False if results are unsupported, fabricated, reward-hacked or otherwise invalid.")
    idea_implemented_correctly: bool
    task_solved: bool
    score: int = Field(ge=0, le=100, description="How well the experiment, as verified, solves the original problem.")
    evidence: list[str]
    discrepancies: list[str]
    actually_implemented_idea: str = Field(description="If the solver implemented a different idea, describe it; otherwise empty.")
    reward_hacking_detected: bool
    lessons_learned: list[str]


class AuditResult(AuditVerdict):
    experiment_id: str
    idea_id: str

    @property
    def trusted(self) -> bool:
        return self.valid and self.idea_implemented_correctly and not self.reward_hacking_detected
