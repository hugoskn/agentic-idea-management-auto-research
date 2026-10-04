from enum import Enum

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"


class ClusterDecision(BaseModel):
    cluster_id: str
    action: SearchMode
    n_branches: int = Field(ge=1)
    rationale: str


class BranchAssignment(BaseModel):
    branch: int = Field(ge=0)
    cluster_id: str
    cluster_action: SearchMode
    idea_action: SearchMode
    rationale: str


class DispatchPlan(BaseModel):
    cluster_decisions: list[ClusterDecision]
    branch_action_assignments: list[BranchAssignment]
    rationale_summary: str
    continue_research: bool = Field(description="False when further experimentation is unlikely to provide meaningful value.")
    stop_reason: str = ""


class IdeaPick(BaseModel):
    idea_id: str
    rationale: str = Field(description="2-4 sentences.")


class SelectedIdea(BaseModel):
    branch: int
    cluster_id: str
    cluster_action: SearchMode
    idea_action: SearchMode
    idea_id: str
    rationale: str


class SelectionResult(BaseModel):
    plan: DispatchPlan
    selected: list[SelectedIdea] = Field(default_factory=list)


class Metric(BaseModel):
    name: str
    value: str
    interpretation: str


class SolverReport(BaseModel):
    understanding: str = Field(description="The research direction as the solver understood it.")
    implementation_summary: str
    changes_made: list[str]
    experiment_description: str
    metrics: list[Metric]
    result: str
    evidence: list[str] = Field(description="Concrete evidence: commands run, outputs observed, files produced.")
    artifacts: list[str] = Field(description="Relative paths of files produced in the working directory.")
    believes_solved: bool
    limitations: list[str]


class Evaluation(BaseModel):
    valid: bool = Field(description="False if the evaluator failed, timed out or did not print a score.")
    score: float = Field(ge=0, le=100)
    output: str


class Experiment(BaseModel):
    id: str
    idea_id: str
    iteration: int
    cluster_action: SearchMode
    idea_action: SearchMode
    workdir: str
    refinement_rounds: int = 0
    report: SolverReport | None = None
    evaluations: list[Evaluation] = Field(default_factory=list)
    error: str | None = None
