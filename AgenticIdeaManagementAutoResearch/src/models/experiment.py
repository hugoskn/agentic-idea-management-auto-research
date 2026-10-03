from enum import Enum

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"


class SelectedIdea(BaseModel):
    idea_id: str
    priority: int = Field(ge=1, description="1 is the highest priority.")
    selection_reason: str
    exploration_or_exploitation: SearchMode


class SelectionResult(BaseModel):
    strategy_rationale: str
    selected: list[SelectedIdea]
    continue_research: bool = Field(description="False when further experimentation is unlikely to provide meaningful value.")
    stop_reason: str = ""


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


class Experiment(BaseModel):
    id: str
    idea_id: str
    iteration: int
    mode: SearchMode
    workdir: str
    refinement_rounds: int = 0
    report: SolverReport | None = None
    error: str | None = None
