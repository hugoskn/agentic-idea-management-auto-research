from pathlib import Path

from pydantic import BaseModel, Field

from models.audit import AuditResult
from models.cluster import ClusteringResult
from models.experiment import Experiment, SelectionResult
from models.idea import Idea, IdeaDraft, IdeaStatus
from models.ranking import IdeaRanking, RankingResult


class Lesson(BaseModel):
    iteration: int
    experiment_id: str
    idea_id: str
    text: str
    trusted: bool


class ResearchConfig(BaseModel):
    ideas_count: int = Field(default=6, ge=1)
    new_ideas_per_iteration: int = Field(default=3, ge=1)
    max_iterations: int = Field(default=3, ge=1)
    experiment_budget: int = Field(default=6, ge=1, description="Total number of Solver branches across the whole run.")
    initial_branches: int = Field(default=5, ge=1, description="Branches in iteration 1, pinned for an initial breadth of exploration.")
    min_branches: int = Field(default=3, ge=1, description="Preferred minimum branches per later iteration.")
    max_branches: int = Field(default=10, ge=1, description="Hard maximum branches per iteration.")
    success_score: int = Field(default=85, ge=0, le=100)
    discard_score: int = Field(default=25, ge=0, le=100)
    patience: int = Field(default=2, ge=1)
    max_attempts_per_idea: int = Field(default=2, ge=1)
    max_refinements: int = Field(default=2, ge=0)


class ResourcePlan(BaseModel):
    plan: list[int] = Field(description="Solver branches per iteration for the whole run, frozen prefix included.")
    rationale: str


class IterationRecord(BaseModel):
    iteration: int
    new_idea_ids: list[str] = Field(default_factory=list)
    clustering: ClusteringResult | None = None
    ranking: RankingResult | None = None
    plan: ResourcePlan | None = None
    selection: SelectionResult | None = None
    experiment_ids: list[str] = Field(default_factory=list)


class ResearchState(BaseModel):
    problem: str
    ideas: dict[str, Idea] = Field(default_factory=dict)
    clusters: ClusteringResult | None = None
    rankings: dict[str, IdeaRanking] = Field(default_factory=dict)
    experiments: list[Experiment] = Field(default_factory=list)
    audit_results: list[AuditResult] = Field(default_factory=list)
    lessons: list[Lesson] = Field(default_factory=list)
    remaining_budget: int
    iterations: list[IterationRecord] = Field(default_factory=list)
    stop_reason: str = ""
    final_recommendation: str = ""

    def add_ideas(self, drafts: list[IdeaDraft], iteration: int) -> list[Idea]:
        start = len(self.ideas) + 1
        added = [Idea(**d.model_dump(), id=f"I{start + n}", iteration=iteration) for n, d in enumerate(drafts)]
        self.ideas.update({i.id: i for i in added})
        return added

    def active_ideas(self) -> list[Idea]:
        return [i for i in self.ideas.values() if i.status != IdeaStatus.DISCARDED]

    def attempts(self, idea_id: str) -> int:
        return sum(e.idea_id == idea_id for e in self.experiments)

    def eligible_ideas(self, max_attempts: int) -> list[Idea]:
        return [i for i in self.ideas.values() if i.status == IdeaStatus.RANKED and self.attempts(i.id) < max_attempts]

    def trusted_audits(self) -> list[AuditResult]:
        return [a for a in self.audit_results if a.trusted]

    def best_audit(self) -> AuditResult | None:
        return max(self.trusted_audits(), key=lambda a: (a.task_solved, a.score), default=None)

    def best_scores(self) -> dict[str, int]:
        scores: dict[str, int] = {}
        for a in self.trusted_audits():
            scores[a.idea_id] = max(a.score, scores.get(a.idea_id, 0))
        return scores

    def trusted_lessons(self) -> list[str]:
        return [f"[{l.idea_id}] {l.text}" for l in self.lessons if l.trusted]

    def evidence(self) -> list[dict]:
        experiments = {e.id: e for e in self.experiments}
        return [
            {
                "idea_id": a.idea_id,
                "idea_title": self.ideas[a.idea_id].title,
                "experiment_id": a.experiment_id,
                "audit_score": a.score,
                "task_solved": a.task_solved,
                "result": experiments[a.experiment_id].report.result,
                "metrics": [m.model_dump() for m in experiments[a.experiment_id].report.metrics],
                "lessons": a.lessons_learned,
            }
            for a in self.trusted_audits()
        ]

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
