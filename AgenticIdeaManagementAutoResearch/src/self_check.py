import asyncio
import os
import tempfile
from pathlib import Path

from main import folder_name, load_env, write_csv
from models.audit import AuditFlag, AuditResult
from models.cluster import Cluster, ClusteringResult, IdeaAssignment
from models.experiment import Experiment, SearchMode, SolverReport
from models.idea import IdeaDraft, IdeaOrigin, IdeaStatus
from models.research_state import IterationRecord, ResearchState, ResourcePlan
from orchestration.research_loop import ResearchConfig, check_stop, learn, recommend
from orchestration.resource_planner import check_plan, plan_resources, stagnation


def draft(title: str) -> IdeaDraft:
    return IdeaDraft(title=title, description="d", how_it_addresses_problem="h", rationale="r", assumptions=[], risks=[])


def report(solved: bool) -> SolverReport:
    return SolverReport(
        understanding="u", implementation_summary="impl", changes_made=[], experiment_description="e",
        metrics=[], result="res", evidence=[], artifacts=[], believes_solved=solved, limitations=[],
    )


def audit(experiment: Experiment, score: int, flags: list[AuditFlag] = [], solved: bool = False, rebuilt: IdeaDraft | None = None) -> AuditResult:
    return AuditResult(
        experiment_id=experiment.id, idea_id=experiment.idea_id, flags=flags, confidence=0.9, reasoning="why",
        reconstructed_idea=rebuilt, task_solved=solved, score=score, evidence=[], discrepancies=[], lessons_learned=["lesson"],
    )


def rejected(plan: list[int], frozen: list[int], to_spend: int) -> bool:
    try:
        check_plan(ResourcePlan(plan=plan, rationale="r"), frozen, to_spend, config)
    except ValueError:
        return True
    return False


config = ResearchConfig(max_attempts_per_idea=2, discard_score=25, success_score=85, patience=2)
state = ResearchState(problem="  Speed up: the <API>?  ", remaining_budget=4)
ideas = state.add_ideas([draft("a"), draft("b"), draft("c")], 1)
assert [i.id for i in ideas] == ["I1", "I2", "I3"]
for idea in ideas:
    idea.status = IdeaStatus.RANKED
state.clusters = ClusteringResult(solution_space_summary="s", clusters=[
    Cluster(id="C1", name="x", description="x", fundamentally_different=True, members=[IdeaAssignment(idea_id="I1", reason="r"), IdeaAssignment(idea_id="I2", reason="r")]),
    Cluster(id="C2", name="y", description="y", fundamentally_different=True, members=[IdeaAssignment(idea_id="I3", reason="r")]),
])
state.iterations.append(IterationRecord(iteration=1))

assert asyncio.run(plan_resources(state, config)).plan == [4]
assert not rejected([5, 3, 2], [5], 5)
assert rejected([4, 3, 2], [5], 5)
assert rejected([5], [5], 5)
assert rejected([5, 3, 1], [5], 5)
assert rejected([5, 11], [5], 11)
assert rejected([5, 1, 1, 3], [5], 5)

e1 = Experiment(id="E1", idea_id="I1", iteration=1, mode=SearchMode.EXPLOITATION, workdir="w", report=report(False))
e2 = Experiment(id="E2", idea_id="I2", iteration=1, mode=SearchMode.EXPLORATION, workdir="w", report=report(False))
e3 = Experiment(id="E3", idea_id="I3", iteration=1, mode=SearchMode.EXPLORATION, workdir="w", report=report(False))
state.experiments += [e1, e2, e3]
state.iterations[0].experiment_ids = ["E1", "E2", "E3"]
learn(state, e1, audit(e1, 60), config)
learn(state, e2, audit(e2, 90, flags=[AuditFlag.IDEA_MISMATCH, AuditFlag.REWARD_HACKING], rebuilt=draft("z")), config)
learn(state, e3, audit(e3, 10), config)
assert state.ideas["I1"].status == IdeaStatus.EVIDENCE_COLLECTED
assert state.ideas["I2"].status == IdeaStatus.RANKED
assert state.ideas["I3"].status == IdeaStatus.DISCARDED
assert [e["idea_id"] for e in state.evidence()] == ["I1", "I3"]
assert not any("I2" in lesson for lesson in state.trusted_lessons())
assert state.best_audit().idea_id == "I1"
assert check_stop(state, config) == ""

for iteration in (2, 3):
    state.iterations.append(IterationRecord(iteration=iteration, experiment_ids=[f"X{iteration}"]))
assert stagnation(state) == 2
assert "No improvement" in check_stop(state, config)

e4 = Experiment(id="E4", idea_id="I2", iteration=3, mode=SearchMode.EXPLOITATION, workdir="w", report=report(True))
state.experiments.append(e4)
learn(state, e4, audit(e4, 92, solved=True), config)
assert "Sufficiently good" in check_stop(state, config)
assert recommend(state).startswith("I2 b (audit score 92, solves the problem)")

e5 = Experiment(id="E5", idea_id="I3", iteration=3, mode=SearchMode.EXPLORATION, workdir="w", report=report(False))
state.experiments.append(e5)
state.iterations[2].experiment_ids.append("E5")
mismatch = audit(e5, 70, flags=[AuditFlag.IDEA_MISMATCH], rebuilt=draft("rebuilt"))
learn(state, e5, mismatch, config)
rebuilt = state.ideas["I4"]
assert rebuilt.origin == IdeaOrigin.RECONSTRUCTED and rebuilt.parent_ids == ["I3"] and rebuilt.iteration == 3
assert rebuilt.status == IdeaStatus.EVIDENCE_COLLECTED and state.ideas["I3"].status == IdeaStatus.DISCARDED
assert mismatch.idea_id == "I4" and mismatch.trusted and not mismatch.legit
assert any(e["idea_id"] == "I4" and e["experiment_id"] == "E5" for e in state.evidence())
assert any(l.startswith("[I4]") for l in state.trusted_lessons())

assert folder_name(state.problem) == "Speed up the API"
assert folder_name("???") == "research"
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "results.csv"
    write_csv(state, path)
    text = path.read_text(encoding="utf-8-sig")
    assert text.startswith("Original problem,Generated ideas") and "reconstructed from I3: rebuilt" in text

    env = Path(tmp) / ".env"
    env.write_text('# AIM_CHECK_COMMENT=x\nAIM_CHECK_EMPTY=\nAIM_CHECK_QUOTED="k=1"\nAIM_CHECK_SHELL=file\n', encoding="utf-8")
    os.environ["AIM_CHECK_SHELL"] = "shell"
    load_env(env)
    assert os.environ["AIM_CHECK_QUOTED"] == "k=1"
    assert os.environ["AIM_CHECK_SHELL"] == "shell"
    assert "AIM_CHECK_EMPTY" not in os.environ and "# AIM_CHECK_COMMENT" not in os.environ
    load_env(Path(tmp) / "missing.env")

print("self-check passed")
