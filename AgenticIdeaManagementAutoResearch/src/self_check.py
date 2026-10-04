import asyncio
import os
import sys
import tempfile
from pathlib import Path

from claude_agent_sdk import ResultMessage

from agents.base import failure_detail
from main import describe_connection, folder_name, load_env, write_csv
import agents.acquisition as acquisition
import agents.idea_generator as generator
from agents.acquisition import check_dispatch
from models.experiment import BranchAssignment, ClusterDecision, DispatchPlan, IdeaPick
from models.ranking import RankingResult
import agents.ranker as ranker
import agents.solver as solver
import orchestration.research_loop as loop
from agents.clusterizer import check_clustering
from agents.idea_generator import check_expansion, to_drafts
from main import ranking_text
from models.ranking import ClusterRank, ClusterRanking, IdeaRank, IdeaRanking
from models.audit import AuditFlag, AuditResult, AuditVerdict
from models.cluster import Cluster, ClusteringResult, IdeaAssignment
from models.experiment import Evaluation, Experiment, SearchMode, SolverReport
from models.idea import ExpandKind, Expansion, IdeaContent, IdeaDraft, IdeaOrigin, IdeaStatus, Proposal
from models.research_state import IterationRecord, ResearchState, ResourcePlan
from orchestration.research_loop import ResearchConfig, check_stop, evaluate, learn, recommend
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

e1 = Experiment(id="E1", idea_id="I1", iteration=1, cluster_action=SearchMode.EXPLOITATION, idea_action=SearchMode.EXPLOITATION, workdir="w", report=report(False))
e2 = Experiment(id="E2", idea_id="I2", iteration=1, cluster_action=SearchMode.EXPLORATION, idea_action=SearchMode.EXPLORATION, workdir="w", report=report(False))
e3 = Experiment(id="E3", idea_id="I3", iteration=1, cluster_action=SearchMode.EXPLORATION, idea_action=SearchMode.EXPLORATION, workdir="w", report=report(False))
state.experiments += [e1, e2, e3]
state.iterations[0].experiment_ids = ["E1", "E2", "E3"]
learn(state, e1, audit(e1, 60), config)
learn(state, e2, audit(e2, 90, flags=[AuditFlag.IDEA_MISMATCH, AuditFlag.REWARD_HACKING], rebuilt=draft("z")), config)
learn(state, e3, audit(e3, 10), config)
assert state.ideas["I1"].status == IdeaStatus.EVIDENCE_COLLECTED
assert state.ideas["I2"].status == IdeaStatus.RANKED
assert state.ideas["I3"].status == IdeaStatus.DISCARDED
assert [a.idea_id for a in state.trusted_audits()] == ["I1", "I3"]
assert not any("I2" in lesson for lesson in state.trusted_lessons())
assert state.best_audit().idea_id == "I1"
assert check_stop(state, config) == ""

for iteration in (2, 3):
    state.iterations.append(IterationRecord(iteration=iteration, experiment_ids=[f"X{iteration}"]))
assert stagnation(state) == 2
assert "No improvement" in check_stop(state, config)

e4 = Experiment(id="E4", idea_id="I2", iteration=3, cluster_action=SearchMode.EXPLOITATION, idea_action=SearchMode.EXPLOITATION, workdir="w", report=report(True))
state.experiments.append(e4)
learn(state, e4, audit(e4, 92, solved=True), config)
assert "Sufficiently good" in check_stop(state, config)
assert recommend(state).startswith("I2 b (audit score 92, solves the problem)")

e5 = Experiment(id="E5", idea_id="I3", iteration=3, cluster_action=SearchMode.EXPLORATION, idea_action=SearchMode.EXPLORATION, workdir="w", report=report(False))
state.experiments.append(e5)
state.iterations[2].experiment_ids.append("E5")
mismatch = audit(e5, 70, flags=[AuditFlag.IDEA_MISMATCH], rebuilt=draft("rebuilt"))
learn(state, e5, mismatch, config)
rebuilt = state.ideas["I4"]
assert rebuilt.origin == IdeaOrigin.RECONSTRUCTED and rebuilt.parent_ids == ["I3"] and rebuilt.iteration == 3
assert rebuilt.status == IdeaStatus.EVIDENCE_COLLECTED and state.ideas["I3"].status == IdeaStatus.DISCARDED
assert mismatch.idea_id == "I4" and mismatch.trusted and not mismatch.legit
assert any(a.idea_id == "I4" and a.experiment_id == "E5" for a in state.trusted_audits())
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

    credentials = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")
    saved = {k: os.environ.pop(k, None) for k in credentials}
    try:
        oauth = Path(tmp) / "oauth.env"
        oauth.write_text("ANTHROPIC_API_KEY=sk-ant-oat01-fake\n", encoding="utf-8")
        load_env(oauth)
        assert "ANTHROPIC_API_KEY" not in os.environ and os.environ["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-fake"
        assert "credential=CLAUDE_CODE_OAUTH_TOKEN" in describe_connection()
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-oat01-shell"
        load_env(Path(tmp) / "missing.env")
        assert "ANTHROPIC_API_KEY" not in os.environ and os.environ["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-fake"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-real"
        load_env(Path(tmp) / "missing.env")
        assert "credential=ANTHROPIC_API_KEY" in describe_connection()
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

rejected_auth = ResultMessage(
    subtype="success", duration_ms=1, duration_api_ms=1, is_error=True, num_turns=1, session_id="s",
    result="Failed to authenticate. API Error: 401 API key is invalid.", api_error_status=401,
)
assert "API status 401" in failure_detail(rejected_auth) and "API key is invalid" in failure_detail(rejected_auth)
assert failure_detail(None) == "no result message"

def raises(check, *args) -> bool:
    try:
        check(*args)
    except ValueError:
        return True
    return False


assert not raises(ranker.check_dense, [("a", 1), ("b", 2), ("c", 2), ("d", 3)], {"a", "b", "c", "d"}, "idea")
assert not raises(ranker.check_dense, [("a", 1), ("b", 1)], {"a", "b"}, "idea")
assert raises(ranker.check_dense, [("a", 1), ("b", 2), ("c", 2), ("d", 4)], {"a", "b", "c", "d"}, "idea")
assert raises(ranker.check_dense, [("a", 2), ("b", 3)], {"a", "b"}, "idea")
assert raises(ranker.check_dense, [("a", 1), ("a", 2)], {"a", "b"}, "idea")
assert raises(ranker.check_dense, [("a", 1), ("x", 2)], {"a", "b"}, "idea")


def clustering(*groups: list[str]) -> ClusteringResult:
    return ClusteringResult(solution_space_summary="s", clusters=[
        Cluster(id=f"C{n + 1}", name="n", description="d", fundamentally_different=False, members=[IdeaAssignment(idea_id=i, reason="r") for i in g])
        for n, g in enumerate(groups)
    ])


assert not raises(check_clustering, clustering(["I1", "I2"], ["I3"]), {"I1", "I2", "I3"})
assert not raises(check_clustering, clustering(["I1"]), {"I1"})
assert raises(check_clustering, clustering(["I1", "I2", "I3"]), {"I1", "I2", "I3"})
assert raises(check_clustering, clustering(*[[f"I{n}"] for n in range(1, 7)]), {f"I{n}" for n in range(1, 7)})
assert raises(check_clustering, clustering(["I1", "I2"], ["I2", "I3"]), {"I1", "I2", "I3"})
assert raises(check_clustering, clustering(["I1"], []), {"I1"})

calls = []


async def fake_ask(agent, system_prompt, payload, output_model, check=lambda _: None, **options):
    calls.append((agent, payload))
    if output_model is ClusterRanking:
        out = ClusterRanking(cluster_ranks=[ClusterRank(cluster_id="C1", rank=1), ClusterRank(cluster_id="C2", rank=2)], rationale="c1 leads")
    else:
        ids = [i["id"] for i in payload["ideas_to_rank"]]
        out = IdeaRanking(idea_ranks=[IdeaRank(idea_id=i, rank=n + 1) for n, i in enumerate(ids)], rationale="ordered")
    check(out)
    return out


ranker.ask = fake_ask
pool = ResearchState(problem="p", remaining_budget=1).add_ideas([draft(t) for t in "abcd"], 1)
estimated = asyncio.run(ranker.estimate("p", clustering(["I1", "I2", "I3"], ["I4"]), pool[1:], {"I1": 60.0}, ["lesson"]))
assert [agent for agent, _ in calls] == ["Ranker[clusters]", "Ranker[C1]"]
stats = calls[0][1]["clusters"]
assert stats[0]["n_evaluated"] == 1 and stats[0]["best_evaluated_score"] == 60 and stats[1]["best_evaluated_score"] is None
assert calls[0][1]["lessons"] == ["lesson"] and [i["id"] for i in calls[1][1]["ideas_to_rank"]] == ["I2", "I3"]
assert estimated.cluster_rank("C2") == 2 and estimated.idea_rank("I3") == "2/2" and estimated.idea_rank("I4") == "1/1"
assert estimated.idea_rank("I1") is None
shown = ranking_text(IterationRecord(iteration=1, clustering=clustering(["I1", "I2", "I3"], ["I4"]), ranking=estimated))
assert shown.splitlines()[1].startswith("#1 C1 n: I2=1, I3=2") and "#2 C2 n: I4=1" in shown

python = f'"{sys.executable}" -c'
with tempfile.TemporaryDirectory() as tmp:
    def run(code: str) -> Evaluation:
        return asyncio.run(evaluate(f'{python} "{code}"', Path(tmp), 30))

    ok = run("print('noise'); print(87.5)")
    assert ok.valid and ok.score == 87.5 and "noise" in ok.output
    assert run("print(150)").score == 100
    for bad in ("print('fast')", "import sys; print(90); sys.exit(1)", "print(float('nan'))", ""):
        result = run(bad)
        assert not result.valid and result.score == 0, (bad, result)
slow = asyncio.run(evaluate(f'{python} "import time; time.sleep(3)"', Path.cwd(), 1))
assert not slow.valid and "timed out" in slow.output


class FakeClient:
    def __init__(self, options):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


prompts = []


async def fake_send(client, agent, prompt, model):
    prompts.append(prompt)
    return report(False)


def scripted(scores: list[float]):
    remaining = iter(scores)

    async def scorer() -> Evaluation:
        return Evaluation(valid=True, score=next(remaining), output="evaluator says hi")

    return scorer


solver.ClaudeSDKClient, solver.send = FakeClient, fake_send
idea = state.ideas["I1"]
_, rounds, evals = asyncio.run(solver.run_solver("p", idea, None, [], Path("."), 2, scripted([40, 60, 70])))
assert rounds == 2 and [e.score for e in evals] == [40, 60, 70] and "evaluator says hi" in prompts[-1]
_, rounds, evals = asyncio.run(solver.run_solver("p", idea, None, [], Path("."), 2, scripted([100])))
assert rounds == 0 and [e.score for e in evals] == [100]
_, rounds, evals = asyncio.run(solver.run_solver("p", idea, None, [], Path("."), 0, scripted([55])))
assert rounds == 0 and [e.score for e in evals] == [55]
_, rounds, evals = asyncio.run(solver.run_solver("p", idea, None, [], Path("."), 2))
assert rounds == 2 and evals == []


async def fake_solver(*args):
    return report(False), 1, [Evaluation(valid=True, score=30, output=""), Evaluation(valid=True, score=88.5, output="")]


async def fake_audit(*args):
    return AuditVerdict(flags=[], confidence=1, reasoning="r", task_solved=False, score=10, evidence=[], discrepancies=[], lessons_learned=[])


loop.run_solver, loop.audit_experiment = fake_solver, fake_audit
scored = Experiment(id="E9", idea_id="I1", iteration=3, cluster_action=SearchMode.EXPLOITATION, idea_action=SearchMode.EXPLOITATION, workdir=".")
result = asyncio.run(loop.run_experiment("p", idea, None, [], scored, ResearchConfig(evaluator="any")))
assert result.score == 88.5 and result.task_solved and len(scored.evaluations) == 2


def proposal(kind: ExpandKind, sibling: str = "", lesson: str = "") -> Proposal:
    content = draft(kind.value).model_dump(include=set(IdeaContent.model_fields))
    return Proposal(**content, kind=kind, sibling_referenced=sibling, lesson_context_idea=lesson)


K = ExpandKind
assert not raises(check_expansion, Expansion(proposals=[proposal(K.PUSH_SCORE), proposal(K.NEW_IDEA)]), 3, True)
assert not raises(check_expansion, Expansion(proposals=[]), 3, False)
assert raises(check_expansion, Expansion(proposals=[proposal(K.NEW_IDEA), proposal(K.NEW_IDEA)]), 3, True)
assert raises(check_expansion, Expansion(proposals=[proposal(K.FIX_ERROR)]), 3, True)
assert raises(check_expansion, Expansion(proposals=[proposal(K.PUSH_SCORE)]), 3, False)
assert raises(check_expansion, Expansion(proposals=[proposal(K.PUSH_SCORE), proposal(K.NEW_IDEA)]), 1, True)
drafts = to_drafts(Expansion(proposals=[proposal(K.PUSH_SCORE), proposal(K.CROSS_POLLINATE, sibling="I7"), proposal(K.NEW_IDEA)]), "I1", {"I7"}, set())
assert [(d.origin, d.parent_ids) for d in drafts] == [
    (IdeaOrigin.REFINEMENT, ["I1"]), (IdeaOrigin.COMBINATION, ["I1", "I7"]), (IdeaOrigin.NEW_DIRECTION, []),
]
for bad in (proposal(K.CROSS_POLLINATE), proposal(K.CROSS_POLLINATE, "I7", "I2"), proposal(K.CROSS_POLLINATE, "I9"), proposal(K.CROSS_POLLINATE, lesson="I9")):
    assert to_drafts(Expansion(proposals=[bad]), "I1", {"I7"}, {"I2"}) == []
assert to_drafts(Expansion(proposals=[proposal(K.CROSS_POLLINATE, lesson="I2")]), "I1", {"I7"}, {"I2"})[0].parent_ids == ["I1"]

branches = ResearchState(problem="p", remaining_budget=0)
branches.add_ideas([draft(t) for t in "xyz"], 1)
previous = IterationRecord(iteration=1, experiment_ids=["E1", "E2", "E3"])
branches.iterations.append(previous)
good = Experiment(id="E1", idea_id="I1", iteration=1, cluster_action=SearchMode.EXPLOITATION, idea_action=SearchMode.EXPLOITATION, workdir="w", report=report(False),
                  evaluations=[Evaluation(valid=True, score=40, output="evaluator ok")])
hacked = Experiment(id="E2", idea_id="I2", iteration=1, cluster_action=SearchMode.EXPLORATION, idea_action=SearchMode.EXPLORATION, workdir="w", report=report(False))
crashed = Experiment(id="E3", idea_id="I3", iteration=1, cluster_action=SearchMode.EXPLORATION, idea_action=SearchMode.EXPLORATION, workdir="w", error="Solver failed: boom")
branches.experiments += [good, hacked, crashed]
branches.audit_results += [audit(good, 40), audit(hacked, 90, flags=[AuditFlag.REWARD_HACKING])]
rows = [row for _, row in loop.branch_outcomes(branches, previous)]
assert [(r["branch"], r["idea_id"], r["success"]) for r in rows] == [(0, "I1", True), (2, "I3", False)]
assert "evaluator ok" in rows[0]["detail"] and "boom" in rows[1]["detail"]

seen = []


async def fake_expand_ask(agent, system_prompt, payload, output_model, check=lambda _: None, **options):
    seen.append(payload)
    out = Expansion(proposals=[proposal(K.PUSH_SCORE if payload["outcome"]["success"] else K.FIX_ERROR)])
    check(out)
    return out


generator.ask = fake_expand_ask
expanded = asyncio.run(loop.expand(branches, previous, ResearchConfig()))
assert [(d.origin, d.parent_ids) for d in expanded] == [(IdeaOrigin.REFINEMENT, ["I1"]), (IdeaOrigin.FIX, ["I3"])]
assert [[s["idea_id"] for s in p["sibling_outcomes"]] for p in seen] == [["I3"], ["I1"]]
assert seen[0]["max_proposals"] == 3 and seen[0]["run_signals"]["top_3"][0]["idea_id"] == "I1"
assert asyncio.run(loop.expand(branches, previous, ResearchConfig(max_ideas_per_branch=0))) == []

X, E = SearchMode.EXPLORATION, SearchMode.EXPLOITATION


def dispatch(decisions: list, assignments: list, go: bool = True) -> DispatchPlan:
    return DispatchPlan(
        cluster_decisions=[ClusterDecision(cluster_id=c, action=a, n_branches=n, rationale="r") for c, a, n in decisions],
        branch_action_assignments=[
            BranchAssignment(branch=b, cluster_id=c, cluster_action=ca, idea_action=ia, rationale="r") for b, c, ca, ia in assignments
        ],
        rationale_summary="shape", continue_research=go,
    )


good_plan = dispatch([("C1", E, 2), ("C2", X, 1)], [(0, "C1", E, E), (1, "C1", E, X), (2, "C2", X, X)])
assert not raises(check_dispatch, good_plan, 3, {"C1": 2, "C2": 1})
assert not raises(check_dispatch, dispatch([], [], go=False), 3, {"C1": 1})
assert raises(check_dispatch, good_plan, 3, {"C1": 1, "C2": 2})
assert raises(check_dispatch, good_plan, 2, {"C1": 2, "C2": 1})
assert raises(check_dispatch, dispatch([("C1", E, 1), ("C1", E, 1)], [(0, "C1", E, E), (1, "C1", E, E)]), 2, {"C1": 2})
assert raises(check_dispatch, dispatch([("C1", E, 2)], [(0, "C1", E, E), (2, "C1", E, E)]), 2, {"C1": 2})
assert raises(check_dispatch, dispatch([("C1", E, 1), ("C2", X, 1)], [(0, "C1", X, E), (1, "C2", X, X)]), 2, {"C1": 1, "C2": 1})
assert raises(check_dispatch, dispatch([("C1", E, 1), ("C2", X, 1)], [(0, "C1", E, E), (1, "C1", E, X)]), 2, {"C1": 2, "C2": 1})
assert raises(check_dispatch, dispatch([("C9", E, 1)], [(0, "C9", E, E)]), 1, {"C1": 1})

picking = ResearchState(problem="p", remaining_budget=3)
for idea in picking.add_ideas([draft(t) for t in "abcd"], 1):
    idea.status = IdeaStatus.RANKED
picking.clusters = clustering(["I1", "I2", "I3"], ["I4"])
picking.ranking = RankingResult(
    clusters=ClusterRanking(cluster_ranks=[ClusterRank(cluster_id="C1", rank=1), ClusterRank(cluster_id="C2", rank=2)], rationale="r"),
    ideas={
        "C1": IdeaRanking(idea_ranks=[IdeaRank(idea_id="I1", rank=2), IdeaRank(idea_id="I2", rank=1), IdeaRank(idea_id="I3", rank=3)], rationale="r"),
        "C2": IdeaRanking(idea_ranks=[IdeaRank(idea_id="I4", rank=1)], rationale="r"),
    },
)
picking.iterations.append(IterationRecord(iteration=1))
dispatched = []


async def fake_dispatch_ask(agent, system_prompt, payload, output_model, check=lambda _: None, **options):
    dispatched.append((agent, system_prompt, payload))
    if output_model is DispatchPlan:
        out = dispatch([("C1", E, 2), ("C2", X, 1)], [(0, "C1", E, X), (1, "C1", E, X), (2, "C2", X, X)])
    else:
        out = IdeaPick(idea_id=payload["available_members"][-1]["id"], rationale="r")
    check(out)
    return out


acquisition.ask = fake_dispatch_ask
chosen = asyncio.run(acquisition.select_ideas(picking, list(picking.ideas.values()), 3, 1, 2))
assert [i["idea_id"] for i in dispatched[0][2]["clusters"][0]["top_untested"]] == ["I2", "I1", "I3"]
assert [agent for agent, _, _ in dispatched] == ["AcquisitionAgent[dispatch]", "AcquisitionAgent[b0]", "AcquisitionAgent[b1]"]
assert [(s.branch, s.idea_id, s.idea_action) for s in chosen.selected] == [(0, "I3", E), (1, "I1", E), (2, "I4", E)]
assert "Iteration 1" in dispatched[1][1] and dispatched[1][2]["branch"]["idea_action"] == "exploitation"
assert dispatched[2][2]["already_picked_by_other_branches"] == ["I3"]
dispatched.clear()
later = asyncio.run(acquisition.select_ideas(picking, list(picking.ideas.values()), 3, 2, 3))
assert [s.idea_action for s in later.selected] == [X, X, X] and "Iteration 1" not in dispatched[1][1]

crash = Experiment(id="E1", idea_id="I1", iteration=1, cluster_action=E, idea_action=E, workdir="w", error="Solver failed: x")
unaudited = Experiment(id="E2", idea_id="I2", iteration=1, cluster_action=E, idea_action=E, workdir="w", report=report(False))
picking.experiments += [crash, unaudited]
learn(picking, crash, None, config)
learn(picking, unaudited, None, config)
assert picking.ideas["I1"].status == IdeaStatus.DISCARDED and picking.ideas["I2"].status == IdeaStatus.RANKED

print("self-check passed")
