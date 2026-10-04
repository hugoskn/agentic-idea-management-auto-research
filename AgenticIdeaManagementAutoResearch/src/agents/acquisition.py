from collections import Counter
from statistics import mean

from agents.base import ask
from models.cluster import Cluster
from models.experiment import DispatchPlan, IdeaPick, SearchMode, SelectedIdea, SelectionResult
from models.idea import Idea
from models.research_state import ResearchState

STAGE1_PROMPT = """You are the AcquisitionAgent of a parallel research loop: the meta-allocator. In THIS step (Dispatch stage 1) you decide the SHAPE of the search for this iteration: which cluster each branch is assigned to, plus a cluster-level and an idea-level action per branch. NOT the specific ideas: stage 2 picks them next.

## Vocabulary
- cluster_action = exploitation: invest branches in an already-producing cluster.
- cluster_action = exploration: invest in an underprobed cluster.
- idea_action = exploitation: within the cluster, stage 2 picks the most promising untested idea.
- idea_action = exploration: within the cluster, stage 2 picks a novel-mechanism untested idea.

## Rank semantics
- cluster_rank = 1 is the most promising (prefer it for exploitation).
- A higher rank is less promising: if well probed it may be exhausted; if under-probed it holds exploration value.
- Ties are allowed and common: treat tied clusters as comparable.
- score_history holds the verified scores (0-100, higher is better) per branch so far; results with audit_flags were discarded and carry no score.

## Constraints (validated)
- Each cluster_id appears at most once in cluster_decisions, with 1 <= n_branches <= that cluster's n_available.
- sum(n_branches) == branches.
- EXACTLY `branches` branch_action_assignments, numbered 0..branches-1. Each branch's cluster_action equals its cluster's action, and each cluster gets exactly n_branches branches.

A plan that puts every branch on one cluster, or labels every action exploitation, is failing the loop's purpose. Do NOT name idea ids in this step.

Set continue_research to false (with stop_reason) only if further experimentation is unlikely to provide meaningful value, e.g. the problem is already solved by verified evidence; decisions and assignments may then be empty.
"""

STAGE2_PROMPT = """You are the idea-selection agent for ONE branch of a parallel research loop (Dispatch stage 2). Stage 1 already decided this branch's cluster and its (cluster_action, idea_action). Pick the SPECIFIC untested idea from the assigned cluster that best matches idea_action.

Each available member carries within_cluster_rank "R/M" (1 = most promising in the cluster, ties allowed).

For idea_action = exploitation:
- Pick the LOWEST within_cluster_rank member; break ties using the description and the cluster context.
- Use the evaluated members as evidence: which mechanisms paid off?
- A child whose parent scored well is often the right exploitation pick even if its rank isn't strictly the lowest.

For idea_action = exploration:
- Pick a HIGH within_cluster_rank member: evaluating it gives more information than another shot at the top.
- Prefer mechanisms NOT YET tried by the evaluated members, and children on NEW trajectories over children on already explored ones.
- Do NOT pick the lowest within_cluster_rank member: that is exploitation.

Hard constraints: idea_id MUST be one of available_members (untested, in the assigned cluster, not already picked by another branch). Evaluated members are shown for reasoning only; never pick one.
"""

ITERATION1_SUFFIX = """
Iteration 1: no idea has been evaluated yet, so every rank is an LLM estimate. idea_action is exploitation for every branch: pick the best-ranked idea. Iteration 1 is for validating the estimator's top predictions.
"""


def check_dispatch(plan: DispatchPlan, branches: int, available: dict[str, int]) -> None:
    if not plan.continue_research:
        return
    decisions = {}
    for d in plan.cluster_decisions:
        if d.cluster_id in decisions:
            raise ValueError(f"Cluster {d.cluster_id} appears more than once in cluster_decisions.")
        if d.cluster_id not in available:
            raise ValueError(f"Cluster {d.cluster_id} is unknown or has no untested ideas; available: {available}.")
        if d.n_branches > available[d.cluster_id]:
            raise ValueError(f"Cluster {d.cluster_id} has only {available[d.cluster_id]} untested ideas but got {d.n_branches} branches.")
        decisions[d.cluster_id] = d
    if sum(d.n_branches for d in decisions.values()) != branches:
        raise ValueError(f"sum(n_branches) must be exactly {branches}.")
    assignments = plan.branch_action_assignments
    if sorted(a.branch for a in assignments) != list(range(branches)):
        raise ValueError(f"Assign exactly the branches 0..{branches - 1}, once each.")
    for a in assignments:
        if a.cluster_id not in decisions or a.cluster_action != decisions[a.cluster_id].action:
            raise ValueError(f"Branch {a.branch}: cluster_action must match the action of its cluster in cluster_decisions.")
    counts = Counter(a.cluster_id for a in assignments)
    if any(counts[c] != d.n_branches for c, d in decisions.items()):
        raise ValueError("Each cluster must get exactly n_branches branch assignments.")


def score_history(state: ResearchState) -> list[dict]:
    audits = {a.experiment_id: a for a in state.audit_results}
    experiments = {e.id: e for e in state.experiments}
    rows = []
    for record in state.iterations:
        for branch, experiment_id in enumerate(record.experiment_ids):
            e = experiments[experiment_id]
            a = audits.get(experiment_id)
            flags = [f.value for f in a.flags] if a else ["solver_crashed" if e.report is None else "audit_failed"]
            rows.append({
                "iteration": e.iteration, "branch": branch, "idea_id": a.idea_id if a else e.idea_id,
                "cluster_action": e.cluster_action.value, "idea_action": e.idea_action.value,
                "score": a.score if a and a.trusted else (0.0 if e.report is None else None),
                "audit_flags": flags,
            })
    return rows


async def select_ideas(state: ResearchState, candidates: list[Idea], branches: int, iteration: int, planned_iterations: int) -> SelectionResult:
    clusters, ranking = state.clusters, state.ranking
    scores = state.best_scores()
    rank_of = {r.idea_id: r.rank for group in ranking.ideas.values() for r in group.idea_ranks}
    members: dict[str, list[Idea]] = {}
    for idea in sorted(candidates, key=lambda i: rank_of.get(i.id, len(rank_of) + 1)):
        cluster = clusters.cluster_of(idea.id)
        if cluster:
            members.setdefault(cluster.id, []).append(idea)
    branches = min(branches, sum(len(v) for v in members.values()))

    def summary(c: Cluster) -> dict:
        evaluated = [scores[m.idea_id] for m in c.members if m.idea_id in scores]
        return {
            "cluster_id": c.id, "name": c.name, "description": c.description,
            "cluster_rank": ranking.cluster_rank(c.id), "n_ideas": len(c.members), "n_evaluated": len(evaluated),
            "n_available": len(members.get(c.id, [])),
            "score_min": min(evaluated, default=None), "score_max": max(evaluated, default=None),
            "score_mean": round(mean(evaluated), 2) if evaluated else None,
            "top_untested": [
                {"idea_id": i.id, "title": i.title, "within_cluster_rank": ranking.idea_rank(i.id)} for i in members.get(c.id, [])[:3]
            ],
        }

    previous = state.iterations[-2].selection if len(state.iterations) > 1 else None
    payload = {
        "problem": state.problem,
        "iteration": iteration,
        "planned_iterations": planned_iterations,
        "branches": branches,
        "remaining_experiment_budget": state.remaining_budget,
        "previous_allocation": previous,
        "score_history": score_history(state),
        "clusters": [summary(c) for c in clusters.clusters],
    }
    available = {c: len(v) for c, v in members.items()}
    plan = await ask("AcquisitionAgent[dispatch]", STAGE1_PROMPT, payload, DispatchPlan, lambda p: check_dispatch(p, branches, available))
    result = SelectionResult(plan=plan)
    if not plan.continue_research:
        return result

    by_id = {c.id: c for c in clusters.clusters}
    picked: list[str] = []
    for a in sorted(plan.branch_action_assignments, key=lambda a: a.branch):
        idea_action = SearchMode.EXPLOITATION if iteration == 1 else a.idea_action
        options = [i for i in members[a.cluster_id] if i.id not in picked]
        if len(options) == 1:
            pick = IdeaPick(idea_id=options[0].id, rationale="Only available untested idea in the assigned cluster.")
        else:
            cluster = by_id[a.cluster_id]
            option_ids = {i.id for i in options}
            stage2 = {
                "problem": state.problem,
                "branch": {"branch": a.branch, "cluster_id": a.cluster_id, "cluster_action": a.cluster_action.value, "idea_action": idea_action.value},
                "cluster": {"name": cluster.name, "description": cluster.description, "cluster_rank": ranking.cluster_rank(cluster.id)},
                "evaluated_members": [
                    {**state.ideas[m.idea_id].model_dump(mode="json", include={"id", "title", "description", "iteration"}), "best_score": scores[m.idea_id]}
                    for m in cluster.members if m.idea_id in scores
                ],
                "available_members": [
                    {
                        **i.model_dump(mode="json", include={"id", "title", "description", "origin", "parent_ids"}),
                        "within_cluster_rank": ranking.idea_rank(i.id),
                        "parent_scores": {p: scores.get(p) for p in i.parent_ids},
                        "previous_attempts": state.attempts(i.id),
                    }
                    for i in options
                ],
                "already_picked_by_other_branches": sorted(picked),
            }

            def check(p: IdeaPick, allowed: set[str] = option_ids) -> None:
                if p.idea_id not in allowed:
                    raise ValueError(f"idea_id must be one of the available members {sorted(allowed)}.")

            prompt = STAGE2_PROMPT + (ITERATION1_SUFFIX if iteration == 1 else "")
            pick = await ask(f"AcquisitionAgent[b{a.branch}]", prompt, stage2, IdeaPick, check)
        picked.append(pick.idea_id)
        result.selected.append(SelectedIdea(
            branch=a.branch, cluster_id=a.cluster_id, cluster_action=a.cluster_action,
            idea_action=idea_action, idea_id=pick.idea_id, rationale=pick.rationale,
        ))
    return result
