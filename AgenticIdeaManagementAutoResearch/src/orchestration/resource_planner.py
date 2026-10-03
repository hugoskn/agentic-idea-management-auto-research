from models.research_state import ResearchState, ResourcePlan


def stagnation(state: ResearchState) -> int:
    trusted = state.trusted_audits()
    history = [
        max((a.score for a in trusted if a.experiment_id in record.experiment_ids), default=0)
        for record in state.iterations
        if record.experiment_ids
    ]
    count = 0
    while len(history) > 1 and history[-1] <= max(history[:-1]):
        count += 1
        history.pop()
    return count


def cluster_coverage(state: ResearchState) -> float:
    if not state.clusters or not state.clusters.clusters:
        return 0.0
    tested = {a.idea_id for a in state.trusted_audits()}
    explored = [c for c in state.clusters.clusters if any(m.idea_id in tested for m in c.members)]
    return len(explored) / len(state.clusters.clusters)


def plan_resources(state: ResearchState, max_parallel: int) -> ResourcePlan:
    slots = max(0, min(max_parallel, state.remaining_budget))
    top = state.best_audit()
    best = (top.score if top else 0) / 100
    coverage = cluster_coverage(state)
    stuck = stagnation(state)
    # ponytail: linear heuristic over coverage, best verified score and stagnation; upgrade to a UCB/Thompson bandit over clusters if selections oscillate.
    ratio = min(0.9, max(0.1, 0.5 * (1 - coverage) + 0.5 * (1 - best) + 0.15 * stuck))
    exploration = round(slots * ratio)
    if slots >= 2:
        exploration = min(slots - 1, max(1, exploration))
    return ResourcePlan(
        slots=slots,
        exploration_slots=exploration,
        exploitation_slots=slots - exploration,
        exploration_ratio=round(ratio, 2),
        rationale=(
            f"{coverage:.0%} of clusters have verified evidence, best verified score is {best:.0%}, "
            f"{stuck} iteration(s) without improvement; exploration ratio {ratio:.2f}."
        ),
    )
