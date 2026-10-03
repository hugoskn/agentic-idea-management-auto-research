from agents.base import ask
from models.research_state import ResearchConfig, ResearchState, ResourcePlan

SYSTEM_PROMPT = """You are the dynamic Resource Planner of a parallel research-search loop.

The total number of Solver branches (total_runs) is FIXED and every branch has the same compute budget. Iteration 1 was pinned to initial_branches for a deterministic bootstrap. Your job: distribute the REMAINING branches across as many additional iterations as you think optimal. Wider iterations evaluate more ideas concurrently; narrower iterations allow more frequent updates from experimental feedback.

Length is your primary lever. Choose it from the evidence, not as a reflex.

Hard constraints (validated):
- plan starts with frozen_prefix verbatim; you cannot revise it.
- plan has at least one iteration after the frozen prefix.
- len(plan) <= max_iterations.
- Every entry after the frozen prefix is between 1 and max_branches_per_iteration.
- The entries after the frozen prefix sum to EXACTLY branches_to_spend.
- Prefer at least min_branches_per_iteration branches per iteration.

Preferred shapes (choose from the evidence):
- Long tail of 3-branch iterations: refinement over parallelism.
- Iterate-and-refine (several iterations of 3-5 branches): balanced.
- Broad-then-deep: heavy early, tapering to polish.
- Concentrate-and-terminate (2-3 large iterations).

Output plan (branches per iteration for the whole run, frozen prefix included) and rationale (2-4 sentences on why this length and this distribution).
"""


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


def check_plan(result: ResourcePlan, frozen: list[int], to_spend: int, config: ResearchConfig) -> None:
    new = result.plan[len(frozen):]
    if result.plan[: len(frozen)] != frozen:
        raise ValueError(f"plan must start with the frozen prefix {frozen}.")
    if not new:
        raise ValueError("plan must add at least one iteration after the frozen prefix.")
    if len(result.plan) > config.max_iterations:
        raise ValueError(f"plan has {len(result.plan)} iterations; max_iterations is {config.max_iterations}.")
    if any(not 1 <= b <= config.max_branches for b in new):
        raise ValueError(f"Every new iteration needs between 1 and {config.max_branches} branches, got {new}.")
    if sum(new) != to_spend:
        raise ValueError(f"New iterations must spend exactly {to_spend} branches, got {sum(new)}.")


async def plan_resources(state: ResearchState, config: ResearchConfig) -> ResourcePlan:
    frozen = [len(r.experiment_ids) for r in state.iterations[:-1]]
    if not frozen:
        branches = min(config.initial_branches, state.remaining_budget)
        return ResourcePlan(plan=[branches], rationale=f"Iteration 1 is pinned to {branches} branches for an initial breadth of exploration.")
    to_spend = min(state.remaining_budget, (config.max_iterations - len(frozen)) * config.max_branches)
    scores = state.best_scores()
    clusters = state.clusters.clusters if state.clusters else []
    payload = {
        "problem": state.problem,
        "position": {
            "iteration": len(frozen) + 1,
            "total_runs": config.experiment_budget,
            "initial_branches": config.initial_branches,
            "min_branches_per_iteration": config.min_branches,
            "max_branches_per_iteration": config.max_branches,
            "max_iterations": config.max_iterations,
            "frozen_prefix": frozen,
            "branches_to_spend": to_spend,
        },
        "evidence": {
            "best_so_far": max(scores.values(), default=None),
            "target": config.success_score,
            "evaluated_scores": [a.score for a in state.trusted_audits()],
            "estimated_scores": [
                state.rankings[i.id].score for i in state.eligible_ideas(config.max_attempts_per_idea) if i.id in state.rankings
            ],
        },
        "cluster_summary": [
            {
                "name": c.name,
                "description": c.description,
                "n_ideas": len(c.members),
                "n_evaluated": sum(m.idea_id in scores for m in c.members),
                "best_evaluated_score": max((scores[m.idea_id] for m in c.members if m.idea_id in scores), default=None),
            }
            for c in clusters
        ],
        "lessons": state.trusted_lessons(),
    }
    return await ask("ResourcePlanner", SYSTEM_PROMPT, payload, ResourcePlan, lambda r: check_plan(r, frozen, to_spend, config))
