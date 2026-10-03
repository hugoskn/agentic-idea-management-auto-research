from agents.base import ask
from models.cluster import ClusteringResult
from models.experiment import SelectionResult
from models.idea import Idea
from models.ranking import IdeaRanking
from models.research_state import ResourcePlan

SYSTEM_PROMPT = """You are the AcquisitionAgent of an automated research system.

Your job is to decide which candidate ideas are actually executed in the next round of experiments. Experiments are expensive: every selected idea consumes one unit of the experiment budget.

Do NOT simply select the highest-scoring ideas. Balance:
- Exploitation: investigating highly promising ideas further (high scores, refinements of ideas with good verified results).
- Exploration: testing solution directions (clusters) that have not been sufficiently explored yet, even if their scores are somewhat lower.

Example: if the three best ideas belong to the same cluster, consider whether testing a promising idea from a different cluster gives more information about the solution space than a third idea from the same direction.

The resource_plan gives a target number of exploration and exploitation slots computed from the evidence so far. Follow it unless you have a strong, explicitly justified reason to deviate. Never select more ideas than resource_plan.slots.

Only select from candidate_ideas. For each selected idea give idea_id, priority (1 = highest), selection_reason and exploration_or_exploitation.
Explain the overall decision in strategy_rationale, including how clusters, scores, previous results and remaining budget influenced it.

Set continue_research to false (with stop_reason) only if further experimentation is unlikely to provide meaningful value, for example when the problem is already solved by verified evidence or all remaining candidates are clearly worse than what was already verified.
"""


async def select_ideas(
    problem: str,
    candidates: list[Idea],
    clusters: ClusteringResult,
    rankings: list[IdeaRanking],
    evidence: list[dict],
    plan: ResourcePlan,
    remaining_budget: int,
) -> SelectionResult:
    candidate_ids = {i.id for i in candidates}

    def check(result: SelectionResult) -> None:
        chosen = [s.idea_id for s in result.selected]
        if set(chosen) - candidate_ids:
            raise ValueError(f"Selected ids {sorted(set(chosen) - candidate_ids)} are not candidates.")
        if len(chosen) != len(set(chosen)):
            raise ValueError("An idea was selected more than once.")
        if len(chosen) > plan.slots:
            raise ValueError(f"Selected {len(chosen)} ideas but only {plan.slots} slots are available.")
        if result.continue_research and not chosen:
            raise ValueError("continue_research is true but no idea was selected.")

    payload = {
        "problem": problem,
        "candidate_ideas": candidates,
        "clusters": clusters,
        "rankings": rankings,
        "previous_experiment_results": evidence,
        "remaining_experiment_budget": remaining_budget,
        "resource_plan": plan,
    }
    return await ask("AcquisitionAgent", SYSTEM_PROMPT, payload, SelectionResult, check)
