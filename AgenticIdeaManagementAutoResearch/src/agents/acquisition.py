from agents.base import ask
from models.cluster import ClusteringResult
from models.experiment import SelectionResult
from models.idea import Idea
from models.ranking import IdeaRanking

SYSTEM_PROMPT = """You are the AcquisitionAgent of an automated research system.

Your job is to decide which candidate ideas are actually executed in the next round of experiments. Experiments are expensive: every selected idea consumes one unit of the experiment budget.

Do NOT simply select the highest-scoring ideas. Balance:
- Exploitation: investigating highly promising ideas further (high scores, refinements of ideas with good verified results).
- Exploration: testing solution directions (clusters) that have not been sufficiently explored yet, even if their scores are somewhat lower.

Example: if the three best ideas belong to the same cluster, consider whether testing a promising idea from a different cluster gives more information about the solution space than a third idea from the same direction.

The Resource Planner set branches: the number of parallel Solver branches this iteration. Select exactly that many ideas, or every candidate if there are fewer. Decide the exploration/exploitation balance yourself from the clusters, scores and verified evidence: avoid both pure exploitation and pure exploration.

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
    branches: int,
    remaining_budget: int,
) -> SelectionResult:
    candidate_ids = {i.id for i in candidates}
    expected = min(branches, len(candidates))

    def check(result: SelectionResult) -> None:
        chosen = [s.idea_id for s in result.selected]
        if set(chosen) - candidate_ids:
            raise ValueError(f"Selected ids {sorted(set(chosen) - candidate_ids)} are not candidates.")
        if len(chosen) != len(set(chosen)):
            raise ValueError("An idea was selected more than once.")
        if result.continue_research and len(chosen) != expected:
            raise ValueError(f"Selected {len(chosen)} ideas but exactly {expected} branches must be filled.")

    payload = {
        "problem": problem,
        "candidate_ideas": candidates,
        "clusters": clusters,
        "rankings": rankings,
        "previous_experiment_results": evidence,
        "remaining_experiment_budget": remaining_budget,
        "branches": branches,
    }
    return await ask("AcquisitionAgent", SYSTEM_PROMPT, payload, SelectionResult, check)
