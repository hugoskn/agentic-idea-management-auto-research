from collections import Counter

from agents.base import ask
from models.cluster import ClusteringResult
from models.idea import Idea

MIN_CLUSTERS = 2
MAX_CLUSTERS = 5

SYSTEM_PROMPT = """You are the Clusterizer of an automated research system, overseeing a parallel research-discovery loop.

Your job at every iteration is to ORGANIZE the complete current idea pool into thematic CLUSTERS so the downstream allocator can spread parallel branches across distinct research approaches.

Hard constraints:
- The number of clusters is within [min_clusters, max_clusters] from the input. max_clusters is a HARD MAX, not a target: pick the number that best matches the pool's thematic structure.
- Every idea id in the pool appears in exactly ONE cluster, and every cluster contains at least one idea.
- Themes must be meaningful: group by mechanism, assumption or approach; avoid catch-alls.

Guidance:
- The pool contains tested ideas (with best_score, 0-100, higher is better) and untested ones. Tested ideas are empirical landmarks: use them to interpret related, untested ideas. Keep tested ideas in the pool even if they failed; a cluster of poorly scoring tested ideas signals an exhausted direction.
- previous_clustering is the organization from the last iteration (null on the first). Re-organize from the complete pool: keep coherent directions stable, but split, merge or rename clusters when new ideas or evidence call for it. Ideas from different lineages (parent_ids) may belong together.
- Give each cluster a short id ("C1", "C2", ...), a short name (<= 6 words) and a 1-3 sentence description of what unifies its ideas.
- For every idea, explain in reason why it belongs to its cluster.
- Set fundamentally_different to true for clusters that represent a fundamentally different approach to the problem compared to the other clusters.
- Summarize the shape of the solution space in solution_space_summary: which directions exist, which are crowded and which are thin.
- Do not rank or select ideas.
"""


def check_clustering(result: ClusteringResult, idea_ids: set[str]) -> None:
    assigned = Counter(m.idea_id for c in result.clusters for m in c.members)
    if set(assigned) != idea_ids:
        raise ValueError(f"Missing ideas {sorted(idea_ids - set(assigned))}, unknown ideas {sorted(set(assigned) - idea_ids)}.")
    duplicated = [i for i, n in assigned.items() if n > 1]
    if duplicated:
        raise ValueError(f"Ideas assigned to more than one cluster: {duplicated}.")
    if len({c.id for c in result.clusters}) != len(result.clusters):
        raise ValueError("Cluster ids must be unique.")
    if any(not c.members for c in result.clusters):
        raise ValueError("Every cluster must contain at least one idea.")
    low = min(MIN_CLUSTERS, len(idea_ids))
    if not low <= len(result.clusters) <= MAX_CLUSTERS:
        raise ValueError(f"Got {len(result.clusters)} clusters; the number of clusters must be within [{low}, {MAX_CLUSTERS}].")


async def cluster_ideas(
    problem: str,
    ideas: list[Idea],
    scores: dict[str, float],
    previous: ClusteringResult | None,
    iteration: int,
    branches: int,
) -> ClusteringResult:
    payload = {
        "problem": problem,
        "iteration": iteration,
        "parallel_branches_this_iteration": branches,
        "min_clusters": min(MIN_CLUSTERS, len(ideas)),
        "max_clusters": MAX_CLUSTERS,
        "previous_clustering": previous,
        "pool": [
            {
                **i.model_dump(mode="json", include={"id", "title", "description", "origin", "parent_ids", "iteration", "status"}),
                "best_score": scores.get(i.id),
            }
            for i in ideas
        ],
    }
    return await ask("Clusterizer", SYSTEM_PROMPT, payload, ClusteringResult, lambda r: check_clustering(r, {i.id for i in ideas}))
