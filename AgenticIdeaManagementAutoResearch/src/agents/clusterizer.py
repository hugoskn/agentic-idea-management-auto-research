from collections import Counter

from agents.base import ask
from models.cluster import ClusteringResult
from models.idea import Idea

SYSTEM_PROMPT = """You are the Clusterizer of an automated research system.

Your job is to make the solution space explicit by organizing research ideas into semantic clusters.

Rules:
- Analyze all ideas and identify the major solution directions they represent.
- Group semantically similar ideas together. Every idea belongs to exactly one cluster.
- Avoid unnecessary clusters: do not create one cluster per idea unless the ideas are truly unrelated, and do not merge fundamentally different approaches.
- Give each cluster a short id ("C1", "C2", ...), a clear name and a description of the solution direction.
- For every idea, explain in reason why it belongs to its cluster.
- Set fundamentally_different to true for clusters that represent a fundamentally different approach to the problem compared to the other clusters.
- Summarize the shape of the solution space in solution_space_summary: which directions exist, which are crowded and which are thin.
- Do not judge, rank or select ideas.
"""


async def cluster_ideas(problem: str, ideas: list[Idea]) -> ClusteringResult:
    idea_ids = {i.id for i in ideas}

    def check(result: ClusteringResult) -> None:
        assigned = Counter(m.idea_id for c in result.clusters for m in c.members)
        if set(assigned) != idea_ids:
            raise ValueError(f"Missing ideas {sorted(idea_ids - set(assigned))}, unknown ideas {sorted(set(assigned) - idea_ids)}.")
        duplicated = [i for i, n in assigned.items() if n > 1]
        if duplicated:
            raise ValueError(f"Ideas assigned to more than one cluster: {duplicated}.")
        if len({c.id for c in result.clusters}) != len(result.clusters):
            raise ValueError("Cluster ids must be unique.")

    payload = {"problem": problem, "ideas": ideas}
    return await ask("Clusterizer", SYSTEM_PROMPT, payload, ClusteringResult, check)
