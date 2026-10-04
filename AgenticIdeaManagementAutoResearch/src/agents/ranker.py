import asyncio
from statistics import mean

from agents.base import ask
from models.cluster import Cluster, ClusteringResult
from models.idea import Idea
from models.ranking import ClusterRanking, IdeaRank, IdeaRanking, RankingResult

SCORING_SCALE = "0-100, higher is better (the evaluator's score when one is configured, otherwise the auditor's)."

DENSE_RANKS = """## Ranking scheme: DENSE RANKS
Integers from 1 (most promising). Ties allowed (same integer). The next distinct rank after a tie is +1, not skipped.
Valid: [1,2,2,3,4] [1,1,1,1,1] [1,2,3,4,5]. Invalid: [1,2,2,4,5] (gap after tie)."""

CLUSTER_PROMPT = f"""You are the Ranker of an automated research system, ranking research-idea CLUSTERS by promise. Your ranks feed a downstream allocator that decides how many parallel research branches to spend on each cluster.

You see each cluster's name, description, n_ideas, n_evaluated, mean_evaluated_score and best_evaluated_score, plus the verified lessons so far. You are judging the DIRECTION, not individual ideas.

{DENSE_RANKS}

## Guidance
- High best_evaluated_score -> LOW rank (promising).
- Many evaluated, all scored poorly -> HIGH rank (exhausted).
- n_evaluated = 0 -> rank on the quality of the direction against the problem's known bottlenecks and the lessons.
- Use the lessons: promote directions they support, demote directions they warn against.
- Do NOT collapse everything to rank 1.

Rank every cluster exactly once. Explain in rationale (1-3 sentences, cite specific clusters).
"""

IDEA_PROMPT = f"""You are the Ranker of an automated research system, ranking research IDEAS within a single cluster by promise. Your ranks feed the downstream allocator that picks which idea each branch attempts.

You see the untested candidates of ONE cluster. The cluster's name and description give the direction they share.

{DENSE_RANKS}

## Guidance
- Rank by how strongly mechanism, novelty and specificity predict a good score.
- LOW rank for ideas that clearly instantiate the cluster's theme with a concrete, testable approach.
- HIGH rank for ideas that repeat existing themes or lack mechanism specificity.
- Do NOT collapse everything to rank 1 unless truly indistinguishable.

Rank every idea exactly once. Explain in rationale (1-3 sentences).
"""


def check_dense(ranks: list[tuple[str, int]], expected: set[str], what: str) -> None:
    ids = [i for i, _ in ranks]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"Rank every {what} exactly once. Expected {sorted(expected)}, got {sorted(ids)}.")
    distinct = sorted({r for _, r in ranks})
    if distinct != list(range(1, len(distinct) + 1)):
        raise ValueError(f"Ranks {distinct} are not dense: start at 1 and do not skip a rank after a tie.")


def cluster_stats(cluster: Cluster, scores: dict[str, float]) -> dict:
    evaluated = [scores[m.idea_id] for m in cluster.members if m.idea_id in scores]
    return {
        "cluster_id": cluster.id,
        "name": cluster.name,
        "description": cluster.description,
        "n_ideas": len(cluster.members),
        "n_evaluated": len(evaluated),
        "mean_evaluated_score": round(mean(evaluated), 2) if evaluated else None,
        "best_evaluated_score": max(evaluated, default=None),
    }


async def rank_clusters(problem: str, clusters: ClusteringResult, scores: dict[str, float], lessons: list[str]) -> ClusterRanking:
    payload = {
        "problem": problem,
        "scoring_scale": SCORING_SCALE,
        "clusters": [cluster_stats(c, scores) for c in clusters.clusters],
        "lessons": lessons,
    }
    expected = {c.id for c in clusters.clusters}
    return await ask(
        "Ranker[clusters]", CLUSTER_PROMPT, payload, ClusterRanking,
        lambda r: check_dense([(c.cluster_id, c.rank) for c in r.cluster_ranks], expected, "cluster"),
    )


async def rank_ideas_in_cluster(problem: str, cluster: Cluster, cluster_rank: int, candidates: list[Idea]) -> IdeaRanking:
    if len(candidates) == 1:
        return IdeaRanking(idea_ranks=[IdeaRank(idea_id=candidates[0].id, rank=1)], rationale="Only untested idea in the cluster.")
    payload = {
        "problem": problem,
        "scoring_scale": SCORING_SCALE,
        "cluster": {"name": cluster.name, "description": cluster.description, "cluster_rank": cluster_rank},
        "ideas_to_rank": [
            i.model_dump(mode="json", include={"id", "title", "description", "how_it_addresses_problem", "origin", "parent_ids"})
            for i in candidates
        ],
    }
    expected = {i.id for i in candidates}
    return await ask(
        f"Ranker[{cluster.id}]", IDEA_PROMPT, payload, IdeaRanking,
        lambda r: check_dense([(i.idea_id, i.rank) for i in r.idea_ranks], expected, "idea"),
    )


async def estimate(
    problem: str,
    clusters: ClusteringResult,
    candidates: list[Idea],
    scores: dict[str, float],
    lessons: list[str],
) -> RankingResult:
    cluster_ranking = await rank_clusters(problem, clusters, scores, lessons)
    result = RankingResult(clusters=cluster_ranking)
    by_id = {i.id: i for i in candidates}
    jobs = {}
    for cluster in clusters.clusters:
        members = [by_id[m.idea_id] for m in cluster.members if m.idea_id in by_id]
        if members:
            jobs[cluster.id] = rank_ideas_in_cluster(problem, cluster, result.cluster_rank(cluster.id), members)
    result.ideas = dict(zip(jobs, await asyncio.gather(*jobs.values())))
    return result
