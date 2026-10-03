from agents.base import ask
from models.cluster import ClusteringResult
from models.idea import Idea
from models.ranking import RankingResult

SYSTEM_PROMPT = """You are the Ranker of an automated research system.

Your job is to evaluate every idea and give it a score from 0 to 100: the estimated likelihood that the idea will solve the specific problem, or make a significant contribution toward solving it.

The score is the idea's expected value for THIS problem, not whether it is easy, elegant or interesting.

Consider:
- Expected impact on the problem.
- Technical feasibility.
- Relevance to the problem.
- Evidence from previous audited experiments: ideas that were tested must be re-scored from their verified results, and ideas related to them (same cluster, refinements, combinations) must be updated using that evidence.
- Complexity and risk.
- Potential for producing a breakthrough.
- Whether the idea represents a meaningfully different solution direction.

For every idea provide score (integer 0-100), reasoning (detailed justification), expected_impact, confidence (low/medium/high), key_assumptions and main_risks.

Calibration:
- Use the full scale. Do not give all ideas similar scores: differences in expected value must be visible in the scores.
- Explain the overall score distribution in calibration_notes.
- Rank every idea exactly once.
"""

MIN_SPREAD = 10


async def rank_ideas(
    problem: str,
    ideas: list[Idea],
    clusters: ClusteringResult,
    evidence: list[dict],
    lessons: list[str],
) -> RankingResult:
    idea_ids = [i.id for i in ideas]

    def check(result: RankingResult) -> None:
        ranked = [r.idea_id for r in result.rankings]
        if sorted(ranked) != sorted(idea_ids):
            raise ValueError(f"Every idea must be ranked exactly once. Expected {sorted(idea_ids)}, got {sorted(ranked)}.")
        scores = [r.score for r in result.rankings]
        if len(scores) >= 4 and max(scores) - min(scores) < MIN_SPREAD:
            raise ValueError(f"Scores span only {min(scores)}-{max(scores)}; differentiate the ideas by expected value.")

    payload = {
        "problem": problem,
        "ideas": ideas,
        "clusters": clusters,
        "evidence_from_audited_experiments": evidence,
        "lessons": lessons,
    }
    return await ask("Ranker", SYSTEM_PROMPT, payload, RankingResult, check)
