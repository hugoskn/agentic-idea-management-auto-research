from agents.base import ask
from models.cluster import ClusteringResult
from models.idea import Idea, IdeaBatch, IdeaOrigin

SYSTEM_PROMPT = """You are the Researcher, the idea generator of an automated research system.

Your only job is to propose research ideas for solving a problem. You do not rank, select or implement them.

Rules:
- First understand the problem: restate it, its constraints and what "solved" would mean, in problem_understanding.
- Generate EXACTLY the number of ideas requested.
- Ideas must be genuinely different approaches, not minor variations of the same approach. Think broadly across different solution directions and prefer semantic diversity between ideas.
- Each idea must be concrete enough that another agent could implement or test it without asking you questions.
- For every idea explain:
  - description: what the idea is.
  - how_it_addresses_problem: how it would address the problem.
  - rationale: why you believe it has a high probability of solving the problem or significantly contributing to its resolution.
  - assumptions: the assumptions behind the idea.
  - risks: the main risks or reasons it might fail.
- Do not rank the ideas and do not prematurely converge on one solution.

When existing ideas, evidence and lessons are provided (mode "evolve"), propose NEW ideas that learn from them. Each new idea must set origin:
- "refinement": improves an idea whose experiment was promising (parent_ids = that idea).
- "combination": combines successful ideas from different clusters (parent_ids = the combined ideas).
- "fix": addresses the failure cause of an idea that did not work (parent_ids = that idea).
- "new_direction": explores a direction not covered by any existing cluster (parent_ids empty).
Mix these origins according to the evidence: refine and combine what works, fix what failed for fixable reasons, and keep exploring new directions while the solution space is under-explored.
Never re-propose an existing idea. In mode "initial", every idea has origin "initial" and empty parent_ids.
"""


async def generate_ideas(
    problem: str,
    count: int,
    existing: list[Idea] | None = None,
    clusters: ClusteringResult | None = None,
    evidence: list[dict] | None = None,
    lessons: list[str] | None = None,
) -> IdeaBatch:
    existing = existing or []
    known_ids = {i.id for i in existing}

    def check(batch: IdeaBatch) -> None:
        if len(batch.ideas) != count:
            raise ValueError(f"Expected exactly {count} ideas, got {len(batch.ideas)}.")
        for idea in batch.ideas:
            unknown = set(idea.parent_ids) - known_ids
            if unknown:
                raise ValueError(f"Idea '{idea.title}' references unknown parent ids {sorted(unknown)}.")
            if not existing and (idea.origin != IdeaOrigin.INITIAL or idea.parent_ids):
                raise ValueError("In initial mode every idea must have origin 'initial' and no parent_ids.")
            if existing and idea.origin in (IdeaOrigin.INITIAL, IdeaOrigin.RECONSTRUCTED):
                raise ValueError(f"Idea '{idea.title}' must use an evolve origin, not '{idea.origin.value}'.")

    payload = {
        "mode": "evolve" if existing else "initial",
        "problem": problem,
        "ideas_requested": count,
        "existing_ideas": existing,
        "clusters": clusters,
        "evidence_from_audited_experiments": evidence or [],
        "lessons": lessons or [],
    }
    return await ask("IdeaGenerator", SYSTEM_PROMPT, payload, IdeaBatch, check)
