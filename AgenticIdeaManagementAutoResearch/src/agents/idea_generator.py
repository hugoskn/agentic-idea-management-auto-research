import logging
from typing import Any

from agents.base import ask
from models.idea import ORIGIN_OF_KIND, ExpandKind, Expansion, Idea, IdeaBatch, IdeaContent, IdeaDraft

log = logging.getLogger("aim")

SYSTEM_PROMPT = """You are the Researcher, the idea generator of an automated research system.

Your only job is to propose the initial research ideas for solving a problem. You do not rank, select or implement them.

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
"""

EXPAND_PROMPT = """You are the Expander of a parallel research-search loop. Each iteration, multiple branches independently try ideas; your job is to take ONE branch's outcome plus the full swarm context and propose new ideas that should join the idea pool.

You see: this branch's parent idea and its outcome; the sibling branches' outcomes this iteration; cluster context; the top-3 and bottom-3 results of the whole run; verified lessons (each prefixed with the [idea_id] it came from); the titles already in the pool.

You have up to max_proposals PROPOSAL SLOTS. Emit AT MOST ONE proposal per kind; skip any kind that doesn't apply. Quality over quota: an empty list is acceptable.

Kinds:
- fix_error: the branch failed (outcome.success is false). Propose a targeted fix of its idea using the failure detail. Not allowed when the branch succeeded.
- push_score: the branch succeeded with headroom. Preserve the validated components of the idea and address its remaining bottleneck with a tighter implementation. Not allowed when the branch failed.
- cross_pollinate: a specific sibling outcome OR lesson gives a compositional improvement. Base the reasoning ONLY on lessons and sibling_outcomes, and cite exactly ONE source: sibling_referenced = an idea_id from sibling_outcomes, OR lesson_context_idea = the idea_id of a lesson. Both empty, both set, or an unknown id: the proposal is dropped.
- new_idea: no in-cluster improvement to offer; propose an ORTHOGONAL direction, without a parent. Workflow: (1) read cluster_context to see which directions already exist, (2) name 1-2 orthogonal directions in the description, (3) instantiate one. If the best you can offer is a variant of an existing direction, use cross_pollinate or push_score instead.

Every proposal is a complete idea, concrete enough for a Solver to implement: title, description, how_it_addresses_problem, rationale, assumptions, risks. Never repeat the parent idea verbatim or re-propose an idea already in the pool.
"""


async def generate_ideas(problem: str, count: int) -> IdeaBatch:
    def check(batch: IdeaBatch) -> None:
        if len(batch.ideas) != count:
            raise ValueError(f"Expected exactly {count} ideas, got {len(batch.ideas)}.")

    payload = {"problem": problem, "ideas_requested": count}
    return await ask("IdeaGenerator", SYSTEM_PROMPT, payload, IdeaBatch, check)


def check_expansion(result: Expansion, slots: int, success: bool) -> None:
    kinds = [p.kind for p in result.proposals]
    if len(kinds) > slots:
        raise ValueError(f"Got {len(kinds)} proposals but only {slots} slots are available.")
    if len(kinds) != len(set(kinds)):
        raise ValueError("Emit at most one proposal per kind.")
    if success and ExpandKind.FIX_ERROR in kinds:
        raise ValueError("fix_error is only allowed when the branch failed.")
    if not success and ExpandKind.PUSH_SCORE in kinds:
        raise ValueError("push_score is only allowed when the branch succeeded.")


def to_drafts(result: Expansion, parent_id: str, sibling_ids: set[str], lesson_ids: set[str]) -> list[IdeaDraft]:
    drafts = []
    for p in result.proposals:
        parents = [] if p.kind == ExpandKind.NEW_IDEA else [parent_id]
        if p.kind == ExpandKind.CROSS_POLLINATE:
            sibling, lesson = p.sibling_referenced.strip(), p.lesson_context_idea.strip()
            if bool(sibling) == bool(lesson) or (sibling and sibling not in sibling_ids) or (lesson and lesson not in lesson_ids):
                log.info("Dropped cross_pollinate proposal '%s' from %s: it must cite exactly one known sibling or lesson.", p.title, parent_id)
                continue
            if sibling and sibling != parent_id:
                parents.append(sibling)
        content = p.model_dump(include=set(IdeaContent.model_fields))
        drafts.append(IdeaDraft(**content, origin=ORIGIN_OF_KIND[p.kind], parent_ids=parents))
    return drafts


async def expand_branch(
    problem: str,
    parent: Idea,
    outcome: dict[str, Any],
    siblings: list[dict[str, Any]],
    context: dict[str, Any],
    lessons: list[str],
    lesson_ids: set[str],
    slots: int,
) -> list[IdeaDraft]:
    payload = {
        "problem": problem,
        "max_proposals": slots,
        "parent_idea": parent,
        "outcome": outcome,
        "sibling_outcomes": siblings,
        **context,
        "lessons": lessons,
    }
    result = await ask(
        f"Expander[{parent.id}]", EXPAND_PROMPT, payload, Expansion,
        lambda r: check_expansion(r, slots, outcome["success"]),
    )
    return to_drafts(result, parent.id, {s["idea_id"] for s in siblings}, lesson_ids)
