import asyncio
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from agents.acquisition import select_ideas
from agents.auditor import audit_experiment
from agents.base import AgentError
from agents.clusterizer import cluster_ideas
from agents.idea_generator import generate_ideas
from agents.ranker import rank_ideas
from agents.solver import run_solver
from models.audit import AuditResult
from models.cluster import Cluster
from models.experiment import Experiment
from models.idea import Idea, IdeaStatus
from models.research_state import IterationRecord, Lesson, ResearchState
from orchestration.resource_planner import plan_resources, stagnation

log = logging.getLogger("aim")


class ResearchConfig(BaseModel):
    ideas_count: int = Field(default=6, ge=1)
    new_ideas_per_iteration: int = Field(default=3, ge=1)
    max_iterations: int = Field(default=3, ge=1)
    experiment_budget: int = Field(default=6, ge=1)
    max_parallel: int = Field(default=3, ge=1)
    success_score: int = Field(default=85, ge=0, le=100)
    discard_score: int = Field(default=25, ge=0, le=100)
    patience: int = Field(default=2, ge=1)
    max_attempts_per_idea: int = Field(default=2, ge=1)
    max_refinements: int = Field(default=2, ge=0)


async def run_research(problem: str, config: ResearchConfig, out_dir: Path) -> ResearchState:
    state = ResearchState(problem=problem, remaining_budget=config.experiment_budget)
    state_path = out_dir / "research_state.json"
    try:
        for iteration in range(1, config.max_iterations + 1):
            log.info("=== Iteration %d (experiment budget left: %d) ===", iteration, state.remaining_budget)
            record = IterationRecord(iteration=iteration)
            state.iterations.append(record)
            await expand_ideas(state, record, config)
            await organize_and_rank(state, record)
            state.save(state_path)

            candidates = state.eligible_ideas(config.max_attempts_per_idea)
            if not candidates:
                state.stop_reason = "No untested ideas left to execute."
                break
            record.plan = plan_resources(state, config.max_parallel)
            log.info("Resource plan: %s", record.plan.rationale)
            record.selection = await select_ideas(
                state.problem, candidates, state.clusters, list(state.rankings.values()),
                state.evidence(), record.plan, state.remaining_budget,
            )
            if not record.selection.continue_research:
                state.stop_reason = f"AcquisitionAgent stopped the research: {record.selection.stop_reason}"
                break

            await execute(state, record, config, out_dir)
            state.save(state_path)
            state.stop_reason = check_stop(state, config)
            if state.stop_reason:
                break
        else:
            state.stop_reason = f"Maximum number of iterations ({config.max_iterations}) reached."
    except AgentError as e:
        log.error("Research aborted: %s", e)
        state.stop_reason = f"Aborted: {e}"
    log.info("Stopped: %s", state.stop_reason)
    state.final_recommendation = recommend(state)
    state.save(state_path)
    return state


async def expand_ideas(state: ResearchState, record: IterationRecord, config: ResearchConfig) -> None:
    if record.iteration == 1:
        batch = await generate_ideas(state.problem, config.ideas_count)
    else:
        batch = await generate_ideas(
            state.problem, config.new_ideas_per_iteration, list(state.ideas.values()),
            state.clusters, state.evidence(), state.trusted_lessons(),
        )
    added = state.add_ideas(batch.ideas, record.iteration)
    record.new_idea_ids = [i.id for i in added]
    for idea in added:
        for parent_id in idea.parent_ids:
            if state.ideas[parent_id].status == IdeaStatus.EVIDENCE_COLLECTED:
                state.ideas[parent_id].status = IdeaStatus.REFINED
    log.info("Generated ideas: %s", ", ".join(f"{i.id} {i.title} [{i.origin.value}]" for i in added))


async def organize_and_rank(state: ResearchState, record: IterationRecord) -> None:
    active = state.active_ideas()
    state.clusters = record.clustering = await cluster_ideas(state.problem, active)
    for idea in active:
        if idea.status == IdeaStatus.GENERATED:
            idea.status = IdeaStatus.CLUSTERED
    log.info("Clusters: %s", "; ".join(f"{c.name}: {[m.idea_id for m in c.members]}" for c in state.clusters.clusters))

    record.ranking = await rank_ideas(state.problem, active, state.clusters, state.evidence(), state.trusted_lessons())
    state.rankings = {r.idea_id: r for r in record.ranking.rankings}
    for idea in active:
        if idea.status == IdeaStatus.CLUSTERED:
            idea.status = IdeaStatus.RANKED
    log.info("Scores: %s", ", ".join(f"{r.idea_id}={r.score}" for r in sorted(record.ranking.rankings, key=lambda r: -r.score)))


async def execute(state: ResearchState, record: IterationRecord, config: ResearchConfig, out_dir: Path) -> None:
    selected = sorted(record.selection.selected, key=lambda s: s.priority)[: record.plan.slots]
    experiments, jobs = [], []
    for choice in selected:
        idea = state.ideas[choice.idea_id]
        idea.status = IdeaStatus.SELECTED
        experiment_id = f"E{len(state.experiments) + 1}"
        workdir = out_dir / "experiments" / f"{experiment_id}_{idea.id}"
        workdir.mkdir(parents=True, exist_ok=True)
        experiment = Experiment(
            id=experiment_id, idea_id=idea.id, iteration=record.iteration,
            mode=choice.exploration_or_exploitation, workdir=str(workdir),
        )
        state.experiments.append(experiment)
        record.experiment_ids.append(experiment_id)
        experiments.append(experiment)
        jobs.append(run_experiment(state.problem, idea, state.clusters.cluster_of(idea.id), state.trusted_lessons(), experiment, config))
        log.info("%s: %s %s [%s] - %s", experiment_id, idea.id, idea.title, choice.exploration_or_exploitation.value, choice.selection_reason)
    state.remaining_budget -= len(jobs)
    audits = await asyncio.gather(*jobs)
    for experiment, audit in zip(experiments, audits):
        learn(state, experiment, audit, config)


async def run_experiment(
    problem: str,
    idea: Idea,
    cluster: Cluster | None,
    lessons: list[str],
    experiment: Experiment,
    config: ResearchConfig,
) -> AuditResult | None:
    workdir = Path(experiment.workdir)
    try:
        experiment.report, experiment.refinement_rounds = await run_solver(
            problem, idea, cluster, lessons, workdir, config.max_refinements
        )
    except Exception as e:
        log.error("%s solver failed: %s", experiment.id, e)
        experiment.error = f"Solver failed: {e}"
        return None
    idea.status = IdeaStatus.IMPLEMENTED
    try:
        verdict = await audit_experiment(problem, idea, cluster, experiment.report, workdir)
    except Exception as e:
        log.error("%s audit failed: %s", experiment.id, e)
        experiment.error = f"Audit failed: {e}"
        return None
    idea.status = IdeaStatus.AUDITED
    return AuditResult(**verdict.model_dump(), experiment_id=experiment.id, idea_id=idea.id)


def learn(state: ResearchState, experiment: Experiment, audit: AuditResult | None, config: ResearchConfig) -> None:
    idea = state.ideas[experiment.idea_id]
    retries_left = state.attempts(idea.id) < config.max_attempts_per_idea
    if audit is None:
        idea.status = IdeaStatus.RANKED if retries_left else IdeaStatus.DISCARDED
        return
    state.audit_results.append(audit)
    texts = list(audit.lessons_learned)
    if audit.trusted:
        solved_or_useful = audit.task_solved or audit.score >= config.discard_score
        idea.status = IdeaStatus.EVIDENCE_COLLECTED if solved_or_useful else IdeaStatus.DISCARDED
    else:
        reason = "; ".join(audit.discrepancies) or "results not supported by the experiment"
        texts.insert(0, f"{experiment.id} excluded from evidence: {reason}. Actually implemented: {audit.actually_implemented_idea or 'n/a'}")
        idea.status = IdeaStatus.RANKED if retries_left else IdeaStatus.DISCARDED
    state.lessons += [
        Lesson(iteration=experiment.iteration, experiment_id=experiment.id, idea_id=idea.id, text=t, trusted=audit.trusted)
        for t in texts
    ]
    log.info(
        "%s audit: trusted=%s solved=%s score=%d -> %s is now %s",
        experiment.id, audit.trusted, audit.task_solved, audit.score, idea.id, idea.status.value,
    )


def check_stop(state: ResearchState, config: ResearchConfig) -> str:
    best = state.best_audit()
    if best and best.task_solved and best.score >= config.success_score:
        return f"Sufficiently good solution found: {best.idea_id} (audit score {best.score})."
    if state.remaining_budget <= 0:
        return "Experiment budget exhausted."
    if stagnation(state) >= config.patience:
        return f"No improvement in {config.patience} iterations; further experimentation is unlikely to add value."
    return ""


def recommend(state: ResearchState) -> str:
    experiments = {e.id: e for e in state.experiments}
    audits = sorted(state.trusted_audits(), key=lambda a: (a.task_solved, a.score), reverse=True)
    chosen = [a for a in audits if a.task_solved][:3] or audits[:1]
    if not chosen:
        return "No verified solution was found. See lessons learned for the directions that were ruled out."
    lines = []
    for audit in chosen:
        experiment = experiments[audit.experiment_id]
        verdict = "solves the problem" if audit.task_solved else "best partial result"
        lines.append(
            f"{audit.idea_id} {state.ideas[audit.idea_id].title} (audit score {audit.score}, {verdict}): "
            f"{experiment.report.implementation_summary} Result: {experiment.report.result} Artifacts: {experiment.workdir}"
        )
    return "\n".join(lines)
