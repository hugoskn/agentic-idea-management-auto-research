import asyncio
import logging
import math
from pathlib import Path

from agents.acquisition import select_ideas
from agents.auditor import audit_experiment
from agents.base import AgentError
from agents.clusterizer import cluster_ideas
from agents.idea_generator import expand_branch, generate_ideas
from agents.ranker import cluster_stats, estimate
from agents.solver import run_solver
from models.audit import AuditFlag, AuditResult
from models.cluster import Cluster
from models.experiment import Evaluation, Experiment
from models.idea import Idea, IdeaContent, IdeaDraft, IdeaOrigin, IdeaStatus
from models.research_state import IterationRecord, Lesson, ResearchConfig, ResearchState
from orchestration.resource_planner import plan_resources, stagnation

log = logging.getLogger("aim")


async def run_research(problem: str, config: ResearchConfig, out_dir: Path) -> ResearchState:
    state = ResearchState(problem=problem, remaining_budget=config.experiment_budget)
    state_path = out_dir / "research_state.json"
    try:
        for iteration in range(1, config.max_iterations + 1):
            log.info("=== Iteration %d (experiment budget left: %d) ===", iteration, state.remaining_budget)
            record = IterationRecord(iteration=iteration)
            state.iterations.append(record)
            await expand_ideas(state, record, config)
            record.plan = await plan_resources(state, config)
            branches = record.plan.plan[iteration - 1]
            log.info("Resource plan %s, %d branches now: %s", record.plan.plan, branches, record.plan.rationale)
            await organize_and_estimate(state, record, branches)
            state.save(state_path)

            candidates = state.eligible_ideas(config.max_attempts_per_idea)
            if not candidates:
                state.stop_reason = "No untested ideas left to execute."
                break
            record.selection = await select_ideas(state, candidates, branches, iteration, len(record.plan.plan))
            if not record.selection.plan.continue_research:
                state.stop_reason = f"AcquisitionAgent stopped the research: {record.selection.plan.stop_reason}"
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
        drafts = [IdeaDraft(**c.model_dump()) for c in batch.ideas]
    else:
        drafts = await expand(state, state.iterations[-2], config)
    added = state.add_ideas(drafts, record.iteration)
    record.new_idea_ids = [i.id for i in added]
    for idea in added:
        for parent_id in idea.parent_ids:
            if state.ideas[parent_id].status == IdeaStatus.EVIDENCE_COLLECTED:
                state.ideas[parent_id].status = IdeaStatus.REFINED
    log.info("Generated ideas: %s", ", ".join(f"{i.id} {i.title} [{i.origin.value}]" for i in added))


def branch_outcomes(state: ResearchState, record: IterationRecord) -> list[tuple[Experiment, dict]]:
    audits = {a.experiment_id: a for a in state.audit_results}
    experiments = {e.id: e for e in state.experiments}
    rows = []
    for branch, experiment_id in enumerate(record.experiment_ids):
        experiment = experiments[experiment_id]
        audit = audits.get(experiment_id)
        final = experiment.evaluations[-1] if experiment.evaluations else None
        if audit and audit.trusted:
            idea_id, score = audit.idea_id, audit.score
            success = score > 0 and (final is None or final.valid)
            detail = experiment.report.result + (f" Evaluator output: {final.output[-300:]}" if final else "")
        elif experiment.report is None:
            idea_id, score, success, detail = experiment.idea_id, 0.0, False, experiment.error or "Solver failed."
        else:
            continue
        rows.append((experiment, {
            "branch": branch, "idea_id": idea_id, "title": state.ideas[idea_id].title,
            "score": score, "success": success, "detail": detail,
        }))
    return rows


def run_signals(state: ResearchState) -> dict:
    experiments = {e.id: e for e in state.experiments}
    rows = []
    for a in state.trusted_audits():
        cluster = state.clusters.cluster_of(experiments[a.experiment_id].idea_id) if state.clusters else None
        rows.append({
            "idea_id": a.idea_id, "title": state.ideas[a.idea_id].title, "iteration": experiments[a.experiment_id].iteration,
            "score": a.score, "cluster": cluster.name if cluster else None,
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"top_3": rows[:3], "bottom_3": rows[-3:]}


async def expand(state: ResearchState, previous: IterationRecord, config: ResearchConfig) -> list[IdeaDraft]:
    outcomes = branch_outcomes(state, previous)
    if not outcomes or not config.max_ideas_per_branch:
        return []
    scores = state.best_scores()
    clusters = state.clusters.clusters if state.clusters else []
    signals = run_signals(state)
    pool = [f"{i.id} {i.title}" for i in state.ideas.values()]
    lessons = state.trusted_lessons()
    lesson_ids = {l.idea_id for l in state.lessons if l.trusted}
    jobs = []
    for experiment, outcome in outcomes:
        own = state.clusters.cluster_of(experiment.idea_id) if state.clusters else None
        context = {
            "cluster_context": {
                "parent_cluster": cluster_stats(own, scores) if own else None,
                "other_clusters": [cluster_stats(c, scores) for c in clusters if c is not own],
            },
            "run_signals": signals,
            "pool_titles": pool,
        }
        siblings = [o for _, o in outcomes if o is not outcome]
        jobs.append(expand_branch(
            state.problem, state.ideas[outcome["idea_id"]], outcome, siblings, context, lessons, lesson_ids, config.max_ideas_per_branch,
        ))
    return [draft for drafts in await asyncio.gather(*jobs) for draft in drafts]


async def organize_and_estimate(state: ResearchState, record: IterationRecord, branches: int) -> None:
    pool = list(state.ideas.values())
    scores = state.best_scores()
    state.clusters = record.clustering = await cluster_ideas(state.problem, pool, scores, state.clusters, record.iteration, branches)
    for idea in pool:
        if idea.status == IdeaStatus.GENERATED:
            idea.status = IdeaStatus.CLUSTERED
    log.info("Clusters: %s", "; ".join(f"{c.id} {c.name}: {[m.idea_id for m in c.members]}" for c in state.clusters.clusters))

    candidates = state.untested_ideas()
    state.ranking = record.ranking = await estimate(state.problem, state.clusters, candidates, scores, state.trusted_lessons())
    for idea in candidates:
        idea.status = IdeaStatus.RANKED
    log.info("Cluster ranks: %s", ", ".join(f"{r.cluster_id}=#{r.rank}" for r in sorted(state.ranking.clusters.cluster_ranks, key=lambda r: r.rank)))


async def execute(state: ResearchState, record: IterationRecord, config: ResearchConfig, out_dir: Path) -> None:
    selected = sorted(record.selection.selected, key=lambda s: s.branch)
    experiments, jobs = [], []
    for choice in selected:
        idea = state.ideas[choice.idea_id]
        idea.status = IdeaStatus.SELECTED
        experiment_id = f"E{len(state.experiments) + 1}"
        workdir = out_dir / "experiments" / f"{experiment_id}_{idea.id}"
        workdir.mkdir(parents=True, exist_ok=True)
        experiment = Experiment(
            id=experiment_id, idea_id=idea.id, iteration=record.iteration,
            cluster_action=choice.cluster_action, idea_action=choice.idea_action, workdir=str(workdir),
        )
        state.experiments.append(experiment)
        record.experiment_ids.append(experiment_id)
        experiments.append(experiment)
        jobs.append(run_experiment(state.problem, idea, state.clusters.cluster_of(idea.id), state.trusted_lessons(), experiment, config))
        log.info(
            "%s: branch %d, %s %s in %s [cluster %s, idea %s] - %s", experiment_id, choice.branch, idea.id, idea.title,
            choice.cluster_id, choice.cluster_action.value, choice.idea_action.value, choice.rationale,
        )
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
    scorer = (lambda: evaluate(config.evaluator, workdir, config.evaluator_timeout)) if config.evaluator else None
    try:
        experiment.report, experiment.refinement_rounds, experiment.evaluations = await run_solver(
            problem, idea, cluster, lessons, workdir, config.max_refinements, scorer
        )
    except Exception as e:
        log.error("%s solver failed: %s", experiment.id, e)
        experiment.error = f"Solver failed: {e}"
        return None
    idea.status = IdeaStatus.IMPLEMENTED
    final = experiment.evaluations[-1] if experiment.evaluations else None
    if final:
        log.info("%s evaluator scores: %s", experiment.id, [e.score for e in experiment.evaluations])
    try:
        verdict = await audit_experiment(problem, idea, cluster, experiment.report, final, workdir)
    except Exception as e:
        log.error("%s audit failed: %s", experiment.id, e)
        experiment.error = f"Audit failed: {e}"
        return None
    idea.status = IdeaStatus.AUDITED
    audit = AuditResult(**verdict.model_dump(), experiment_id=experiment.id, idea_id=idea.id)
    if final:
        audit.score = final.score
        audit.task_solved = final.valid and final.score >= config.success_score
    return audit


async def evaluate(command: str, workdir: Path, timeout: int) -> Evaluation:
    proc = await asyncio.create_subprocess_shell(
        command, cwd=workdir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        # ponytail: kill() stops the shell, not its children on Windows; use a job object or process group if evaluators spawn long-lived subprocesses.
        proc.kill()
        await proc.wait()
        return Evaluation(valid=False, score=0, output=f"Evaluator timed out after {timeout}s.")
    text = out.decode("utf-8", errors="replace").strip()
    lines = text.splitlines()
    try:
        score = float(lines[-1]) if proc.returncode == 0 and lines else math.nan
    except ValueError:
        score = math.nan
    tail = text[-4000:]
    if not math.isfinite(score):
        return Evaluation(valid=False, score=0, output=f"Exit code {proc.returncode}; the last line is not a score.\n{tail}")
    return Evaluation(valid=True, score=min(100.0, max(0.0, score)), output=tail)


def learn(state: ResearchState, experiment: Experiment, audit: AuditResult | None, config: ResearchConfig) -> None:
    idea = state.ideas[experiment.idea_id]
    retries_left = state.attempts(idea.id) < config.max_attempts_per_idea
    if audit is None:
        crashed = experiment.report is None
        idea.status = IdeaStatus.RANKED if retries_left and not crashed else IdeaStatus.DISCARDED
        return
    texts = list(audit.lessons_learned)
    if audit.trusted and AuditFlag.IDEA_MISMATCH in audit.flags:
        content = audit.reconstructed_idea.model_dump(include=set(IdeaContent.model_fields))
        draft = IdeaDraft(**content, origin=IdeaOrigin.RECONSTRUCTED, parent_ids=[idea.id])
        [rebuilt] = state.add_ideas([draft], experiment.iteration)
        log.info("%s implemented a different idea than %s; result attributed to reconstructed %s %s", experiment.id, idea.id, rebuilt.id, rebuilt.title)
        idea.status = IdeaStatus.RANKED if retries_left else IdeaStatus.DISCARDED
        audit.idea_id = rebuilt.id
        idea = rebuilt
    state.audit_results.append(audit)
    if audit.trusted:
        solved_or_useful = audit.task_solved or audit.score >= config.discard_score
        idea.status = IdeaStatus.EVIDENCE_COLLECTED if solved_or_useful else IdeaStatus.DISCARDED
    else:
        texts.insert(0, f"{experiment.id} discarded by audit ({', '.join(f.value for f in audit.flags)}): {audit.reasoning}")
        idea.status = IdeaStatus.RANKED if retries_left else IdeaStatus.DISCARDED
    state.lessons += [
        Lesson(iteration=experiment.iteration, experiment_id=experiment.id, idea_id=idea.id, text=t, trusted=audit.trusted)
        for t in texts
    ]
    log.info(
        "%s audit: trusted=%s solved=%s score=%g -> %s is now %s",
        experiment.id, audit.trusted, audit.task_solved, audit.score, idea.id, idea.status.value,
    )


def check_stop(state: ResearchState, config: ResearchConfig) -> str:
    best = state.best_audit()
    if best and best.task_solved and best.score >= config.success_score:
        return f"Sufficiently good solution found: {best.idea_id} (score {best.score:g})."
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
            f"{audit.idea_id} {state.ideas[audit.idea_id].title} (audit score {audit.score:g}, {verdict}): "
            f"{experiment.report.implementation_summary} Result: {experiment.report.result} Artifacts: {experiment.workdir}"
        )
    return "\n".join(lines)
