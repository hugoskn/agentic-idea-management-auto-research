import argparse
import asyncio
import csv
import logging
import os
import re
from pathlib import Path

from models.research_state import IterationRecord, ResearchState
from orchestration.research_loop import ResearchConfig, run_research

log = logging.getLogger("aim")

MAX_WORKDIR_LENGTH = 240
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
ENV_FILE = PROJECT_DIR / ".env"

COLUMNS = [
    "Original problem",
    "Generated ideas",
    "Idea clusters",
    "Ranked ideas",
    "Selected ideas",
    "Execution results",
    "Auditing results",
    "Lessons learned",
    "Final recommended solution",
]


def load_env(path: Path = ENV_FILE) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if sep and key and not key.startswith("#") and value:
            os.environ.setdefault(key, value)


def describe_connection() -> str:
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        credential = "ANTHROPIC_AUTH_TOKEN"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        credential = "ANTHROPIC_API_KEY"
    else:
        credential = "Claude Code login session"
    endpoint = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    model = os.environ.get("ANTHROPIC_MODEL", "CLI default")
    return f"credential={credential} endpoint={endpoint} model={model}"


def folder_name(problem: str, max_length: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", problem)
    name = re.sub(r"\s+", " ", name).strip()[:max_length].rstrip(" .")
    return name or "research"


def ranking_text(record: IterationRecord) -> str:
    if not record.ranking or not record.clustering:
        return ""
    names = {c.id: c.name for c in record.clustering.clusters}
    lines = [f"Clusters: {record.ranking.clusters.rationale}"]
    for r in sorted(record.ranking.clusters.cluster_ranks, key=lambda r: r.rank):
        ideas = record.ranking.ideas.get(r.cluster_id)
        ranked = ", ".join(f"{i.idea_id}={i.rank}" for i in sorted(ideas.idea_ranks, key=lambda i: i.rank)) if ideas else "no untested ideas"
        lines.append(f"#{r.rank} {r.cluster_id} {names.get(r.cluster_id, '')}: {ranked}" + (f" ({ideas.rationale})" if ideas else ""))
    return "\n".join(lines)


def iteration_row(state: ResearchState, record: IterationRecord, is_last: bool) -> list[str]:
    experiments = [e for e in state.experiments if e.id in record.experiment_ids]
    audits = [a for a in state.audit_results if a.experiment_id in record.experiment_ids]
    ideas = [state.ideas[i] for i in record.new_idea_ids]
    clusters = record.clustering.clusters if record.clustering else []
    selected = sorted(record.selection.selected, key=lambda s: s.branch) if record.selection else []
    return [
        state.problem,
        "\n".join(f"{i.id} [{i.origin.value}] {i.title}: {i.description}" for i in ideas),
        "\n".join(f"{c.id} {c.name}: {', '.join(m.idea_id for m in c.members)}" for c in clusters),
        ranking_text(record),
        "\n".join(
            ([f"Dispatch: {record.selection.plan.rationale_summary}"] if record.selection else [])
            + [f"b{s.branch} {s.idea_id} in {s.cluster_id} [cluster {s.cluster_action.value}, idea {s.idea_action.value}]: {s.rationale}" for s in selected]
        ),
        "\n".join(
            f"{e.id} {e.idea_id}: {e.report.result if e.report else e.error}"
            + (f" Evaluator scores: {', '.join(f'{v.score:g}' if v.valid else 'invalid' for v in e.evaluations)}" if e.evaluations else "")
            for e in experiments
        ),
        "\n".join(
            f"{a.experiment_id} {a.idea_id}: flags={', '.join(f.value for f in a.flags) or 'none'} "
            f"solved={a.task_solved} score={a.score:g} confidence={a.confidence:.2f} "
            f"discrepancies={'; '.join(a.discrepancies) or 'none'}"
            + (f" reconstructed from {state.ideas[a.idea_id].parent_ids[0]}: {state.ideas[a.idea_id].title}" if a.trusted and a.reconstructed_idea else "")
            for a in audits
        ),
        "\n".join(
            f"[{'trusted' if l.trusted else 'untrusted'}] {l.idea_id}: {l.text}"
            for l in state.lessons
            if l.experiment_id in record.experiment_ids
        ),
        f"{state.final_recommendation}\nStop reason: {state.stop_reason}" if is_last else "",
    ]


def write_csv(state: ResearchState, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for record in state.iterations:
            writer.writerow(iteration_row(state, record, record is state.iterations[-1]))


async def research(problem: str, ideas_count: int = 6, output_root: Path = RESULTS_DIR, **settings) -> Path:
    problem = problem.strip()
    if not problem:
        raise ValueError("problem must not be empty")
    config = ResearchConfig(ideas_count=ideas_count, **settings)
    load_env()
    root =Path(output_root).resolve()
    room = MAX_WORKDIR_LENGTH - len(str(root / "experiments" / "E999_I999"))
    if room < 10:
        raise ValueError(f"output_root is too deep for experiment working directories: {root}")
    out_dir = root / folder_name(problem, min(80, room))
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "research.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.info("Connection: %s", describe_connection())
    try:
        state = await run_research(problem, config, out_dir)
    finally:
        log.removeHandler(handler)
        handler.close()
    csv_path = out_dir / "results.csv"
    write_csv(state, csv_path)
    return csv_path


def main(problem: str, ideas_count: int = 6, **settings) -> Path:
    return asyncio.run(research(problem, ideas_count, **settings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIM: agentic idea management for automated research.")
    parser.add_argument("problem")
    parser.add_argument("--ideas", type=int, default=6, help="number of initial ideas")
    parser.add_argument("--iterations", type=int, default=3, help="hard cap on iterations planned by the Resource Planner")
    parser.add_argument("--budget", type=int, default=6, help="total number of Solver branches")
    parser.add_argument("--parallel", type=int, default=10, help="maximum Solver branches per iteration")
    parser.add_argument("--evaluator", help="shell command run in each experiment directory; its last output line is the 0-100 score")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = main(
        args.problem, args.ideas, output_root=args.output,
        max_iterations=args.iterations, experiment_budget=args.budget, max_branches=args.parallel, evaluator=args.evaluator,
    )
    print(f"Results written to {path}")
