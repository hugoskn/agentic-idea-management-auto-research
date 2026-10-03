import argparse
import asyncio
import csv
import logging
import re
from pathlib import Path

from models.research_state import IterationRecord, ResearchState
from orchestration.research_loop import ResearchConfig, run_research

log = logging.getLogger("aim")

MAX_WORKDIR_LENGTH = 240

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


def folder_name(problem: str, max_length: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", problem)
    name = re.sub(r"\s+", " ", name).strip()[:max_length].rstrip(" .")
    return name or "research"


def iteration_row(state: ResearchState, record: IterationRecord, is_last: bool) -> list[str]:
    experiments = [e for e in state.experiments if e.id in record.experiment_ids]
    audits = [a for a in state.audit_results if a.experiment_id in record.experiment_ids]
    ideas = [state.ideas[i] for i in record.new_idea_ids]
    clusters = record.clustering.clusters if record.clustering else []
    rankings = sorted(record.ranking.rankings, key=lambda r: -r.score) if record.ranking else []
    selected = sorted(record.selection.selected, key=lambda s: s.priority) if record.selection else []
    return [
        state.problem,
        "\n".join(f"{i.id} [{i.origin.value}] {i.title}: {i.description}" for i in ideas),
        "\n".join(f"{c.id} {c.name}: {', '.join(m.idea_id for m in c.members)}" for c in clusters),
        "\n".join(f"{r.idea_id} score={r.score} confidence={r.confidence.value}: {r.reasoning}" for r in rankings),
        "\n".join(f"P{s.priority} {s.idea_id} [{s.exploration_or_exploitation.value}]: {s.selection_reason}" for s in selected),
        "\n".join(f"{e.id} {e.idea_id}: {e.report.result if e.report else e.error}" for e in experiments),
        "\n".join(
            f"{a.experiment_id} {a.idea_id}: valid={a.valid} idea_implemented={a.idea_implemented_correctly} "
            f"solved={a.task_solved} score={a.score} discrepancies={'; '.join(a.discrepancies) or 'none'}"
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


async def research(problem: str, ideas_count: int = 6, output_root: Path = Path("."), **settings) -> Path:
    problem = problem.strip()
    if not problem:
        raise ValueError("problem must not be empty")
    config = ResearchConfig(ideas_count=ideas_count, **settings)
    root = Path(output_root).resolve()
    room = MAX_WORKDIR_LENGTH - len(str(root / "experiments" / "E999_I999"))
    if room < 10:
        raise ValueError(f"output_root is too deep for experiment working directories: {root}")
    out_dir = root / folder_name(problem, min(80, room))
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "research.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
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
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--budget", type=int, default=6, help="maximum number of experiments")
    parser.add_argument("--parallel", type=int, default=3, help="maximum concurrent solvers")
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = main(
        args.problem, args.ideas, output_root=args.output,
        max_iterations=args.iterations, experiment_budget=args.budget, max_parallel=args.parallel,
    )
    print(f"Results written to {path}")
