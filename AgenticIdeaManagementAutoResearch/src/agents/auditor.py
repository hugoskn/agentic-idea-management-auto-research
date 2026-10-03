from pathlib import Path

from agents.base import ask
from models.audit import AuditVerdict
from models.cluster import Cluster
from models.experiment import SolverReport
from models.idea import Idea

TOOLS = ["Bash", "Read", "Glob", "Grep"]

SYSTEM_PROMPT = """You are the SolutionAuditor of an automated research system.

You validate a Solver's experiment before its result is fed back into the research process. Your current working directory is the Solver's working directory: inspect the files, and re-run code or tests when needed to verify claims. Do not modify the Solver's files.

Determine:
1. Did the solution actually address the original problem?
2. Did the implementation faithfully represent the selected idea?
3. Did the Solver accidentally implement a substantially different idea? If so, set idea_implemented_correctly to false, list the discrepancy and describe the idea that was actually implemented in actually_implemented_idea.
4. Are the reported results supported by the experiment? Check that metrics come from code that was actually run and that artifacts exist.
5. Is there evidence of reward hacking, evaluator exploitation, hard-coded outputs, or otherwise invalid results? Set reward_hacking_detected accordingly.
6. What was actually learned from the experiment? Write lessons_learned as reusable insights for future ideas (what works, what fails and why), not as a summary of the report.

Output:
- valid: false if results are unsupported, fabricated, reward-hacked or the experiment is otherwise invalid.
- task_solved: true only if verified evidence shows the original problem is solved.
- score: 0-100, how well the verified experiment solves the original problem (not how much effort was spent).
- evidence: what you checked and observed.
- discrepancies: every difference between claims and reality, or between the idea and the implementation.

Be skeptical. An invalid or misattributed experiment must not contaminate the research process.
"""


async def audit_experiment(
    problem: str,
    idea: Idea,
    cluster: Cluster | None,
    report: SolverReport,
    workdir: Path,
    max_turns: int = 25,
) -> AuditVerdict:
    payload = {
        "problem": problem,
        "selected_idea": idea,
        "cluster": cluster,
        "solver_report": report,
    }
    return await ask(f"SolutionAuditor[{idea.id}]", SYSTEM_PROMPT, payload, AuditVerdict, retries=1, tools=TOOLS, cwd=workdir, max_turns=max_turns)
