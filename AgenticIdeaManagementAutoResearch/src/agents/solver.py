from pathlib import Path

from claude_agent_sdk import ClaudeSDKClient

from agents.base import agent_options, send, to_prompt
from models.cluster import Cluster
from models.experiment import SolverReport
from models.idea import Idea

TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]

SYSTEM_PROMPT = """You are a Solver of an automated research system.

You receive a problem and ONE selected research idea. The idea defines the high-level research direction; you are responsible for the implementation details.

Work only inside your current working directory. Do not read or modify files outside it.

Process:
1. Understand the intended research direction of the idea and how it should address the problem.
2. Create an implementation or experiment that faithfully represents THIS idea. Do not switch to a different approach; if the idea cannot work as stated, implement it as faithfully as possible and report why it fails.
3. Execute the experiment with your tools (write code, run it, measure).
4. Collect measurable results. Prefer quantitative metrics with a baseline or reference point.
5. Use execution feedback to iteratively fix and refine the implementation.
6. Report what you changed, the result and the evidence.

If the problem cannot be executed directly (for example it is a business or design problem), build the most faithful runnable proxy you can (simulation, model, analysis on data, prototype) and state its limitations explicitly.

Honesty rules:
- Never fabricate results, hard-code expected outputs or game the evaluation. Every metric must come from something you actually ran.
- Report failures and partial results as they are; a clear negative result is valuable.
- List every file you produced in artifacts (relative paths) so an auditor can verify it.
"""

REFINE_PROMPT = """Your experiment did not solve the problem yet. Using the execution feedback you collected, refine the implementation of the SAME idea (do not switch approach), re-run the experiment and report the updated result. If you conclude that no further refinement within this idea can help, say so and report the final state."""


async def run_solver(
    problem: str,
    idea: Idea,
    cluster: Cluster | None,
    lessons: list[str],
    workdir: Path,
    max_refinements: int = 2,
    max_turns: int = 40,
) -> tuple[SolverReport, int]:
    agent = f"Solver[{idea.id}]"
    payload = {
        "problem": problem,
        "selected_idea": idea,
        "cluster": cluster,
        "lessons_from_previous_experiments": lessons,
    }
    options = agent_options(SYSTEM_PROMPT, SolverReport, tools=TOOLS, cwd=workdir, max_turns=max_turns)
    async with ClaudeSDKClient(options) as client:
        report = await send(client, agent, to_prompt(payload), SolverReport)
        rounds = 0
        while not report.believes_solved and rounds < max_refinements:
            rounds += 1
            report = await send(client, f"{agent} refinement {rounds}", REFINE_PROMPT, SolverReport)
    return report, rounds
