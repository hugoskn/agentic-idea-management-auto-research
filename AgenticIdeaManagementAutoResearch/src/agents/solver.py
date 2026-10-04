from pathlib import Path
from typing import Awaitable, Callable

from claude_agent_sdk import ClaudeSDKClient

from agents.base import agent_options, send, to_prompt
from models.cluster import Cluster
from models.experiment import Evaluation, SolverReport
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

If scored_by_evaluator is true, a fixed evaluator scores the files in your working directory after each report and you receive its output. Follow the solution interface described in the problem exactly (file names, entry points, output format). You cannot run the evaluator yourself.

If the problem cannot be executed directly (for example it is a business or design problem), build the most faithful runnable proxy you can (simulation, model, analysis on data, prototype) and state its limitations explicitly.

Honesty rules:
- Never fabricate results, hard-code expected outputs or game the evaluation. Every metric must come from something you actually ran.
- Report failures and partial results as they are; a clear negative result is valuable.
- List every file you produced in artifacts (relative paths) so an auditor can verify it.
"""

REFINE_PROMPT = """Your experiment did not solve the problem yet. Using the execution feedback you collected, refine the implementation of the SAME idea (do not switch approach), re-run the experiment and report the updated result. If you conclude that no further refinement within this idea can help, say so and report the final state."""

EVALUATED_REFINE_PROMPT = """The fixed evaluator scored your working directory (0-100, higher is better; invalid means it failed or found the solution incorrect). Its output is below. Refine the implementation of the SAME idea (do not switch approach) to raise the score and report the updated result. If a change lowers the score, revert it: the files you leave behind are what gets evaluated and audited.

"""


async def run_solver(
    problem: str,
    idea: Idea,
    cluster: Cluster | None,
    lessons: list[str],
    workdir: Path,
    max_refinements: int = 2,
    evaluate: Callable[[], Awaitable[Evaluation]] | None = None,
    max_turns: int = 40,
) -> tuple[SolverReport, int, list[Evaluation]]:
    agent = f"Solver[{idea.id}]"
    payload = {
        "problem": problem,
        "selected_idea": idea,
        "cluster": cluster,
        "lessons_from_previous_experiments": lessons,
        "scored_by_evaluator": evaluate is not None,
    }
    options = agent_options(SYSTEM_PROMPT, SolverReport, tools=TOOLS, cwd=workdir, max_turns=max_turns)
    evaluations: list[Evaluation] = []
    async with ClaudeSDKClient(options) as client:
        report = await send(client, agent, to_prompt(payload), SolverReport)
        rounds = 0
        while rounds < max_refinements:
            if evaluate:
                evaluations.append(await evaluate())
                if evaluations[-1].score >= 100:
                    break
                prompt = EVALUATED_REFINE_PROMPT + to_prompt({"evaluation": evaluations[-1]})
            elif report.believes_solved:
                break
            else:
                prompt = REFINE_PROMPT
            rounds += 1
            report = await send(client, f"{agent} refinement {rounds}", prompt, SolverReport)
    if evaluate and len(evaluations) == rounds:
        evaluations.append(await evaluate())
    return report, rounds, evaluations
