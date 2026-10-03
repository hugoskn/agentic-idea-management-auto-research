# AgenticIdeaManagementAutoResearch

An agentic research loop inspired by *AIM: Agentic Idea Management for Automated Research*, built on the Claude Agent SDK (`ClaudeSDKClient`). Ideas are first-class objects that move through a lifecycle: generated → clustered → ranked → selected → implemented → audited → evidence collected → refined or discarded.

## Requirements

- Python 3.10+ (developed on 3.12)
- Claude Code CLI installed and authenticated (`claude --version`)
- `pip install -r AgenticIdeaManagementAutoResearch/requirements.txt`

## Credentials and provider

By default the agents use your Claude Code login session. To use something else, copy `AgenticIdeaManagementAutoResearch/.env.example` to `.env` in the same folder (it is gitignored) or set the same variables in your shell; shell values take precedence over `.env`, and empty values are ignored.

| Goal | Variables |
|---|---|
| Your own Anthropic API key | `ANTHROPIC_API_KEY` (optional `ANTHROPIC_MODEL`) |
| DeepSeek or another Anthropic-compatible API | `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic`, `ANTHROPIC_API_KEY=<provider key>`, `ANTHROPIC_MODEL=deepseek-flash` |
| Gateway with bearer-token auth | `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` |

Claude Code picks `ANTHROPIC_AUTH_TOKEN` first, then `ANTHROPIC_API_KEY`, then the login session. Each run logs which credential, endpoint and model it used (never the key itself). An `ANTHROPIC_API_KEY` already exported in your shell also overrides the login session.

## Usage

From the command line:

```bash
cd AgenticIdeaManagementAutoResearch/src
python main.py "Find the fastest way to deduplicate 10M near-identical strings in Python" --ideas 6 --budget 6 --iterations 3
```

From Python (with `src` on the path):

```python
from main import main

csv_path = main("Reduce the p95 latency of a JSON parsing benchmark", ideas_count=5, max_iterations=2, experiment_budget=4)
print(csv_path)
```

The run saves its output in `AgenticIdeaManagementAutoResearch/results/{the_problem}/` (the problem trimmed and sanitized to a valid folder name; override the root with `--output` or `output_root`), containing:

- `results.csv`: one row per iteration with the problem, generated ideas, clusters, ranked ideas, selected ideas, execution results, audit results, lessons learned, and the final recommended solution in the last row.
- `research_state.json`: the full research state, saved after every step.
- `research.log`: the run log.
- `experiments/E<n>_<idea>/`: each Solver's isolated working directory.

## Architecture

```
src/
  agents/          one ClaudeSDKClient agent per responsibility, structured JSON in and out
    base.py        shared client/options, structured-output parsing, validated retries
    idea_generator.py, clusterizer.py, ranker.py, acquisition.py, solver.py, auditor.py
  models/          Pydantic models (Idea, Cluster, Ranking, Experiment, Audit, ResearchState)
  orchestration/
    research_loop.py      owns the state; generate → cluster → rank → select → execute ∥ → audit → learn
    resource_planner.py   LLM planner of Solver branches per iteration, stagnation detection
  main.py          entry point and CSV export
  self_check.py    offline check of state transitions, planning, stop rules and CSV output
```

- Agents never mutate the research state. Their outputs are validated by Pydantic and by per-agent checks (exact idea count, every idea clustered/ranked once, score spread, selection within budget); a rejected output is retried with the rejection reason.
- Selected ideas run in parallel, each Solver in its own directory, and each Solver's result goes to the SolutionAuditor. The Solver keeps a multi-turn `ClaudeSDKClient` session so it can refine its implementation from execution feedback.
- The SolutionAuditor flags `trivial`, `task_mismatch`, `idea_mismatch` and `reward_hacking`. Results flagged `trivial`, `task_mismatch` or `reward_hacking` are kept for the record but excluded from the evidence and lessons fed to later agents; they still consume budget. When `idea_mismatch` is the only flag, the auditor reconstructs the idea the code actually implements, it joins the pool (origin `reconstructed`) and the score and lessons are attributed to it.
- `--budget` is the fixed total of Solver branches. Iteration 1 is pinned to 5 branches; from iteration 2 the Resource Planner agent re-plans how the remaining branches are spread over iterations (1 to `--parallel` per iteration, preferring at least 3, within `--iterations`), trading parallel breadth against more frequent feedback. The AcquisitionAgent fills exactly that many branches and decides the exploration/exploitation balance itself.
- The loop stops when a verified solution reaches `success_score`, the experiment budget runs out, `max_iterations` is reached, there has been no improvement for `patience` iterations, or the AcquisitionAgent decides more experiments are not worth it.

Solvers and auditors run Bash in their experiment directory. Bash sandboxing is not available on Windows, so run untrusted problems inside a VM or container.

Offline self-check (makes no API calls): `cd AgenticIdeaManagementAutoResearch/src && python self_check.py`
