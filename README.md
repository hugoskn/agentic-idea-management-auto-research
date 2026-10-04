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

### Evaluator (recommended)

Pass `--evaluator "<command>"` (or `evaluator=` from Python) to score solutions with a fixed evaluator instead of the auditor's judgement, as in the AIM paper. The command runs in each experiment directory after every Solver attempt (up to `1 + max_refinements` times per branch) and must print the score (0-100, higher is better) as the last line of its output. A non-zero exit code, a timeout (`evaluator_timeout`, default 600 s) or a non-numeric last line counts as invalid and scores 0. Make the evaluator return 0 for incorrect solutions and normalize the score against your baseline, so that only real improvements score above 0.

The Solver does not get the command: it receives the evaluator output after each attempt and refines, so describe in the problem what the evaluator expects (file names, entry point, output format). Keep the evaluator script outside the experiment directories, e.g. with an absolute path:

```bash
python main.py "Write solve.py exposing sort(xs) that sorts 10M ints faster than sorted()" --evaluator "python C:/tasks/sort/eval.py"
```

The score of the final files enters the research state; the SolutionAuditor reviews it for reward hacking and idea/task mismatches.

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

- Agents never mutate the research state. Their outputs are validated by Pydantic and by per-agent checks (exact idea count, every idea clustered exactly once into 2-5 clusters, dense ranks covering every cluster/idea, selection filling the planned branches); a rejected output is retried with the rejection reason.
- Every iteration the Clusterizer re-organizes the complete idea pool, tested and discarded ideas included (with their best scores) as landmarks, starting from the previous clustering. The Ranker then estimates promise ordinally, as in the paper: dense ranks for the clusters (from their evaluated scores and the lessons), then dense ranks for the untested ideas within each cluster. No numeric score is predicted.
- New ideas come from the paper's Expand operator: after an iteration, every branch gets its own Expander call with up to `max_ideas_per_branch` (default 3) proposal slots, at most one per kind: `fix_error` (branch failed, including Solver crashes), `push_score` (branch succeeded), `cross_pollinate` (must cite exactly one sibling outcome or lesson, otherwise it is dropped) and `new_idea` (an orthogonal direction). The Expander sees the branch's idea and outcome, the sibling outcomes, cluster context, the run's top-3/bottom-3 and the lessons. Branches the auditor discarded are not expanded.
- Selected ideas run in parallel, each Solver in its own directory, and each Solver's result goes to the SolutionAuditor. The Solver keeps a multi-turn `ClaudeSDKClient` session so it can refine its implementation from execution feedback.
- The SolutionAuditor flags `trivial`, `task_mismatch`, `idea_mismatch` and `reward_hacking`. Results flagged `trivial`, `task_mismatch` or `reward_hacking` are kept for the record but excluded from the evidence and lessons fed to later agents; they still consume budget. When `idea_mismatch` is the only flag, the auditor reconstructs the idea the code actually implements, it joins the pool (origin `reconstructed`) and the score and lessons are attributed to it.
- `--budget` is the fixed total of Solver branches. Iteration 1 is pinned to 5 branches; from iteration 2 the Resource Planner agent re-plans how the remaining branches are spread over iterations (1 to `--parallel` per iteration, preferring at least 3, within `--iterations`), trading parallel breadth against more frequent feedback. The AcquisitionAgent then dispatches them in two stages, as in the paper: one call decides the shape of the iteration (which cluster each branch goes to, with a cluster-level and an idea-level explore/exploit action per branch; plans that overfill a cluster or break the branch count are rejected), then one call per branch picks the specific untested idea that matches its idea-level action, never one an earlier branch already took. In iteration 1 every idea-level action is exploit, to validate the top-ranked ideas. A Solver crash counts as an evaluated failure (its `fix_error` ideas take over); a result the auditor discarded leaves its idea untested, so it can be dispatched again up to `max_attempts_per_idea`.
- The loop stops when a verified solution reaches `success_score`, the experiment budget runs out, `max_iterations` is reached, there has been no improvement for `patience` iterations, or the AcquisitionAgent decides more experiments are not worth it.

Solvers and auditors run Bash in their experiment directory. Bash sandboxing is not available on Windows, so run untrusted problems inside a VM or container.

Offline self-check (makes no API calls): `cd AgenticIdeaManagementAutoResearch/src && python self_check.py`
