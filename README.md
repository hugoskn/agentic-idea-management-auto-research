# AgenticIdeaManagementAutoResearch

An agentic research loop inspired by *AIM: Agentic Idea Management for Automated Research*, built on the Claude Agent SDK (`ClaudeSDKClient`). Ideas are first-class objects that move through a lifecycle: generated → clustered → ranked → selected → implemented → audited → evidence collected → refined or discarded.

## Requirements

- Python 3.10+ (developed on 3.12)
- Claude Code CLI installed and authenticated (`claude --version`)
- `pip install -r AgenticIdeaManagementAutoResearch/requirements.txt`

Set `ANTHROPIC_MODEL` to choose the model the agents use; otherwise the CLI default applies.

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

The run creates a folder named after the problem containing:

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
    resource_planner.py   exploration/exploitation split, stagnation detection
  main.py          entry point and CSV export
  self_check.py    offline check of state transitions, planning, stop rules and CSV output
```

- Agents never mutate the research state. Their outputs are validated by Pydantic and by per-agent checks (exact idea count, every idea clustered/ranked once, score spread, selection within budget); a rejected output is retried with the rejection reason.
- Selected ideas run in parallel, each Solver in its own directory, and each Solver's result goes to the SolutionAuditor. The Solver keeps a multi-turn `ClaudeSDKClient` session so it can refine its implementation from execution feedback.
- Experiments whose audit is invalid, misattributed or reward-hacked are kept for the record but excluded from the evidence and lessons fed to later agents.
- The resource planner turns cluster coverage, the best verified score and stagnation into exploration/exploitation slots that the AcquisitionAgent must follow or explicitly justify deviating from.
- The loop stops when a verified solution reaches `success_score`, the experiment budget runs out, `max_iterations` is reached, there has been no improvement for `patience` iterations, or the AcquisitionAgent decides more experiments are not worth it.

Solvers and auditors run Bash in their experiment directory. Bash sandboxing is not available on Windows, so run untrusted problems inside a VM or container.

Offline self-check (makes no API calls): `cd AgenticIdeaManagementAutoResearch/src && python self_check.py`
