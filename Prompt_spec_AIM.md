Using the Claude Agent SDK (`ClaudeSDKClient`: https://code.claude.com/docs/en/agent-sdk/python), I need you to design and implement an agentic research flow inspired by the "AIM: Agentic Idea Management for Automated Research" approach.

The goal is to create a reusable orchestration flow where the input is a problem, the system generates multiple possible approaches, organizes and evaluates them, selects the most promising ones, executes them, validates the results, and learns from the outcomes.

## 1. Main Entry Point

Create a main function that receives:

* `problem: str`

  * The problem that needs to be researched and resolved.
  * This is the primary input to the entire agentic flow.

* `ideas_count: int`

  * The number of initial research ideas that should be generated.
  * Default to a reasonable value if not provided.

The function should generate a CSV file in a new folder named as the problem (trimmed). And the file must contain a structured result with the following columns:

* The original problem
* Generated ideas
* Idea clusters
* Ranked ideas
* Selected ideas
* Execution results
* Auditing results
* Lessons learned
* Final recommended solution or solutions

Use strongly typed models where appropriate, preferably Pydantic models.

---

## 2. Researcher / Idea Generator Agent

Create a `Researcher` or `IdeaGenerator` agent.

Its responsibility is to generate exactly `ideas_count` distinct ideas for solving the problem.

The agent's `system_instructions` must explicitly tell it:

* Understand the problem before generating ideas.
* Generate genuinely different approaches, not minor variations of the same approach.
* Think broadly across different solution directions.
* Each idea must be concrete enough that another agent could implement or test it.
* For every idea, explain:

  * What the idea is.
  * How it would address the problem.
  * Why the agent believes it has a high probability of solving the problem or significantly contributing to its resolution.
  * The assumptions behind the idea.
  * The main risks or reasons it might fail.
* Do not rank the ideas yet.
* Do not prematurely converge on one solution.
* Prefer semantic diversity between ideas.

The output should be structured, for example:

```text
Idea {
    id
    title
    description
    rationale
    assumptions
    risks
}
```

---

## 3. Clusterizer Agent

Create a `Clusterizer` agent.

Its responsibility is to organize the generated ideas into meaningful semantic clusters.

The agent should:

* Analyze all generated ideas.
* Identify the major solution directions represented by the ideas.
* Group semantically similar ideas together.
* Avoid creating unnecessary clusters.
* Give each cluster a clear name and description.
* Explain why each idea belongs to its assigned cluster.
* Identify clusters that represent fundamentally different approaches to the problem.

Example:

```text
Cluster: Caching-Based Solutions
    - Idea 2
    - Idea 7
    - Idea 11

Cluster: Database Optimization
    - Idea 1
    - Idea 4

Cluster: Architectural Changes
    - Idea 3
    - Idea 8
    - Idea 9
```

The goal is to make the solution space explicit and understandable before selecting which ideas to investigate.

---

## 4. Ranker Agent

Create a `Ranker` agent.

Its responsibility is to evaluate every idea and assign a score from `0` to `100`.

The score represents the estimated likelihood that the idea will:

1. Solve the problem, or
2. Make a significant contribution toward solving the problem.

For every idea, the agent MUST provide:

* `score`: integer from 0 to 100
* `reasoning`: detailed justification for the score
* `expected_impact`
* `confidence`
* `key_assumptions`
* `main_risks`

Important:

The score must represent the idea's expected value for solving the specific problem, not whether the idea is easy or interesting.

The ranker should consider factors such as:

* Expected impact
* Technical feasibility
* Relevance to the problem
* Evidence available from previous experiments
* Complexity
* Risk
* Potential for producing a breakthrough
* Whether the idea represents a meaningfully different solution direction

Do not allow all ideas to receive similar scores without justification.

---

## 5. Acquisition / Selection Agent

Create an `AcquisitionAgent`.

Its responsibility is to decide which ideas should actually be executed.

Do not simply select the highest-scoring ideas.

The agent should balance:

* Exploitation: investigating highly promising ideas further.
* Exploration: testing different solution directions that have not yet been sufficiently explored.

For example, if there are three highly ranked ideas belonging to the same cluster, the agent should consider whether testing another promising idea from a different cluster would provide better exploration of the solution space.

The agent should receive:

* Ideas
* Clusters
* Scores
* Previous experiment results, if any
* Remaining experiment budget

It should return:

```text
SelectedIdea {
    idea_id
    priority
    selection_reason
    exploration_or_exploitation
}
```

The selection decision must be explicitly justified.

---

## 6. Solver / Execution Agent

For every selected idea, create a separate Solver agent.

The Solver receives:

* The original problem
* The selected idea
* The idea's rationale
* Relevant cluster information
* Any lessons from previous experiments

Its job is to actually implement or execute the idea.

The Solver should:

1. Understand the intended research direction.
2. Create an implementation or experiment.
3. Execute the experiment.
4. Collect measurable results.
5. Explain what it changed.
6. Report the result and evidence.

The solver should be allowed to iteratively refine its implementation using execution feedback.

This is important: the idea should define the high-level research direction, while the Solver is responsible for figuring out the implementation details.

---

## 7. Solution Auditor Agent

Create a `SolutionAuditor` agent.

This agent is responsible for validating the Solver's result before the result is fed back into the research process.

It must determine:

1. Did the solution actually address the original problem?
2. Did the implementation faithfully represent the selected idea?
3. Did the Solver accidentally implement a substantially different idea?
4. Are the reported results supported by the experiment?
5. Is there evidence of reward hacking, evaluator exploitation, or invalid results?
6. What was actually learned from the experiment?

The auditor should produce:

```text
AuditResult {
    valid
    idea_implemented_correctly
    task_solved
    score
    evidence
    discrepancies
    lessons_learned
}
```

If the implementation does not correspond to the original idea, explicitly identify the discrepancy and explain what idea was actually implemented.

Do not allow an invalid or misattributed experiment to contaminate the subsequent research process.

---

## 8. Learning / Research Memory

Create a persistent research state for the current problem.

The state should maintain at least:

```text
ResearchState {
    problem
    ideas
    clusters
    rankings
    experiments
    audit_results
    lessons
    remaining_budget
}
```

After each experiment:

1. Store the result.
2. Store the audit.
3. Extract lessons.
4. Update the ranking of related ideas.
5. Generate new ideas based on what was learned.

New ideas should be allowed to come from:

* Refining successful ideas.
* Combining successful ideas from different clusters.
* Fixing failed ideas.
* Exploring completely new directions.

---

## 9. Iterative Research Loop

The overall orchestration should work approximately like this:

```text
Problem
   ↓
Generate Ideas
   ↓
Cluster Ideas
   ↓
Rank Ideas
   ↓
Select Ideas
   ↓
Execute Selected Ideas
   ↓
Audit Results
   ↓
Extract Lessons
   ↓
Update Research State
   ↓
Generate New Ideas
   ↓
Re-rank
   ↓
Select Again
   ↓
...
```

The loop should continue until one of the following conditions occurs:

* A sufficiently good solution is found.
* The experiment budget is exhausted.
* The maximum number of iterations is reached.
* Further experimentation is unlikely to provide meaningful value.

---

## 10. Exploration vs Exploitation

Make exploration vs exploitation an explicit concept in the architecture.

The system should avoid two extremes:

### Pure exploitation

Continuously improving the same promising idea without investigating alternatives.

### Pure exploration

Continuously generating new ideas without spending enough effort implementing and refining promising ones.

The system should dynamically determine the appropriate balance based on the evidence collected so far.

---

## 11. Parallel Execution

Use the Claude Agent SDK's capabilities to execute independent Solver agents in parallel where appropriate.

For example:

```text
Selected Idea A → Solver A
Selected Idea B → Solver B
Selected Idea C → Solver C
```

These should execute independently when there are no dependencies between them.

The orchestration layer should collect their results and send them to the Solution Auditor.

Be careful with shared resources and state.

---

## 12. ClaudeSDKClient

Use `ClaudeSDKClient` as the primary mechanism for interacting with Claude agents.

Design the code so that:

* Each agent has clear system instructions.
* Agents receive structured input.
* Agents return structured output.
* The orchestration layer owns the overall state.
* Agents do not directly modify global research state.
* Agent outputs are validated before being added to the research state.

Avoid building one giant system prompt that performs all responsibilities.

The architecture should explicitly separate:

```text
Idea Generation
       ↓
Organization
       ↓
Evaluation
       ↓
Selection
       ↓
Execution
       ↓
Auditing
       ↓
Learning
```

---

## 13. Project Structure

Create a clean Python project structure similar to:

```text
src/
    agents/
        idea_generator.py
        clusterizer.py
        ranker.py
        acquisition.py
        solver.py
        auditor.py

    models/
        idea.py
        cluster.py
        ranking.py
        experiment.py
        audit.py
        research_state.py

    orchestration/
        research_loop.py
        resource_planner.py

    main.py
```

Keep the individual agents independently testable.

---

## 14. Important Design Principle

The system should treat **ideas as first-class objects**, not merely intermediate text.

An idea should have a lifecycle:

```text
Generated
   ↓
Clustered
   ↓
Ranked
   ↓
Selected
   ↓
Implemented
   ↓
Audited
   ↓
Evidence collected
   ↓
Updated / refined / discarded
```

The goal is to build an agentic research system that can reason about the **solution space**, not merely repeatedly modify the current solution.

---

## 15. Implementation Expectations

Please implement the complete working prototype.

Before writing code:

1. Inspect the current project.
2. Determine the existing Python version and dependency management.
3. Check whether the Claude Agent SDK is already installed.
4. Reuse existing project conventions where appropriate.
5. If dependencies are missing, explain what needs to be installed.

Then implement the system.

Include:

* Pydantic models
* ClaudeSDKClient integration
* Agent system instructions
* Structured agent outputs
* Research state management
* Iterative orchestration
* Parallel solver execution where appropriate
* Exploration/exploitation logic
* Solution auditing
* Error handling
* Logging
* A simple example showing how to call the main entry point

The final implementation should be practical and runnable, not pseudocode.

When making architectural decisions, prioritize simplicity and maintainability while preserving the core idea-management architecture described above.
