# FastEval

## What it is

FastEval is a generic evaluation runner for agent benchmarks where the bottleneck is not model inference alone, but the full loop:

1. model call
2. tool call
3. artifact generation
4. scoring

The core idea is simple:

- keep the orchestration layer async
- push expensive tool execution out of the local machine
- bound concurrency explicitly
- support multiple scoring modes
- make benchmark-specific logic pluggable rather than hard-coded

In the intended form, FastEval is not a benchmark itself. It is the execution substrate for benchmarks.


## Why this exists

Most eval harnesses are accidentally tuned for low parallelism:

- one local machine runs many Python/bash tool calls
- local CPU becomes the real cap
- sandboxes sit idle between turns
- judging is tied to one model or one benchmark format

FastEval is meant to flip that model:

- model calls are mostly network wait
- tool execution can either be local CPU work or another remote I/O hop
- the host becomes an I/O orchestrator
- the active cap becomes a mix of provider rate limits, local runtime capacity, remote runtime capacity, and judge throughput

That is the reason to call it FastEval: more tasks in flight without local CPU becoming the first bottleneck.


## Core model

FastEval should center on one idea:

- one task schema
- one execution path
- one output schema
- multiple scoring modes

The benchmark-specific work should mostly happen at ingest time:

1. lay the benchmark out into the FastEval task format
2. run every task through the same execution loop
3. choose scoring based on fields present in the task row
4. write one result row with answer, artifacts, thinking, and score

That means the system should not treat LLM judging as the default concept.
LLM judging is one scorer.
Ground-truth comparison is another scorer.
Programmatic verification is another scorer.

This is the central principle:

- assume the schema first
- build execution and scoring around that schema
- judge generality by whether new benchmarks fit the schema without invasive special-casing


## Task schema

The task schema should stay stable across benchmarks.

Core fields:

- `id`
- `task`
- `reference_files`
- `config`
- `output_dir`

`output_dir` should be understood as artifact policy, not as the universal workspace.
Some benchmarks want the model to work directly in the artifact directory.
Others want the model to work in `/app`, `/srv`, or another task-defined workspace and only sync selected files back later.

Optional scoring fields:

- `rubric`
- `ground_truth`
- `match_type`
- `case_sensitive`
- `normalize`
- later: `verifier`, `scorer`

Optional execution fields:

- `setup`
- `environment`
- `session`
- `allowed_tools`
- `artifact_paths`
- `timeout`
- `workspace_root`
- `default_cwd`
- `sync_paths`
- `prompt_instructions`

The important point is that tasks do not need separate benchmark-specific run formats.
They need a common execution schema plus optional scoring metadata.

`output_dir` can remain optional.
For benchmarks without artifact generation, it can be absent.
For benchmarks that produce files, it can point to an artifact directory.


## Execution contract

The task schema needs one more explicit concept: an execution contract per case.

That contract separates four things that should not be overloaded into one `output_dir` field:

- prompt contract
- workspace contract
- artifact contract
- runtime contract

### Prompt contract

Defines what the model is told about files and paths.

Examples:

- `kwbench`: write outputs only inside the artifact directory
- `terminal_bench`: work in `/app` and create files exactly where the task says

### Workspace contract

Defines where tools actually execute.

Examples:

- bash working directory
- Python working directory
- writable roots
- whether Python file writes are redirected
- whether Python writes are limited to the cwd subtree

### Artifact contract

Defines what gets synced back to the local results directory.

Examples:

- sync the whole output directory
- sync nothing
- sync selected files from a benchmark workspace

### Runtime contract

Defines how the execution backend should honor the case.

Examples:

- local CPU-bound session
- Daytona remote session
- later another remote backend

This is how `kwbench` should be preserved without a hidden fallback:

- `kwbench` becomes one explicit execution contract
- `terminal_bench` becomes another explicit execution contract
- the engine resolves the contract per case and passes it through the same run path


## Scoring flow

The intended flow should be:

1. load a task in the common schema
2. run the task once
3. resolve the scorer
4. write one output row with the answer, model metadata, artifacts, and score payload

A good scoring precedence is:

1. if `rubric` exists, use judge scoring
2. else if `ground_truth` exists, use deterministic scoring
3. else if `verifier` exists, use verifier/programmatic scoring
4. else mark the task as unscored

That is a much cleaner abstraction than making the judge path foundational.

### Deterministic scoring

For benchmarks with known answers, the default should be deterministic scoring rather than an LLM judge.

That means:

- exact match
- normalized exact match
- regex match
- contains/set matching when needed

In the simplest form:

- `ground_truth` exists
- no rubric exists
- compare answer against `ground_truth`
- store the score directly

This is cheaper, faster, and more benchmark-correct for things like math or factual-answer tasks.

### Judge scoring

Judge scoring should be used when deterministic comparison is not sufficient.

Examples:

- open-ended writing
- business analysis
- artifact quality
- multi-criterion task success

The current rubric flow fits here, but it should be understood as one scoring mode, not the default architecture.

### Verifier scoring

Some benchmarks are not best expressed as either a rubric or a simple ground-truth compare.

Examples:

- coding tasks with hidden tests
- terminal tasks with expected side effects
- environment-dependent tasks

Those need a verifier path:

- execute the task
- run a benchmark-defined verifier
- record pass/fail or metric output

This is especially important for terminal and coding benchmarks.


## Provider independence

FastEval should be provider-independent at the interface level.

That means the benchmark/task layer should not care whether:

- the subject model is OpenAI, Gemini, Claude, or an OpenAI-compatible endpoint
- the judge model is Gemini, OpenAI, Claude, or another configured provider
- the runtime backend is local or Daytona

But there still needs to be one place where these decisions are resolved.

That resolution layer should decide:

- subject provider
- judge provider
- runtime backend
- runtime concurrency
- scoring mode

The important design constraint is:

- provider-specific behavior is real
- provider-specific wiring should live in one runtime/config resolver
- benchmark/task definitions should not contain provider logic


## The right abstraction

The most important design insight is this:

- model I/O and tool execution are different systems
- they should be scheduled together
- they should not be the same abstraction

What the runtime really does is:

1. call the subject model
2. wait for a response
3. only if the model asks for tools, resume some stateful execution context
4. run Python/bash or verifier work
5. suspend it again
6. continue model I/O

That means the core boundary is not "Daytona support."
It is not even "REPL support."

It is:

- benchmark adaptation
- model/judge I/O
- tool runtime
- scoring
- orchestration

FastEval should therefore be thought of as five layers.

### 1. Benchmark adapter

Defines:

- how to load tasks into the common task schema
- how to normalize another benchmark's JSONL into the FastEval JSONL
- what prompt shape to use
- what tools are allowed
- what scoring hints exist on the task rows

This is the user-written layer.
The benchmark author should do the schema conversion here and nowhere else.

Examples:

- rubric-scored knowledge-work benchmark
- exact-match math benchmark
- terminal benchmark with verifier
- coding benchmark with repo setup + unit tests

### 2. Model and judge I/O

Defines:

- subject model provider
- judge model provider
- conversation loop
- retry policy
- response metadata capture

This layer does network I/O.
It should not own local or remote execution semantics.

This already exists in partial form via the provider modules in [eval/llms]( eval/llms).

### 3. Tool runtime

Defines:

- where tool execution happens
- how state is preserved between tool calls
- how refs are uploaded
- how artifacts are synced back
- how Python/bash/verifier commands are executed
- how idle sessions are suspended

This is the layer where Daytona belongs.
It should be one implementation of a common interface, not a special-case path.

The runtime should consume the execution contract and honor:

- prompt path semantics
- bash cwd
- Python cwd
- Python write policy
- artifact sync policy

The resource model matters here:

- a `local` runtime is host-CPU-dependent
- a remote runtime like `daytona` is mostly another awaited network hop from the orchestrator's point of view
- both are "tool execution," but they stress different resources and should not be treated as the same capacity class

The right runtime concept is:

- `ToolRuntime`: backend type
- `ToolSession`: one case-local execution context

Examples:

- `local`
- `daytona`
- later: `docker`, `ssh`, `modal`, other remote compute backends

The key point is:

- the model should not care whether a tool call ran locally or on Daytona
- the benchmark should not care either
- only the runtime resolver should care

### 4. Scoring backend

Defines:

- deterministic scorers
- programmatic verifiers
- LLM judge backends
- aggregation logic
- scoring resolution from task fields

This should be benchmark-selectable, not globally hard-coded.

### 5. Run orchestration

Defines:

- scheduling
- resume
- progress metadata
- persistence
- uploads
- observability

This is the async service-first stack under [service]( service).


## Tool runtime and tool session

The tool runtime is the abstraction that makes "local CPU", "Daytona", or something else feel interchangeable.

But interchangeable does not mean identical at the resource level.
The important distinction is:

- local runtime work is CPU/process-bound on the host
- remote runtime work is primarily I/O-bound on the host and compute-bound on the remote backend

That is why the orchestration layer should choose a runtime backend without coupling model I/O to local compute.

From first principles, a case needs a stateful execution object with behavior like:

- `ensure_ready()`
- `run_python(...)`
- `run_bash(...)`
- `upload_files(...)`
- `download_artifacts(...)`
- `checkpoint()`
- `suspend()`
- `close()`

That means the lower-level abstraction should be closer to:

- `ToolRuntime`
  - factory / backend selection
- `ToolSession`
  - per-case state

The current repo has a useful start on this with `PythonREPL` and `DaytonaPythonREPL` in [eval/tools.py]( eval/tools.py), but the design should evolve from "pick a REPL implementation" to "pick a tool runtime backend."

This matters because some benchmarks are not fundamentally REPL benchmarks.
They are stateful-environment benchmarks.

Examples:

- math benchmark:
  - maybe no tools at all
- knowledge-work benchmark:
  - maybe Python and file output
- terminal benchmark:
  - needs bash, persistent filesystem, maybe persistent working directory
- coding benchmark:
  - needs repo materialization, tests, artifacts, maybe a verifier in the same environment


## Parallelism invariants

The abstraction is only useful if it preserves the current concurrency model.

These invariants should be explicit:

### 1. Model I/O and tool execution are decoupled

- model calls are mostly network wait
- tool calls consume bounded execution capacity
- a case can be waiting on the model without occupying tool runtime capacity

This is the main throughput win.

### 1a. Runtime capacity is backend-specific

FastEval should distinguish at least two resource classes:

- local runtime capacity:
  - CPU/process-bound
  - should be limited by something like `cpu_sem`
- remote runtime capacity:
  - mostly I/O-bound on the host
  - should be limited by remote backend concurrency and provider tolerance, not by local CPU alone

In other words:

- a local Python/bash call is not the same kind of resource consumption as a Daytona call
- both belong under the tool-runtime abstraction
- they should not collapse to the same mental model of "tool == host CPU"

The benchmark contract should not change that resource model.
Changing the prompt or workspace root must not accidentally occupy the main loop.
Switching from local to Daytona should change the runtime capacity class, not the benchmark adapter shape.

### 2. Session state must not imply active compute

- a `ToolSession` may exist without being actively started
- a remote sandbox may be stopped between tool calls
- a local process may be torn down and restored from checkpoint if the runtime supports that

This is what lets FastEval avoid paying for idle execution time.

### 3. The event loop must not do blocking lifecycle work

The async orchestration path must never directly block on:

- repo setup
- verifier execution
- package installation
- filesystem-heavy sync
- environment bootstrap

Those must be:

- awaited remote I/O, or
- offloaded to threads/processes, and
- bounded by an explicit gate

### 4. Tool-related lifecycle work must be bounded

The following all consume execution capacity and should be treated as such:

- creating a tool session
- starting a sandbox
- setup commands
- verifier commands
- artifact collection if expensive
- teardown/archive if expensive

They cannot be treated as "free" work just because they are not interactive tool calls.

### 5. The result worker must stay cheap

Inline in the result path is acceptable only for:

- deterministic string/numeric scoring
- small JSON serialization
- metadata writes

It is not acceptable for:

- long verifier runs
- big repo operations
- heavy sandbox setup


## Why this is the right split

This split solves the actual problem the repo is running into.

The current code drifted toward:

- provider module owns prompting
- provider tool config owns REPL choice
- Daytona is a special mode

That makes the backend feel more benchmark-specific and provider-specific than it should be.

The better model is:

- benchmark adapter writes the common task rows
- user chooses subject model
- user chooses judge
- user chooses tool runtime
- orchestrator runs the same task loop

So the intended public mental model is:

1. pick benchmark
2. adapt it into the FastEval schema
3. choose provider
4. choose judge
5. choose tool runtime
6. run

If the schema is right and the runtime abstraction is right, the rest should not need benchmark-specific branches.

This is also the concrete fix for the current Terminal Bench failure mode:

- prompt instructions should come from the benchmark contract
- bash cwd should come from the benchmark contract
- Python write redirection should come from the benchmark contract
- artifact syncing should come from the benchmark contract

`kwbench` keeps its current behavior because its contract is artifact-dir-first.
Terminal Bench gets workspace-first behavior because its contract says so.


## What the current repo already has

The current codebase already contains useful pieces of FastEval.

### Present today

- async orchestration with resumable runs
- bounded eval concurrency
- bounded tool concurrency
- provider abstraction for multiple model backends
- remote tool execution via Daytona
- artifact syncing
- judge provider selection via config
- result/meta persistence
- service API for running and monitoring jobs

### Partially present, but not first-class yet

- a common execution schema
- scoring selected from task content
- deterministic scoring as a peer of judge scoring
- benchmark-independent result semantics

Key files:

- [service/api.py]( service/api.py)
- [service/engine.py]( service/engine.py)
- [eval/tools.py]( eval/tools.py)
- [eval/core.py]( eval/core.py)
- [runner.py]( runner.py)


## What it is not yet

Today’s repo is still benchmark-shaped around the current task format.

That means it is not yet a clean public “run any benchmark” system.

Current assumptions still baked in:

- task rows look like `id/task/reference_files/rubric/output_dir/config`
- prompt construction is document-centric
- artifact handling is still too tightly coupled to workspace handling
- judge flow is still too central relative to deterministic scorers
- provider modules own too much benchmark/runtime logic together
- API allows only one active run

That is enough for the current benchmark family.
It is not enough for a general benchmark platform.


## Distance to a Harbor-like platform

If “Harbor” means a public benchmark runtime that can host many benchmark types, many model backends, and many scoring modes on the same substrate, FastEval is directionally aligned but not there yet.

The gap is not mostly in remote execution anymore.
The gap is in making the common schema and scoring model explicit.

### Close to the target

- async orchestration model
- remote execution backend
- explicit concurrency controls
- resumability
- provider indirection
- result persistence

### Still missing

- first-class benchmark plugin interface
- first-class tool runtime/session interface
- first-class execution contract
- benchmark-native terminal/session model
- benchmark setup/teardown hooks
- hidden-test and verifier workflow
- multi-run and multi-tenant API model
- stronger public-facing auth, quotas, and sandbox cleanup guarantees
- benchmark packaging/versioning
- canonical result schema across benchmark types

The first important missing piece is especially this:

- deterministic ground-truth scoring should be a first-class default path when `ground_truth` exists
- verifier-native scoring should be a first-class path for terminal and coding benchmarks

The shortest honest summary:

FastEval is already a plausible execution core for a Harbor-like system.
It is not yet the platform boundary itself.


## What would make it genuinely generic

### 1. Benchmark plugins

Introduce a benchmark contract like:

- `load_cases()`
- `build_case_context(case)`
- `build_prompt(case, context)`
- `allowed_tools(case)`
- `score_case(case, answer, artifacts)`
- `summarize_run(results)`

That would move benchmark-specific logic out of [eval/core.py]( eval/core.py) and [service/engine.py]( service/engine.py).

### 2. Separate scoring from judging

Scoring modes should include:

- exact match
- normalized exact match
- unit test pass/fail
- execution-based checker
- rubric LLM judge
- hybrid scorer

Right now the rubric judge path is useful, but too central.
The desired default is:

- no judge if deterministic scoring is available
- judge only when the task actually requires semantic evaluation

### 3. Proper tool runtime

For environment-heavy benchmarks, the core abstraction is not "run one bash command."
It is:

- provision a case-local execution session
- resolve a benchmark-defined execution contract
- keep state across tool calls
- suspend active compute between turns when possible
- capture filesystem outputs
- optionally run a verifier in the same session

That is broader than a REPL abstraction and narrower than copying another framework's full environment model.

This is why terminal and coding benchmarks are a useful portability test.
If FastEval can express them via task schema + benchmark adapter + tool runtime, then the abstraction is becoming real.

### 3a. Explicit workspace and artifact policy

Benchmarks must be able to say, independently:

- where the model should work
- what path instructions appear in the prompt
- where Python writes are redirected, if anywhere
- whether Python writes are constrained to the cwd subtree
- what paths get synced back as artifacts

Without that split, `output_dir` becomes a hidden global default and non-kwbench tasks get distorted.

### 4. Benchmark lifecycle hooks

Need hooks like:

- `prepare_case`
- `materialize_repo`
- `seed_environment`
- `finalize_case`
- `collect_artifacts`
- `cleanup_case`

Without these, many coding and terminal benchmarks remain awkward.

### 5. Public-service hardening

If published as a platform, not just a repo:

- auth
- quota enforcement
- per-user isolation
- cleanup of orphaned sandboxes
- stronger observability
- cost controls
- configurable retention


## Where Daytona fits

Daytona is the right tool-runtime implementation for FastEval when the benchmark needs:

- Python execution
- shell execution
- filesystem artifacts
- isolation
- high concurrency without local CPU saturation

It is especially valuable when the host should mostly orchestrate and the expensive execution should happen elsewhere.

With Daytona enabled, the local machine stops being the main compute bottleneck for tool calls.
That does not make the whole system “purely I/O bound” in a strict sense, but it does shift the dominant host-side work toward:

- network wait
- serialization
- disk I/O
- bookkeeping

So the right way to think about Daytona is:

- the model call is remote I/O
- the Daytona tool call is also remote I/O from the host's perspective
- the host mostly coordinates between those two systems
- the actual compute pressure moves off the local machine

That is the right direction for a high-throughput benchmark runner.

The important architectural point is:

- Daytona should be one runtime backend
- not the abstraction itself

So the target is not:

- "support Daytona mode"

It is:

- "support a runtime interface, with Daytona as one backend"


## Example: TAU-bench on Daytona

TAU-bench is a good example of why the execution/runtime split matters.

Conceptually, a TAU-bench adapter would do this:

### Dataset adaptation

The adapter converts TAU-bench rows into the FastEval schema:

- `id`
- `task`
- `reference_files` if any
- `config`
- `output_dir`
- optional `ground_truth` or `verifier` depending on the task
- optional `allowed_tools`
- optional `environment` / `session` hints

This is benchmark-specific and user-written.

### Model and judge choice

The user chooses:

- subject model, for example a Nebius-hosted model
- judge, if the task needs one

This is independent of the execution backend.

### Runtime choice

The user chooses:

- `runtime: local`
- or `runtime: daytona`

If the benchmark task needs tools, the orchestrator obtains a `ToolSession` from the chosen runtime.

The benchmark contract still controls:

- prompt instructions
- workspace root
- bash/Python cwd
- artifact sync policy

### Execution flow

For a TAU-bench task on Daytona:

1. the case is loaded into the common schema
2. the model receives the task
3. if the model asks for tools, the Daytona-backed session is started or resumed
4. Python/bash/file operations run in the sandbox
5. outputs are checkpointed to the sandbox filesystem
6. the sandbox is stopped between turns
7. when the task ends, artifacts are synced back
8. scoring happens using deterministic, verifier, or judge logic

The important thing is that the model I/O path is unchanged.
Only the tool runtime implementation changes.

### Why Daytona is a good fit here

TAU-bench-like tasks often want:

- filesystem state
- command execution
- optional setup
- optional verification
- many tasks in flight without pinning local CPU

Daytona fits that well because the session can exist without active host-side compute pressure, and the sandbox only needs to be live while tool work is actually happening.

### What the user would write

The benchmark author should only need to provide:

- the adapter that converts TAU-bench into the FastEval task schema
- optional task-specific prompt formatting
- optional scoring/verifier logic if the benchmark needs more than plain `ground_truth`

They should not need to rewrite:

- provider clients
- scheduler
- concurrency model
- runtime lifecycle policy

That is the point of the abstraction.


## First portability test

The first serious test of the FastEval spec should be Terminal Bench.

Not because it is the only important benchmark, but because it forces the right questions:

- can tasks be represented in the common schema
- can setup/environment/session requirements live in that schema cleanly
- can execution stay generic
- can scoring use verifier-native logic instead of rubric-by-default
- can artifacts and state be captured consistently

If Terminal Bench only works by adding benchmark-specific branches all over the runtime, the abstraction is not done.
If it works mainly by schema plus plugin/scorer implementations, the FastEval direction is probably sound.


## A credible public positioning today

What could be claimed now:

> FastEval is an async, Daytona-backed evaluation harness for tool-using LLM benchmarks, with a common task schema, configurable scoring modes, configurable judge providers, and resumable high-concurrency runs.

What should not be claimed yet:

> FastEval can run any benchmark.

That second claim needs the schema/plugin/scorer/runtime split to be real.


## Suggested roadmap

### Phase 1: make the abstraction honest

- make the schema the explicit public contract
- define benchmark plugin interface
- define scorer interface
- make `ground_truth` scoring first-class
- make verifier scoring first-class
- move current kwbench logic into one plugin

### Phase 2: prove breadth

- add a math benchmark adapter with deterministic scoring
- add a terminal benchmark adapter with persistent shell sessions
- add a coding benchmark adapter with verifier hooks

### Phase 3: make it a platform

- multi-run API
- auth and quotas
- run registry
- benchmark registry
- cost and resource dashboards


## Bottom line

FastEval is already a good name if the goal is:

- more concurrent tasks
- remote tool execution
- explicit concurrency control
- benchmark execution as infrastructure

The repo is closer to “a strong FastEval core” than to “a finished Harbor competitor”.
The missing work is mostly around generic benchmark abstraction, not around the basic async/daytona execution model.
