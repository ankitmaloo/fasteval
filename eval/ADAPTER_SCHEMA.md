# Dataset Adapter Schema

This document is for the person or model writing a new benchmark adapter.

The adapter's job is simple:

1. load the source benchmark format
2. resolve each source row into a FastEval task dict
3. return `list[dict]` from `load_cases()`

FastEval does **not** infer benchmark structure automatically.
The adapter must emit the schema below.


## The Contract

Write a benchmark plugin in:

- `eval/benchmarks/<your_benchmark>.py`

It must implement:

```python
def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
    ...
```

Each returned dict is one resolved task row.

That resolved task row is the dataset schema FastEval consumes.


## Exact Task Schema

Only two fields are truly required by the engine:

- `id: str`
- `task: str`

Everything else is optional, but optional fields are how you tell FastEval:

- how to score
- how to prompt
- how to run tools
- how to prepare the runtime

### Required

```json
{
  "id": "unique_task_id",
  "task": "Prompt text shown to the subject model"
}
```

### Common optional fields

These are the main fields the codebase already knows how to consume.

```json
{
  "reference_files": ["relative/or/absolute/path.ext"],
  "config": {},
  "output_dir": "artifacts",
  "source": "benchmark_name",
  "category": "task_group",
  "max_steps": 10
}
```

### Scoring fields

These determine how the result is evaluated.

```json
{
  "rubric": {
    "mandatory": ["criterion 1", "criterion 2"],
    "good_to_have": ["criterion 3"],
    "ideal": ["criterion 4"]
  },
  "ground_truth": "42",
  "match_type": "exact",
  "case_sensitive": false,
  "normalize": true,
  "verifier": "optional benchmark-specific verifier metadata"
}
```

Use:

- `rubric` for judge scoring
- `ground_truth` for deterministic scoring
- `score_case()` on the plugin for custom/verifier scoring

### Runtime and execution fields

These describe what the benchmark needs from the runtime.

```json
{
  "work_dir": "/app",
  "setup_cmd": "mkdir -p /app",
  "verify_cmd": "python -m pytest /tmp/tests.py -q 2>&1; echo \"EXIT_CODE=$?\"",
  "verify_timeout": 180,
  "allowed_tools": ["bash", "code"],
  "artifact_paths": ["/app/output.txt"],
  "timeout": 900
}
```

The benchmark plugin can also implement:

- `build_prompt()`
- `allowed_tools()`
- `execution_profile()`
- `prepare_case()`
- `score_case()`

to interpret these fields.


## What The Code Actually Reads

This repo consumes resolved task rows in a few places:

- `service/engine.py`
  - scheduling, case creation, runtime context, scoring dispatch
- `eval/scorers.py`
  - built-in deterministic and rubric scorer resolution
- `eval/benchmarks/base.py`
  - plugin hooks
- `eval/core.py`
  - prompt construction

Important behavior:

- the engine preserves the original task row into the final output JSONL
- your plugin can stash benchmark-specific fields on the row
- those fields come back into `build_prompt()`, `prepare_case()`, `score_case()`, and `summarize_run()`

So the resolved task row is both:

- the execution schema
- the scoring schema


## Scoring Resolution

FastEval resolves scoring using this precedence:

1. plugin `score_case()` if it returns a score
2. `rubric` -> judge scorer
3. `ground_truth` -> deterministic scorer
4. otherwise unscored

This means your adapter should choose one of these patterns:

### Pattern A: Deterministic answer benchmark

Emit:

- `ground_truth`
- optional `match_type`

No judge needed.

### Pattern B: Judge-scored benchmark

Emit:

- `rubric`

Run with:

- `judge_enabled: true`
- `judge_provider: <provider alias>`

### Pattern C: Environment/verifier benchmark

Emit whatever metadata the benchmark needs, then implement:

- `score_case()`

This is the right pattern for terminal, coding, or stateful tasks.


## Example: GSM8K Resolved Row

This repo already has a real GSM8K plugin:

- `eval/benchmarks/gsm8k.py`

That plugin resolves each source row into roughly this schema:

```json
{
  "id": "gsm8k_0000",
  "task": "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast every morning ...",
  "ground_truth": "18",
  "source": "gsm8k",
  "category": "math",
  "config": {}
}
```

And the plugin behavior is:

- `allowed_tools()` returns `[]`
- `build_prompt()` asks for final answer after `####`
- `score_case()` extracts the final number and compares to `ground_truth`

So GSM8K is:

- dataset adapter: yes
- runtime env: no
- judge: no
- custom evaluator: yes, lightweight numeric extraction


## Example: TAU-Bench Resolved Row

There is no built-in TAU-bench plugin in this repo today.
This example shows what a TAU-bench-style adapter should resolve rows into.

If benchmark `X` is TAU-bench-like and needs:

- a runtime environment
- tool use
- a judge

then the resolved task row should look more like this:

```json
{
  "id": "tau_retail_0001",
  "task": "You are a retail support agent. Resolve the user's issue using the available tools and produce the correct final customer-facing response.",
  "source": "tau_bench",
  "category": "customer-support",
  "work_dir": "/app",
  "setup_cmd": "python /app/bootstrap.py",
  "allowed_tools": ["bash", "code"],
  "output_dir": "artifacts",
  "rubric": {
    "mandatory": [
      "The final answer resolves the user's request correctly",
      "The answer does not violate the policy constraints",
      "The answer is consistent with the tool results"
    ],
    "good_to_have": [
      "The response is concise and clearly written",
      "The agent explains the decision in customer-friendly language"
    ],
    "ideal": [
      "The response anticipates the next likely user question"
    ]
  },
  "config": {
    "enable_bash": true,
    "enable_code": true,
    "enable_search": false
  }
}
```

That is what "resolved" means in FastEval:

- source benchmark row in
- FastEval task dict out

The benchmark plugin can then also implement:

- `build_prompt()` to format the agent instructions properly
- `execution_profile()` to define workspace/output semantics
- `prepare_case()` if files or environment state must be materialized before turn 1
- `score_case()` if rubric scoring is not enough


## Example: Terminal Bench Resolved Row

This repo already has a real terminal-style benchmark plugin:

- `eval/benchmarks/terminal_bench.py`

Terminal Bench is the best built-in example of a benchmark that needs:

- a runtime environment
- tool use
- benchmark-specific workspace semantics
- benchmark-specific setup
- benchmark-specific verification

A resolved Terminal Bench task row looks roughly like this:

```json
{
  "id": "assign-seats",
  "task": "Help! I'm hosting a dinner party ... Write your final answer to '/app/results.txt'.",
  "difficulty": "easy",
  "category": "algorithms",
  "tags": ["csp"],
  "source": "terminal-bench",
  "work_dir": "/app",
  "setup_cmd": "sudo mkdir -p /app && sudo chmod 777 /app",
  "verify_cmd": null,
  "verify_timeout": 180,
  "max_agent_timeout_sec": 900,
  "run_tests_in_same_shell": true,
  "task_dir": "/tmp/terminal-bench/original-tasks/assign-seats",
  "has_dockerfile": true,
  "has_tests": true,
  "test_content": "from pathlib import Path\\n..."
}
```

This benchmark shows how tool/runtime-specific behavior is expressed.

### What the plugin does

The Terminal Bench plugin does not rely on the default behavior alone.
It implements several hooks:

- `allowed_tools()`
  - returns `["bash", "code"]`
- `execution_profile()`
  - tells the engine to use `/app` as the workspace
  - tells the prompt not to redirect outputs into the harness artifact directory
  - constrains Python writes to the working tree
- `prepare_case()`
  - materializes task files into the runtime
  - replays Dockerfile-style `COPY` and `RUN` steps before the first model turn
  - runs `setup_cmd`
- `score_case()`
  - uploads the verifier test
  - runs pytest in the same runtime session
  - turns verifier exit status into a score

That is the pattern to copy for any benchmark that needs real tools and environment state.

### Why this example matters

GSM8K shows:

- pure dataset mapping
- no tools
- no judge

Terminal Bench shows:

- dataset mapping
- tools
- runtime setup
- workspace contract
- custom verification

Together they show the two ends of the adapter model:

- simple answer benchmark
- stateful runtime benchmark


## Registering The Benchmark

Once the adapter file exists, register it in:

- `eval/config.yaml`

Example:

```yaml
benchmarks:
  gsm8k:
    module: eval/benchmarks/gsm8k.py
    class: GSM8KPlugin

  terminal_bench:
    module: eval/benchmarks/terminal_bench.py
    class: TerminalBenchPlugin

  tau_bench:
    module: eval/benchmarks/tau_bench.py
    class: TAUBenchPlugin
```

If you want a standalone scorer override:

```yaml
benchmarks:
  tau_bench:
    module: eval/benchmarks/tau_bench.py
    class: TAUBenchPlugin
    scorer: eval/scorers/tau_bench_reward.py:score
```


## If You Need A New Runtime Environment

If the benchmark works with an existing runtime:

- `local`
- `daytona`

then the adapter does not need runtime code.
Just choose `runtime_type` at run time.

If you need a new runtime backend, write it in:

- `eval/tools.py`

Specifically:

1. add a new `ToolSession` implementation
2. add a new `ToolRuntime` implementation
3. register it in `make_tool_runtime()`
4. ensure `make_tool_session()` can construct it

Today the concrete examples are:

- `LocalToolRuntime`
- `DaytonaToolRuntime`

That is the exact extension point for a new runtime provider.


## If The Benchmark Needs Runtime Setup

If the benchmark needs files, repos, or environment setup before the first model turn, implement:

- `prepare_case()` in your plugin file

This is where benchmark-specific runtime preparation belongs.

Use it for:

- uploading files into the runtime
- running setup commands
- materializing workspace state

If the benchmark needs post-run verification, implement:

- `score_case()`


## Provider And Judge Selection

Subject model and judge are not hard-coded in the plugin.
They are selected in config and at run time.

Provider aliases live in:

- `eval/config.yaml`

Example:

```yaml
provider: qwen3.5-nebius

runtime:
  type: daytona
  concurrency: 20

judge:
  provider: qwen3.5-nebius
```


## Run Commands

### Start the server

```bash
cd <repo-root>
set -a && source .env && set +a
source .venv/bin/activate
uvicorn service.api:app --host 0.0.0.0 --port 8000
```

### Run GSM8K

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "async",
    "client": "provider",
    "llm": "qwen3.5-nebius",
    "provider": "qwen3.5-nebius",
    "benchmark": "gsm8k",
    "runtime_type": "local",
    "judge_enabled": false,
    "eval_sem": 16,
    "cpu_sem": 1,
    "case_max_steps": 1,
    "output": "service/results/gsm8k-qwen35.jsonl"
  }'
```

### Run TAU-bench-style benchmark

Assuming you wrote:

- `eval/benchmarks/tau_bench.py`

and registered `tau_bench` in `eval/config.yaml`:

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "async",
    "client": "provider",
    "llm": "qwen3.5-nebius",
    "provider": "qwen3.5-nebius",
    "benchmark": "tau_bench",
    "runtime_type": "daytona",
    "judge_enabled": true,
    "judge_provider": "qwen3.5-nebius",
    "judge_sem": 8,
    "judge_criterion_workers": 12,
    "eval_sem": 32,
    "cpu_sem": 8,
    "case_max_steps": 12,
    "output": "service/results/tau-bench-qwen35.jsonl"
  }'
```

### Run Terminal Bench

This is the built-in example of a tool-using runtime benchmark:

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "async",
    "client": "provider",
    "llm": "qwen3.5-nebius",
    "provider": "qwen3.5-nebius",
    "benchmark": "terminal_bench",
    "dataset": "/tmp/terminal-bench",
    "runtime_type": "daytona",
    "judge_enabled": false,
    "eval_sem": 10,
    "cpu_sem": 8,
    "case_max_steps": 10,
    "output": "service/results/terminal-bench-qwen35.jsonl"
  }'
```

That run uses:

- benchmark adapter: `eval/benchmarks/terminal_bench.py`
- runtime backend: `eval/tools.py` via `runtime_type: "daytona"`
- no judge
- benchmark-specific verifier scoring through `score_case()`


## What An LLM Should Actually Do

If an LLM is creating a new benchmark integration, the minimum checklist is:

1. create `eval/benchmarks/<name>.py`
2. implement `load_cases()` to emit the resolved task schema in this doc
3. if needed, implement:
   - `build_prompt()`
   - `allowed_tools()`
   - `execution_profile()`
   - `prepare_case()`
   - `score_case()`
4. register the benchmark in `eval/config.yaml`
5. choose provider, judge, and runtime in the run request
6. only touch `eval/tools.py` if a new runtime backend is required

That is the exact onboarding contract.
