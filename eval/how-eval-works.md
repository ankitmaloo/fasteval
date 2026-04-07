# How the Eval System Works

## The pipeline

Every eval run goes through two phases:

```
Phase 1: Agent execution
  Tasks are queued → workers pull tasks → each task runs an agent loop
  (LLM call → optional tool call → LLM call → ... until done or max_steps)

Phase 2: Scoring
  Completed tasks enter the scoring queue → scored → written to output JSONL
  Scoring is either:
    - Deterministic (instant): exact match, regex, numeric comparison
    - LLM judge (slow): each rubric criterion gets its own judge LLM conversation
```

Both phases run concurrently — tasks that finish their agent loop enter scoring while other tasks are still running.

---

## Adding and running a new benchmark, step by step

This section walks through creating a new benchmark from scratch. Follow each step in order.

### Prerequisites

- The eval server is running: `uvicorn service.api:app --port 8000`
- You have a model provider configured in `eval/config.yaml` (see "Adding a provider" below)
- Environment variables for your provider API key are set

### Step 0: Understand what belongs where

If you want to run a benchmark `X` that needs:

- a runtime environment
- an inference model
- a judge model

then FastEval splits the work across a few explicit extension points.

**Where to write benchmark-specific code**

- Dataset adapter + benchmark behavior:
  Write `eval/benchmarks/<your_benchmark>.py`
- Benchmark registration:
  Add an entry under `benchmarks:` in `eval/config.yaml`
- Custom scoring logic:
  Usually implement `score_case()` in your benchmark plugin
  Optional standalone scorer override: point `benchmarks.<name>.scorer` at `path/to/file.py:function_name`

**Where to write runtime code**

- Tool/runtime backend implementations live in `eval/tools.py`
- Runtime selection is resolved in `service/engine.py`

**Where to choose inference and judge**

- Provider aliases live in `eval/config.yaml`
- The run request picks:
  - subject model via `provider`
  - judge model via `judge_provider`
  - runtime via `runtime_type`

So the mental model is:

1. benchmark plugin normalizes the dataset into FastEval tasks
2. runtime backend executes tools
3. provider solves the task
4. scorer or judge evaluates the result

If you only need a new benchmark, you usually do **not** touch `eval/tools.py`.
If you want a different runtime environment, that is when you edit `eval/tools.py`.

---

## Running benchmark X with runtime + inference + judge

Assume benchmark `X` needs:

- a custom dataset format
- bash/python tool use
- a runtime environment
- rubric judging

This is the exact flow.

### Step 1: Write the benchmark adapter

Create:

- `eval/benchmarks/x.py`

This file should:

- load the source benchmark format
- map each raw row into the FastEval task schema
- optionally define prompt/runtime/scoring behavior

At minimum, each emitted task row needs:

- `id`
- `task`

Usually, benchmark `X` will also emit some of:

- `reference_files`
- `rubric`
- `ground_truth`
- `work_dir`
- `setup_cmd`
- `verify_cmd`
- `allowed_tools`
- `output_dir`

This is the dataset preprocessing step. It is benchmark-specific on purpose.
FastEval does not try to infer this automatically.

### Step 2: Register the benchmark

Edit:

- `eval/config.yaml`

Add an entry under `benchmarks:`

```yaml
benchmarks:
  x:
    module: eval/benchmarks/x.py
    class: XBenchmarkPlugin
```

At runtime, the engine resolves this through:

- `eval/benchmarks/__init__.py`

That registry loads your benchmark plugin by name.

### Step 3: Decide whether default scoring is enough

You have three common options.

**Option A: Judge scoring**

Emit `rubric` on each task row.
Then run with:

- `judge_enabled: true`
- `judge_provider: <provider alias>`

The engine resolves rubric scoring through:

- `eval/scorers.py`
- `service/engine.py`

**Option B: Deterministic answer scoring**

Emit fields like:

- `ground_truth`
- `match_type`

Then no judge is required.

**Option C: Benchmark-specific evaluator**

Implement:

- `score_case()` in `eval/benchmarks/x.py`

This is the right place if benchmark `X` has:

- hidden tests
- file-based verification
- environment-dependent success criteria
- custom answer normalization

If you want the evaluator in a separate file, register it via:

```yaml
benchmarks:
  x:
    module: eval/benchmarks/x.py
    class: XBenchmarkPlugin
    scorer: eval/scorers/x_custom.py:score
```

### Step 4: Decide whether you need a new runtime backend

If benchmark `X` works with an existing runtime:

- `local`
- `daytona`

then you do not write runtime code.
You only choose the runtime in config or the API request.

If benchmark `X` needs a different execution environment, then write it in:

- `eval/tools.py`

Specifically:

1. add a new `ToolRuntime` implementation
2. add a matching `ToolSession` implementation
3. update `make_tool_runtime()` to recognize the new runtime type
4. if needed, update `make_tool_session()` arguments passed through from the engine

Examples:

- `LocalToolRuntime`
- `DaytonaToolRuntime`

That is the right extension point for:

- Docker-backed runtime
- SSH-backed runtime
- another remote sandbox provider

You should not encode a new runtime inside the benchmark plugin.
The benchmark should describe what it needs.
The runtime should implement how tool execution actually happens.

### Step 5: Decide whether the benchmark needs runtime setup

If the runtime already gives the benchmark what it needs, nothing more is required.

If benchmark `X` needs per-task setup, implement it in:

- `prepare_case()` inside `eval/benchmarks/x.py`

Use this for things like:

- uploading task files
- running setup commands before the first model turn
- materializing benchmark workspace state

This setup runs before the first model call for that case.

If benchmark `X` needs custom verification after the task finishes, put that in:

- `score_case()`

### Step 6: Pick the subject model and judge

Edit or reuse provider aliases in:

- `eval/config.yaml`

Then choose:

- subject model with `provider`
- judge model with `judge_provider`

These are provider aliases, not benchmark code.

Example:

```yaml
runtime:
  type: daytona
  concurrency: 20

judge:
  provider: qwen3.5-nebius
```

And in the run request:

```json
{
  "engine": "async",
  "client": "provider",
  "provider": "qwen3.5-nebius",
  "benchmark": "x",
  "judge_enabled": true,
  "judge_provider": "qwen3.5-nebius",
  "runtime_type": "daytona"
}
```

### Step 7: Run the benchmark

Once the adapter exists and the benchmark is registered, the engine flow is:

1. load benchmark `X` through `eval/benchmarks/x.py`
2. normalize rows into the FastEval task schema
3. build per-case runtime context in `service/engine.py`
4. create a tool session via `eval/tools.py`
5. run the subject model
6. execute tools in the selected runtime
7. score with:
   - `score_case()`, or
   - deterministic scorer, or
   - judge scorer
8. write result rows + run metadata

### Step 8: What a new user actually has to write

For most benchmarks, the minimum work is:

1. create `eval/benchmarks/x.py`
2. implement `load_cases()`
3. optionally implement:
   - `build_prompt()`
   - `allowed_tools()`
   - `execution_profile()`
   - `prepare_case()`
   - `score_case()`
4. register the benchmark in `eval/config.yaml`
5. choose runtime/provider/judge in the run config

Only write a new runtime in `eval/tools.py` if the existing runtime backends are not enough.

That is the intended onboarding model:

- benchmark authors write adapters
- runtime authors write backends
- eval operators choose provider, judge, and runtime in config

### Step 1: Decide what kind of benchmark you're building

There are three patterns:

**Pattern A — Q&A with known answers (like GSM8K)**
- You have questions and correct answers
- Model answers once, no tools needed
- Scoring is instant: compare model output to ground truth
- No judge LLM needed

**Pattern B — Open-ended with rubric judging (like KWBench)**
- You have tasks and rubrics defining what a good answer looks like
- Model may use tools (code execution, bash, search)
- Scoring is slow: a judge LLM evaluates each rubric criterion
- Needs a judge LLM configured

**Pattern C — Custom scoring**
- You have your own scoring logic (run a test suite, check a file, call an API)
- Implement `score_case()` on your plugin

### Step 2: Create the plugin file

Create `eval/benchmarks/<your_bench>.py`. Here are complete, copy-pasteable templates for each pattern.

**Pattern A — Q&A with known answers:**

```python
"""My Q&A benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.benchmarks.base import BaseBenchmarkPlugin
from eval.scorers import ScorerResult


class MyQABenchPlugin(BaseBenchmarkPlugin):
    name = "my_qa_bench"

    def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
        """Load your questions and answers.

        Return a list of dicts, each with at least:
          - "id": unique string
          - "task": the question text
          - "ground_truth": the correct answer string

        Optional fields:
          - "match_type": "exact" (default), "regex", "contains", or "set"
          - "case_sensitive": bool (default True)
          - "normalize": bool (default False, collapses whitespace)
          - "source": where this task came from
          - "category": grouping label
        """
        # Option 1: Load from a local file
        if dataset_path is not None and dataset_path.exists():
            import json
            return [json.loads(line) for line in dataset_path.open() if line.strip()]

        # Option 2: Download from HuggingFace
        from datasets import load_dataset
        ds = load_dataset("your-org/your-dataset", split="test")
        return [
            {
                "id": f"task_{i:04d}",
                "task": row["question"],
                "ground_truth": row["answer"],
            }
            for i, row in enumerate(ds)
        ]

    def allowed_tools(self, case: dict[str, Any]) -> list[str] | None:
        """Return [] to disable all tools. Model just answers the question."""
        return []

    def build_prompt(self, case: dict[str, Any], context: dict[str, Any] | None) -> str | None:
        """Optional: customize the prompt sent to the model.
        Return None to just send the task text as-is."""
        return (
            f"{case['task']}\n\n"
            "Think step by step. Write your final answer at the end."
        )

    def summarize_run(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Optional: compute aggregate stats shown in the meta file."""
        scored = [r for r in results if isinstance(r.get("scoring"), dict)]
        correct = sum(1 for r in scored if r["scoring"].get("score", 0) == 1.0)
        return {
            "total": len(results),
            "scored": len(scored),
            "correct": correct,
            "accuracy": round(correct / max(len(scored), 1), 4),
        }
```

**Pattern B — Open-ended with rubric judging:**

```python
"""My rubric-judged benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.benchmarks.base import BaseBenchmarkPlugin


class MyJudgedBenchPlugin(BaseBenchmarkPlugin):
    name = "my_judged_bench"

    def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
        """Load tasks with rubrics.

        Each rubric has three tiers:
          - "mandatory": must ALL pass for any score above 0 (weight: 40%)
          - "good_to_have": partial credit (weight: 35%)
          - "ideal": excellence markers (weight: 25%)

        Each criterion is a plain English string that the judge LLM
        evaluates as PASS or FAIL.
        """
        # Option 1: From a local JSONL file
        if dataset_path is not None and dataset_path.exists():
            import json
            return [json.loads(line) for line in dataset_path.open() if line.strip()]

        # Option 2: Hardcoded / generated
        return [
            {
                "id": "case_001",
                "task": "Write a go-to-market strategy for a B2B SaaS product targeting hospitals.",
                "rubric": {
                    "mandatory": [
                        "Identifies the specific buyer persona (CIO, CMO, procurement) and explains why",
                        "Proposes a realistic sales motion (not just 'do marketing')",
                        "Addresses the long hospital procurement cycle (6-18 months)",
                    ],
                    "good_to_have": [
                        "References regulatory constraints (HIPAA, FDA) that affect GTM",
                        "Proposes a pilot/proof-of-concept approach before full rollout",
                    ],
                    "ideal": [
                        "Quantifies the TAM with specific assumptions",
                        "Identifies the physician champion dynamic as key to adoption",
                    ],
                },
                "source": "my_dataset",
                "category": "strategy",
            },
        ]

    def summarize_run(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        scores = [
            r["eval"]["score"]
            for r in results
            if isinstance(r.get("eval"), dict) and isinstance(r["eval"].get("score"), (int, float))
        ]
        return {
            "avg_score": round(sum(scores) / max(len(scores), 1), 4),
            "scored_count": len(scores),
        }
```

**Pattern C — Custom scoring:**

```python
"""My custom-scored benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.benchmarks.base import BaseBenchmarkPlugin
from eval.scorers import ScorerResult


class MyCustomBenchPlugin(BaseBenchmarkPlugin):
    name = "my_custom_bench"

    def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
        return [
            {"id": "task_001", "task": "Write a Python function that sorts a list."},
        ]

    def score_case(
        self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None,
    ) -> ScorerResult | None:
        """Custom scoring logic. Return a ScorerResult or None to fall back to default."""
        # Example: check if the answer contains a valid Python function
        has_def = "def " in answer
        has_return = "return " in answer or "sort" in answer
        score = 1.0 if (has_def and has_return) else 0.0
        return ScorerResult(
            score=score,
            detail={"has_def": has_def, "has_return": has_return},
            method="my_custom_check",
        )
```

**Pattern D — Foreign schema (e.g. terminal bench, coding bench, anything non-standard):**

Your source data doesn't have to look like `{question, answer}`. The plugin's job is to translate whatever schema you have into task dicts. Only two fields are required by the engine: `id` and `task`. Everything else is yours — stash whatever you need on the dict and read it back in `build_prompt()`, `score_case()`, `allowed_tools()`, etc.

```python
"""Terminal benchmark — tasks have setup commands, environments, and verifier scripts."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from eval.benchmarks.base import BaseBenchmarkPlugin
from eval.scorers import ScorerResult


class TerminalBenchPlugin(BaseBenchmarkPlugin):
    name = "terminal_bench"

    def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
        """Transform your source schema into task dicts.

        The engine only reads "id" and "task". Everything else is opaque —
        the engine passes the full dict back to your plugin methods.
        """
        # Your source data might look like:
        #   {"instruction": "...", "setup": "apt install ...", "verify": "pytest tests/",
        #    "env": {"LANG": "C"}, "expected_files": ["output.txt"], "difficulty": "hard"}
        #
        # Transform it:
        import json
        raw = [json.loads(l) for l in open(dataset_path) if l.strip()]
        cases = []
        for i, row in enumerate(raw):
            cases.append({
                # Required by engine
                "id": row.get("id", f"term_{i:04d}"),
                "task": row["instruction"],

                # Your custom fields — engine ignores these, plugin reads them
                "setup_cmd": row.get("setup", ""),
                "verify_cmd": row.get("verify", ""),
                "env": row.get("env", {}),
                "expected_files": row.get("expected_files", []),
                "difficulty": row.get("difficulty", "medium"),
                "time_limit": row.get("time_limit", 120),
            })
        return cases

    def allowed_tools(self, case: dict[str, Any]) -> list[str] | None:
        """Terminal bench uses bash only."""
        return ["bash"]

    def build_prompt(self, case: dict[str, Any], context: dict[str, Any] | None) -> str | None:
        """Use your custom fields to build the prompt."""
        parts = [case["task"]]
        if case.get("expected_files"):
            files = ", ".join(case["expected_files"])
            parts.append(f"\nYour solution should produce these files: {files}")
        if case.get("env"):
            env_str = ", ".join(f"{k}={v}" for k, v in case["env"].items())
            parts.append(f"\nEnvironment: {env_str}")
        return "\n".join(parts)

    def score_case(
        self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None,
    ) -> ScorerResult | None:
        """Run the verifier command to score."""
        verify_cmd = case.get("verify_cmd")
        if not verify_cmd:
            return None  # fall back to default scoring

        output_dir = artifacts.get("output_dir", ".") if artifacts else "."
        try:
            result = subprocess.run(
                verify_cmd, shell=True, capture_output=True, text=True,
                timeout=case.get("time_limit", 120), cwd=output_dir,
            )
            passed = result.returncode == 0
            return ScorerResult(
                score=1.0 if passed else 0.0,
                detail={
                    "returncode": result.returncode,
                    "stdout": result.stdout[:500],
                    "stderr": result.stderr[:500],
                },
                method="verifier",
            )
        except subprocess.TimeoutExpired:
            return ScorerResult(score=0.0, detail={"error": "timeout"}, method="verifier")

    def summarize_run(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        scored = [r for r in results if isinstance(r.get("scoring"), dict)]
        passed = sum(1 for r in scored if r["scoring"].get("score", 0) == 1.0)
        # Use your custom fields for richer summaries
        by_difficulty = {}
        for r in results:
            diff = r.get("difficulty", "unknown")
            by_difficulty.setdefault(diff, {"total": 0, "passed": 0})
            by_difficulty[diff]["total"] += 1
            if isinstance(r.get("scoring"), dict) and r["scoring"].get("score", 0) == 1.0:
                by_difficulty[diff]["passed"] += 1
        return {
            "pass_rate": round(passed / max(len(scored), 1), 4),
            "by_difficulty": by_difficulty,
        }
```

The key idea: **the task dict is yours**. Put `setup_cmd`, `verify_cmd`, `env`, `expected_files`, `difficulty`, `time_limit` — whatever your benchmark needs. The engine carries the dict through the pipeline untouched. Your plugin reads those fields back in `build_prompt()`, `score_case()`, and `summarize_run()`.

The engine only looks at:
- `id` — to track the task
- `task` (or `prompt`) — the text sent to the model
- `rubric` — if present and `judge_enabled`, triggers LLM judging
- `ground_truth` — if present, triggers deterministic scoring
- `config` — passed to the provider module (tool flags, repl settings)

Everything else passes through opaquely.

**Critical: `build_prompt()` is not optional for rich schemas.** By default, the engine sends only the `task` string to the model. None of your custom fields (`setup_cmd`, `env`, `expected_files`, etc.) reach the model unless `build_prompt()` assembles them into the prompt. If you skip `build_prompt()`, the model sees only the bare question and has no idea about setup commands, environments, file requirements, or any other context your benchmark provides. For simple Q&A benchmarks where `task` contains everything, `build_prompt()` is truly optional. For anything richer, it's required.

### Step 3: Register in config.yaml

Open `eval/config.yaml` and add your benchmark under the `benchmarks:` section:

```yaml
benchmarks:
  # ... existing benchmarks ...

  my_qa_bench:
    module: eval/benchmarks/my_qa_bench.py
    class: MyQABenchPlugin
```

The `module` path is relative to the repo root. `class` is the exact class name in that file.

### Step 4: Run it

Start the server (if not already running):

```bash
# Make sure your API keys are in the environment
source .env  # or export NEBIUS_API_KEY=... etc.
uvicorn service.api:app --host 0.0.0.0 --port 8000
```

**For Pattern A (Q&A, no judge):**

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "client": "provider",
    "llm": "qwen3.5-nebius",
    "provider_config_path": "eval/config.yaml",
    "benchmark": "my_qa_bench",
    "judge_enabled": false,
    "case_max_steps": 1,
    "eval_sem": 64
  }'
```

**For Pattern B (rubric judge):**

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "client": "provider",
    "llm": "qwen3.5-nebius",
    "provider_config_path": "eval/config.yaml",
    "benchmark": "my_judged_bench",
    "judge_enabled": true,
    "judge_provider": "gemini",
    "judge_sem": 4,
    "judge_criterion_workers": 15,
    "case_max_steps": 3,
    "eval_sem": 32,
    "cpu_sem": 4
  }'
```

**To limit to specific tasks:**

Add `"task_ids": ["task_001", "task_002", "task_003"]` to the request body.

**To save output to a specific file:**

Add `"output": "service/results/my_run.jsonl"` to the request body.

### Step 5: Monitor progress

```bash
curl http://localhost:8000/eval/status | python3 -m json.tool
```

Key things to look for:
- `status`: `running` or `completed`
- `completed` / `total`: how far along
- `ok` / `failed`: successes vs errors
- `current_in_flight_judges`: should be > 0 if judging is active

### Step 6: Read results

Results are in the JSONL file. Each line is one task:

```bash
# Quick summary
python3 -c "
import json
rows = [json.loads(l) for l in open('service/results/my_run.jsonl')]
for r in rows[:5]:
    s = r.get('scoring', {})
    print(f'{r[\"id\"]}: score={s.get(\"score\", \"?\")} method={s.get(\"method\", \"?\")}')
"
```

The meta file has run-level stats:

```bash
cat service/results/my_run.meta.json | python3 -m json.tool
```

### What can go wrong

| Symptom | Cause | Fix |
|---|---|---|
| `0/0 tasks` | Plugin's `load_cases()` returned empty list, or `task_ids` filter matched nothing | Check your plugin loads data correctly. Check IDs match. |
| All tasks `failed` with 401 | API key not in server's environment | Source `.env` before starting uvicorn |
| All tasks `failed` with 429 | Rate limit exceeded | Lower `eval_sem`. If judging, lower `judge_sem × judge_criterion_workers`. |
| Judging is extremely slow | `judge_criterion_workers` too low | Set to 15 (matches max criteria per task). See "Concurrency model" below. |
| `status: error` with S3 message | `S3_BUCKET` set but not accessible | Unset `S3_BUCKET` if you don't need remote persistence |
| `FileNotFoundError` on output | Run had 0 tasks, no output file created | Check task loading and filtering |

---

## Adding a provider

To use a new model, add it to the `providers:` section in `eval/config.yaml`.

For OpenAI-compatible APIs (most common):

```yaml
providers:
  my-model:
    llm: eval/llms/oaichat.py
    base_url: https://api.example.com/v1/
    api_key: ${MY_API_KEY}
    model: org/model-name
```

The `${MY_API_KEY}` syntax reads from environment variables. Set the variable before starting the server.

For native providers (Anthropic, Google, OpenAI):

```yaml
providers:
  claude:
    llm: eval/llms/claude.py
  gemini:
    llm: eval/llms/gemini.py
  openai:
    llm: eval/llms/openai.py
```

Then use `"llm": "my-model"` in the API request.

---

## Concurrency model

Four semaphores control all parallelism. Getting these wrong doesn't cause errors — it causes the run to be 10x slower than it should be.

### `eval_sem` — how many tasks run at once

Default: 64. This is the main throughput knob.

Each task holds one slot for its entire agent loop (all turns). 32 tasks with eval_sem=32 means all 32 run concurrently. 100 tasks with eval_sem=32 means 32 at a time, rest queued.

Set to match your API rate limit headroom.

### `cpu_sem` — how many tool calls run at once

Default: min(8, cpu_count). Only matters for agentic benchmarks that use tools (code execution, bash). Irrelevant for pure Q&A benchmarks.

Within each task's agent loop, a tool call must acquire this before executing. Prevents CPU/process exhaustion.

### `judge_sem` — how many cases are being judged at once

Default: 4. After a task finishes its agent loop, it enters the judge queue. This controls how many cases can be in the judging phase simultaneously.

Increase this to drain the judge backlog faster. Decrease if hitting rate limits on the judge endpoint.

### `judge_criterion_workers` — how many criteria per case run in parallel

Default: 4. **This is per-case, not global.**

A typical task has 10-15 rubric criteria. Each criterion is a separate judge LLM conversation. This controls how many of those conversations run in parallel for a single case.

With `judge_criterion_workers=4` and 15 criteria: criteria are batched 4 at a time. Case takes 4x longer than necessary.

With `judge_criterion_workers=15` and 15 criteria: all run at once. Case finishes as fast as the slowest criterion.

**Rule of thumb:** set this to the max number of criteria any task in your benchmark has. For kwbench, that's 15.

### How they compose

```
eval_sem=32, cpu_sem=4, judge_sem=4, criterion_workers=15

Agent phase:
  32 tasks concurrently
    each: LLM → tool (acquires cpu_sem) → LLM → done
    up to 4 tool executions at any instant across all 32 tasks

Judge phase:
  4 cases being judged concurrently
    each: 15 criteria in parallel, each criterion = 1 judge LLM call
    peak concurrent judge LLM calls = 4 × 15 = 60

Total peak LLM calls = 32 (agent) + 60 (judge) = 92
```

### Recommended settings by benchmark type

| Benchmark type | eval_sem | cpu_sem | judge_sem | criterion_workers | case_max_steps |
|---|---|---|---|---|---|
| Q&A, no tools (gsm8k) | 64 | 1 | — | — | 1 |
| Q&A with judge | 32 | 1 | 4 | 15 | 1 |
| Agentic with tools | 32 | 4 | — | — | 3 |
| Agentic with tools + judge (kwbench) | 32 | 4 | 4 | 15 | 3 |

---

## What `case_max_steps` means

Each "step" is one LLM call. If the model requests a tool, the tool runs and the result is fed back as the next turn. The loop continues until the model stops requesting tools or max_steps is reached.

- `1` — one LLM call, no tools possible. Use for Q&A benchmarks.
- `3` — model can do: call → tool → call → tool → call. Three LLM calls, two tool calls max. Standard for agentic tasks.
- If exceeded, task is marked `error: max_steps_exceeded`.

## What `judge_provider` means

The model that evaluates rubric criteria. Completely independent from the model being evaluated (`llm`). You can evaluate GPT with Gemini as judge, or use the same model as both.

Set it to any provider name from config.yaml. Defaults to `judge.provider` in config.yaml (usually `gemini`).

## Scoring modes

The system picks a scoring method based on what fields the benchmark puts on each task:

| Task has | Scoring | Needs judge? |
|---|---|---|
| `rubric` + `judge_enabled: true` | LLM evaluates each criterion PASS/FAIL | Yes |
| `rubric` + `judge_enabled: false` + `ground_truth` | Falls back to deterministic | No |
| `ground_truth` (no rubric) | Deterministic: exact/regex/contains/set match | No |
| Plugin returns `score_case()` result | Plugin-defined (e.g. GSM8K numeric extraction) | No |
| Nothing | Unscored | No |

Rubric scoring formula:
- All mandatory must pass → base 40%
- `+ 35% × (fraction of good_to_have passed)`
- `+ 25% × (fraction of ideal passed)`
- If any mandatory fails → score is 0

---

## Output

Results go to `service/results/<name>.jsonl` (one JSON row per task) and `service/results/<name>.meta.json` (run metadata updated every 5 seconds).

### Checking progress

```bash
# Live status
curl http://localhost:8000/eval/status

# List all runs
curl http://localhost:8000/eval/runs

# Get full results for a specific run
curl http://localhost:8000/eval/runs/testnew
```

### Meta file

Key fields in `*.meta.json`:

- `status`: `running`, `completed`, or `error`
- `completed` / `total`: progress
- `ok` / `failed`: success vs error count
- `current_in_flight_evals`: tasks in agent phase right now
- `current_in_flight_judges`: cases being judged right now
- `peak_in_flight_*`: max concurrency seen during the run
- `benchmark_summary`: plugin-provided summary (accuracy, avg_score, etc.)
- `heartbeat_at`: updates every 5s while running — if this stops advancing, the run is stuck

### Result rows

Each line in the JSONL is one completed task:

- `id`: task ID
- `task`: the question/prompt
- `llm_answer`: the model's response
- `status`: `ok` or `error`
- `error`: error message if failed
- `scoring.method`: which scorer ran (`rubric_judge`, `exact_match`, `gsm8k_numeric`, etc.)
- `scoring.score`: 0.0 to 1.0
- `scoring.detail`: method-specific breakdown
- `eval`: legacy rubric judge detail (only for rubric_judge — has `mandatory`, `good_to_have`, `ideal` boolean arrays)
- `metrics.total_s`: wall time for this task
- `metrics.model_wait_s`: LLM call time
- `metrics.tool_cpu_s`: tool execution time
- `llm_metadata`: token usage, cost, turn count

---

## All API fields reference

### `POST /eval/start`

| Field | Type | Default | Description |
|---|---|---|---|
| `client` | string | `"fake"` | `"provider"` for real models, `"fake"` for simulation, `"replay"` for fixture playback |
| `llm` | string | required | Provider name from config.yaml (e.g. `"qwen3.5-nebius"`, `"gemini"`, `"claude"`) |
| `provider_config_path` | string | `"eval/config.yaml"` | Path to config file |
| `benchmark` | string | null | Benchmark plugin name from config.yaml `benchmarks:` section |
| `dataset` | string | null | Explicit dataset file path. null = plugin decides |
| `task_ids` | list | null | Filter to specific task IDs. null = all tasks |
| `output` | string | null | Output path. null = auto-generated |
| `eval_sem` | int | 64 | Max concurrent tasks in agent phase |
| `cpu_sem` | int | 8 | Max concurrent tool executions |
| `case_max_steps` | int | 3 | Max agent turns per task |
| `max_retries` | int | 2 | Retries on transient LLM errors |
| `judge_enabled` | bool | false | Enable rubric judging |
| `judge_provider` | string | config default | Which model judges rubric criteria |
| `judge_sem` | int | 4 | Max cases being judged concurrently |
| `judge_criterion_workers` | int | 4 | Max criteria evaluated in parallel per case |
| `repl_mode` | string | `"local"` | `"local"` or `"daytona"` for tool execution |
| `wandb_project` | string | null | W&B project for logging |

### `GET /eval/status`

Returns the current run's meta.json content. Poll this to monitor progress.

### `GET /eval/runs`

Returns list of all run metadata.

### `GET /eval/runs/{name}`

Returns full results + stats for a specific run.

### `POST /eval/resume`

Resume an interrupted run. Requires `"output": "path/to/existing.jsonl"`. Skips already-completed tasks.
