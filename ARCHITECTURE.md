# Architecture

This is a parallel eval runner. It takes a dataset of tasks, sends each to an LLM, optionally judges the output with a separate model, and writes scored results to JSONL.

## Files

```
runner.py              # Generic async orchestrator. Cases in, results out.
clients.py             # LLM client abstraction (fake, replay, provider)
service/engine.py      # Glue: loads data, builds clients, wires judge, persists results
service/api.py         # FastAPI server wrapping engine.py
eval/core.py           # Judge logic (Gemini-based) + sync eval runner (legacy)
eval/tools.py          # PythonREPL + bash execution with concurrency tracking
eval/storage.py        # S3 upload/download, background thread pool
eval/llms/*.py         # LLM modules (openai.py, oaichat.py, etc.) — each exports generate() + LLM_ID
eval/config.yaml       # Provider configs (base_url, api_key, model per provider name)
```

## Data flow

```
dataset.jsonl
    │
    ▼
engine.py loads tasks, filters by task_ids, skips already-completed (resume)
    │
    ▼
engine.py builds LLMClient (fake / replay / provider)
    │
    ▼
runner.py runs all cases concurrently (capped by eval_sem)
    │  each case: prompt → LLM → optional tool call → LLM → ... → final text
    │
    ▼
on_case_complete callback puts CaseResult into result_queue
    │
    ▼
result workers (1 if no judge, judge_sem if judge enabled):
    │  build result row
    │  optionally run judge (Gemini evaluates each rubric criterion)
    │  write row to JSONL
    │  update meta.json
    │  sync to S3
    │  log to W&B
    │
    ▼
output.jsonl + output.meta.json
```

## The two runner paths

**Async (default):** `service/engine.py` → `runner.py`. This is the main path. Everything below describes this path.

**Legacy (sync):** `service/api.py` with `engine=legacy` → `eval/core.py` `run_eval()`. Uses `ThreadPoolExecutor` instead of asyncio. Kept for backward compat. Not covered further.

## How parallelism works

There are three layers of concurrency, each independently capped:

### Layer 1: Eval concurrency (`eval_sem`)

How many tasks run against the LLM simultaneously. Controlled by `asyncio.Semaphore(eval_sem)`.

```
runner.py creates eval_sem worker tasks (asyncio.create_task)
each worker pulls from a shared queue and runs one case at a time
a case holds the semaphore for its entire lifetime (LLM call + tool calls)
```

Default: 64. For rate-limited APIs, lower this.

### Layer 2: Tool concurrency (`cpu_sem`)

How many tool calls (code execution, bash) run simultaneously. Controlled by `asyncio.Semaphore(cpu_sem)` in the runner, plus `threading.BoundedSemaphore` in `eval/tools.py` for provider clients.

**Two different mechanisms depending on client type:**

- **fake/replay clients:** The runner's built-in tool loop fires. Tools run in a `ProcessPoolExecutor(cpu_sem)` — actual subprocess isolation. The runner's `_cpu_sem` asyncio semaphore gates access.

- **provider clients:** The LLM module's `generate()` runs tool calls internally (the runner never sees them). Tool concurrency is tracked via `eval/tools.py`'s `_tool_sem` (a threading BoundedSemaphore set to `cpu_sem`). The runner's ProcessPoolExecutor exists but sits idle (no tasks submitted to it).

### Layer 3: Judge concurrency (`judge_sem`)

How many completed tasks are judged simultaneously. Only active when `judge_enabled=True`. Controlled by `asyncio.Semaphore(judge_sem)`.

Each judge task also spawns a `ThreadPoolExecutor(criterion_workers)` internally to evaluate rubric criteria in parallel.

## Client types

All implement `LLMClient.complete(messages) -> Response`.

**FakeLLMClient** — Deterministic, no API calls. Sleeps to simulate latency. Tool/no-tool decision is hash-based (same case_id always produces the same behavior). Used for testing parallelism and correctness.

**ReplayLLMClient** — Reads responses from a JSON fixture file keyed by `(case_id, step)`. Deterministic, no API calls. Used for reproducible integration tests.

**ProviderLLMClient** — Wraps a real `eval/llms/*.py` module. The module's `generate(task, references, config)` function does everything: multi-turn LLM conversation, tool calls, code execution. The adapter runs it via `asyncio.to_thread` (or directly if `generate_async` exists) and always returns `tool_needed=False` — the runner sees it as a single-step case.

This is a key design point: **for provider clients, the runner's multi-step loop is a no-op.** All the agentic logic lives inside the LLM module's `generate()`. The runner just provides concurrency management.

## The provider client bridge

```
runner._run_case()
    │
    ▼
ProviderLLMClient.complete()
    │  looks up ProviderCaseContext for this case_id
    │  (task text, references, config dict with _repl instance)
    │
    ▼
asyncio.to_thread(module.generate, task, references, config)
    │  generate() runs in a thread pool thread
    │  it makes API calls, parses tool calls, runs code via config["_repl"]
    │  returns {"text": "...", "metadata": {...}}
    │
    ▼
Response(text=..., tool_needed=False)  ← always single-step
```

Each task gets its own `PythonREPL()` instance (separate variable namespace). But all REPL `run()` calls acquire a global `_repl_run_lock` because `os.chdir()` and `redirect_stdout()` are process-global.

## Judge pipeline

Judging happens after a task completes, in the result-processing stage — not in the runner.

```
CaseResult arrives in result_queue
    │
    ▼
_persist_result():
    │  build result row (task fields + metrics + llm_answer)
    │
    ▼
_maybe_judge():                              ← only if judge_enabled and status=="ok"
    │  acquire judge_sem
    │  asyncio.to_thread(_judge_task_with_rubric)
    │      │
    │      ▼  (runs in a thread)
    │      _build_judged_answer()             ← append output files to answer
    │      judge_rubric()                     ← from eval/core.py
    │          │
    │          ▼
    │          ThreadPoolExecutor(criterion_workers)
    │              each thread: judge_criterion() → Gemini API call
    │              returns True/False per criterion
    │      score_rubric(mandatory, good_to_have, ideal) → float
    │
    ▼
row["eval"] = {"mandatory": [...], "good_to_have": [...], "ideal": [...], "score": 0.75}
    │
    ▼
write row to JSONL (under write_lock)
update meta.json
sync to S3
log to W&B
```

The judge model is Gemini (`gemini-3-flash-preview`), hardcoded in `eval/core.py`. Each criterion gets its own PythonREPL so the judge can run code to verify answers (e.g., read a CSV to check if a number is correct).

## Scoring

```python
score_rubric(mandatory, good_to_have, ideal) -> float:
    if any mandatory fails: return 0.0
    base = 0.40
    base += 0.35 * (pass_rate of good_to_have)
    base += 0.25 * (pass_rate of ideal)
    return base  # max 1.0
```

## Resume

If the output JSONL already exists (from a previous run or S3 restore), the engine reads it, extracts completed task IDs, and only runs the remaining tasks. The original dataset indices are preserved so that `case_id` generation stays consistent.

## Result worker lifecycle

```
result_queue = asyncio.Queue()
N result_worker tasks = min(remaining_cases, judge_sem or 1)

runner.run_cases(on_case_complete=put_to_queue)
    ↓ all cases complete
result_queue.join()          ← wait for all results to be persisted/judged
    ↓
send N sentinel Nones        ← one per worker
asyncio.gather(workers)      ← wait for clean shutdown
wb_run.finish()
```

If a persist error occurs, subsequent results in the queue are drained (task_done called) but not written. The error is re-raised after the queue is drained.

## Thread safety summary

| Resource | Protection | Why |
|---|---|---|
| JSONL file writes | `asyncio.Lock` (`write_lock`) | Multiple result workers write concurrently |
| Meta file updates | Inside `write_lock` | Same |
| `os.chdir` + `redirect_stdout` in REPL | `threading.Lock` (`_repl_run_lock`) | Process-global state, races across threads |
| Tool in-flight counter | `threading.Lock` (`_metrics_lock`) | Multiple provider threads update concurrently |
| Judge in-flight counter | `asyncio.Lock` (`judge_lock`) | Multiple result workers track concurrently |
| S3 uploads | Snapshot-then-upload pattern | Avoids reading file while main loop writes to it |

## Threading model (worst case with provider + judge)

```
asyncio event loop
├── eval_sem runner workers (asyncio tasks)
│   └── asyncio.to_thread → provider generate() thread
│       └── may call PythonREPL.run() (acquires _repl_run_lock)
│
├── result workers (asyncio tasks, count = judge_sem)
│   └── asyncio.to_thread → _judge_task_with_rubric thread
│       └── ThreadPoolExecutor(criterion_workers)
│           └── judge_criterion threads → Gemini API + PythonREPL.run()
│
└── storage fire_and_forget → background ThreadPoolExecutor(2)
```

Max threads: `eval_sem` (for provider to_thread) + `judge_sem` (for judge to_thread) + `judge_sem * criterion_workers` (for criteria) + 2 (storage). With defaults (64 + 4 + 16 + 2 = 86).

## Config reference

| Parameter | Default | Controls |
|---|---|---|
| `eval_sem` | 64 | Max concurrent LLM tasks |
| `cpu_sem` | min(8, cpu_count) | Max concurrent tool calls |
| `judge_enabled` | false | Whether to run judge after each task |
| `judge_sem` | 4 | Max concurrent judge tasks |
| `judge_criterion_workers` | 4 | Threads per judge task for criteria |
| `max_retries` | 2 | LLM call retries on transient errors |
| `client` | fake | fake / replay / provider |
| `case_max_steps` | 3 | Max LLM↔tool rounds per case (fake/replay only) |

## Running

```bash
# Start server
uvicorn service.api:app --host 0.0.0.0 --port 8000

# Fire an eval (fake client, no tokens)
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"client":"fake","eval_sem":16}'

# Fire an eval (real provider with judge)
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"client":"provider","llm":"openai","eval_sem":64,"cpu_sem":8,"judge_enabled":true,"judge_sem":4}'

# Check status
curl http://localhost:8000/eval/status

# List runs
curl http://localhost:8000/eval/runs
```
