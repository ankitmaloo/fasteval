# Changes Log

## 2026-03-18

This document records the fixes merged from the separate review worktree into the main worktree, plus the follow-up changes made during integration.

### Scope

The changes covered:

- async API run lifecycle safety
- provider-mode memory retention
- REPL timeout enforcement
- artifact-directory confinement for generated files
- judge access to task artifacts
- regression test coverage

### 1. Single-run API guard race

Files:

- `service/api.py`
- `tests/test_api.py`

Problem:

- The API checked whether a run was active, released the lock, built the thread/config, then reacquired the lock to publish `_active`.
- Two overlapping `/eval/start` or `/eval/resume` requests could both pass the first check and launch competing runs.
- A finishing older thread could also clear `_active` after a newer run had already taken over.

Change:

- Added a per-run token.
- Reserved `_active` atomically under the lock before starting the background thread.
- Added token-checked cleanup so only the owning run can clear `_active`.

Rationale:

- The service is intended to allow only one run at a time.
- Without atomic reservation, the status endpoint and run ownership become unreliable under concurrent requests.
- Tokenized cleanup prevents stale background threads from corrupting newer run state.

### 2. Provider-mode memory retention

Files:

- `service/engine.py`
- `clients.py`
- `tests/test_runner.py`

Problem:

- Provider mode eagerly built per-case context for all remaining tasks before execution began.
- Each context could allocate references, metadata, and a dedicated `PythonREPL`.
- Completed-case context and metadata stayed resident for the full run.

Change:

- Switched provider case construction to lazy creation via `case_context_factory`.
- Added per-case release hooks so metadata and context state are dropped after result persistence.
- Stopped retaining full in-memory case state past completion.

Rationale:

- Memory usage should scale with in-flight work, not total dataset size.
- Large runs were paying a resident-memory cost for tasks that had not started yet and tasks that had already completed.
- Releasing state after persistence reduces the chance of long-run memory growth and makes concurrency knobs more predictable.

### 3. REPL timeout enforcement and isolation

Files:

- `eval/tools.py`
- `tests/test_engine.py`

Problem:

- `PythonREPL.run()` accepted a timeout but executed `exec()` inline and did not enforce the timeout.
- A long-running or infinite code cell could stall unrelated provider work.
- The earlier design serialized REPL execution too aggressively and made pauses hard to distinguish from scheduler issues.

Change:

- Reworked `PythonREPL` to run code in a dedicated worker process per REPL instance.
- Enforced timeout in the parent process.
- On timeout or worker failure, the worker is torn down and recreated on next use.
- Preserved state across successful calls inside the same REPL worker.

Rationale:

- Python cannot safely interrupt arbitrary running code inside the same thread.
- A subprocess boundary gives a practical way to enforce timeout and recover from hung cells.
- Isolating each REPL instance removes the prior shared failure mode where one stuck cell could effectively pause other work.

### 4. Async execution boundaries preserved

Files:

- `clients.py`
- `service/engine.py`
- `runner.py`

Problem:

- The goal of the fix set was not only correctness, but to avoid reintroducing event-loop blocking while fixing timeouts and memory usage.

Change:

- Sync provider `generate(...)` calls continue to run behind `asyncio.to_thread(...)`.
- Provider judging is invoked behind `asyncio.to_thread(...)`.
- Runner tool work remains governed by `asyncio` semaphores and the process pool path in `runner.py` / `tools.py`.
- Conversation flush remains off the event loop through `asyncio.to_thread(...)`.

Rationale:

- The async service should spend its time coordinating work, not executing blocking provider or judge logic on the loop thread.
- Preserving these boundaries keeps `eval_sem`, `cpu_sem`, and `judge_sem` meaningful and prevents hidden latency inflation in the event loop.

### 5. Artifact-directory confinement for provider output

Files:

- `eval/tools.py`
- `eval/core.py`
- `eval/llms/openai.py`
- `eval/llms/oaichat.py`
- `eval/llms/claude.py`
- `eval/llms/gemini.py`
- `service/engine.py`
- `tests/test_engine.py`

Problem:

- Earlier runs allowed models to create files in inconsistent locations.
- Some model prompts and provider adapters treated output directories ambiguously.
- Judge-side collection only looked at the top level of the artifact directory, so nested output could be missed.

Change:

- Standardized prompt language to say that all output files must be written only inside the provided path.
- Passed the resolved task artifact directory directly into provider prompts.
- Set provider bash execution cwd to the task artifact directory.
- Sandboxed Python REPL file writes so writes through `open` / `io.open` are redirected back into the configured artifact root.
- Made artifact cleanup recursive so stale nested output is removed before a rerun.
- Made judge-side output collection recursive so nested output files under the artifact root are included.

Rationale:

- The benchmark expects per-task output to live under a known artifact directory so judging and resume behavior are deterministic.
- Models frequently fail soft instructions; the Python REPL path now enforces the directory constraint instead of relying only on prompt compliance.
- Recursive cleanup and recursive collection are required for correctness once subdirectories are allowed inside the artifact root.

### 6. Judge REPL now receives the correct artifact context

Files:

- `service/engine.py`
- `eval/core.py`
- `tests/test_engine.py`

Problem:

- The judge already preloaded Excel files from the artifact directory, but its REPL working directory was not explicitly tied to that same task artifact root.
- That created an unnecessary mismatch between preloaded files and filesystem context.

Change:

- Passed the resolved task artifact directory into judge REPL creation.
- Updated the judge prompt to reflect that the REPL runs with the task artifact directory as cwd when present.
- Continued preloading Excel workbooks from that same artifact directory for direct inspection.

Rationale:

- The judge should evaluate against the exact files produced for the task, in the exact artifact location used by the eval.
- Aligning cwd, preloaded workbooks, and recursive artifact collection reduces ambiguity and makes judge behavior more predictable.

### 7. Regression coverage added

Files:

- `tests/test_api.py`
- `tests/test_engine.py`
- `tests/test_runner.py`

Coverage added for:

- token-guarded `_active` cleanup
- rejection of overlapping runs
- lazy provider context construction
- provider state release after completion
- enforced REPL timeout
- redirected escaped writes into artifact root
- recursive artifact discovery
- judge receiving the resolved artifact directory

Rationale:

- These fixes touch failure modes that are easy to regress silently.
- The new tests lock in the expected behavior around concurrency, filesystem handling, and judge wiring.

### Validation

Validated with the repo venv and Python:

- `PYTHONPATH=$(pwd) .venv/bin/python -m py_compile service/api.py service/engine.py eval/core.py eval/tools.py clients.py tests/test_api.py tests/test_engine.py tests/test_runner.py`
- `PYTHONPATH=$(pwd) .venv/bin/python -m pytest -q tests/test_api.py tests/test_engine.py tests/test_runner.py`

Latest result after the final judge/artifact integration pass:

- `108 passed`

### Notes

- These changes include both the fixes first developed in the separate review worktree and the follow-up integration fixes made directly in the main worktree.
- The strongest enforcement today is on Python REPL file creation. Bash is run from the artifact directory, but it is not a full filesystem jail.

## 2026-03-19

### 8. Judge artifact evidence consolidated into one bundle

Files:

- `eval/core.py`
- `service/engine.py`
- `tests/test_engine.py`

Problem:

- The async judge path was assembling evidence in three separate ways:
  - stitching inline artifact text into a synthetic `judged_answer`
  - preloading only some artifact types into the judge REPL
  - separately passing the artifact directory as the REPL cwd
- That meant the judge prompt, the preloaded variables, and the filesystem view were not derived from one canonical source.
- In practice this made file-based cases harder to reason about and increased the chance that a judge saw different evidence depending on whether it relied on prompt text or the REPL.

Change:

- Added a single artifact-bundle builder in `eval/core.py` that scans the task artifact directory once and produces:
  - `artifact_root`
  - `artifact_manifest`
  - deterministic prompt context for the judge
  - REPL seed globals
- The bundle now preloads and names artifact evidence consistently:
  - text-like artifacts as `artifacts_text[...]` plus `<name>_text`
  - parsed DOCX text/tables as `artifacts_text[...]`, `artifacts_tables[...]`, `<name>_text`, and `<name>_tables`
  - Excel workbooks as `<name>_data` and `<name>_formulas`
- Updated the judge prompt so it explicitly references the artifact context block and the canonical preloaded variables.
- Updated both async judging (`service/engine.py`) and legacy judging (`eval/core.py`) to use the same artifact bundle instead of rebuilding evidence differently in each path.

Rationale:

- The judge should evaluate one deterministic evidence package, not a loose combination of answer text, ad hoc preload rules, and filesystem fallback.
- Using a single bundle keeps the prompt view and REPL view aligned, which makes file-based judging easier to debug and less sensitive to implementation details.
- Stable variable names such as `artifact_root`, `artifact_manifest`, `artifacts_text`, and `_preloaded_files` make it clearer what the judge can inspect.

Validation:

- `PYTHONPATH=$(pwd) .venv/bin/python -m py_compile service/engine.py eval/core.py eval/tools.py tests/test_engine.py`
- `PYTHONPATH=$(pwd) .venv/bin/python -m pytest -q tests/test_engine.py -k 'BuildJudgedAnswer or OutputArtifactHandling or JudgeTaskWithRubric or judge_task_passes_conv_store or provider_case_context or stale_openai_env'`
- `PYTHONPATH=$(pwd) .venv/bin/python -m pytest -q tests/test_async_eval.py -k 'judge_pipeline_runs_in_parallel_and_is_capped or judge_skips_cases_without_rubric or judge_requires_gemini_key'`
