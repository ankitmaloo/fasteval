# AGENTS.md

This repository contains two evaluation stacks:

1. Async service-first stack (current default)
2. Legacy eval/judge stack (kept for compatibility)

Read this before making changes.

## Mental Model

- The API server entrypoint is `service/api.py`.
- The async execution engine is `service/engine.py`.
- The provider modules (OpenAI, Claude, Gemini, etc.) are in `eval/llms/`.
- Core async primitives are in root files:
  - `runner.py` (dynamic scheduling + semaphores + process pool)
  - `clients.py` (LLM clients/adapters)
  - `tools.py` (process-pool tool execution helper)
- Legacy judging/eval path is in `eval/core.py` and invoked only with `engine="legacy"`.

## Current Runtime Modes

### Async mode (`engine="async"`)

- Default mode in API.
- Clients:
  - `fake` (no external API calls)
  - `replay` (fixture-driven deterministic runs)
  - `provider` (loads existing `eval/llms/*.py` modules)
- Uses dynamic case scheduling.
- Uses two concurrency controls:
  - `eval_sem`: max in-flight eval coroutines
  - `cpu_sem`: max concurrent tool/REPL heavy executions
- Writes JSONL results + `.meta.json` progress file.
- Supports resume (skip completed IDs).
- Supports optional S3 sync.
- Can auto-fetch dataset/reference files from HF dataset repo.

### Legacy mode (`engine="legacy"`)

- Uses `eval/core.py:run_eval`.
- Includes rubric scoring + judge-model calls.
- Judge criteria are evaluated in parallel threadpool calls.
- Kept for backward compatibility and judge-based scoring.

## Important File Map

- `service/api.py`:
  - FastAPI routes (`/eval/start`, `/eval/resume`, `/eval/status`, `/eval/runs`, `/health`)
  - Run lifecycle state (`_active`, `_last_meta_path`)
  - Thread handoff for background execution
- `service/engine.py`:
  - `AsyncRunConfig`
  - dataset loading + HF materialization
  - provider resolution (module name, path, config alias)
  - async run orchestration and result/meta writes
- `eval/tools.py`:
  - REPL and bash tool functions used by provider modules
  - shared tool concurrency gating and metrics for provider mode
- `eval/llms/*.py`:
  - synchronous provider modules with `LLM_ID` + `generate(task, references, config)`
- `eval/async_eval.py`, `eval/server.py`:
  - compatibility shims that re-export from `service/*`

## Start/Setup

### Local

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r eval/requirements.txt
uv run --python .venv/bin/python uvicorn service.api:app --host 0.0.0.0 --port 8000
```

### EC2

- Use root script: `./setup_ec2.sh`
- Script installs deps, materializes HF dataset, and registers systemd service.

## API Usage Examples

### Async fake (safe first check)

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"engine":"async","client":"fake","eval_sem":64,"cpu_sem":8}'
```

### Async provider (OpenAI module)

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"engine":"async","client":"provider","llm":"openai","eval_sem":64,"cpu_sem":8}'
```

### Legacy (judge path)

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"engine":"legacy","llm":"gemini"}'
```

## Environment Expectations

Use `.env` at repo root. Fill relevant keys for the mode you run.

- Provider keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc.
- Optional S3: `S3_BUCKET`, `AWS_*`
- Optional W&B: `WANDB_API_KEY`, `WANDB_PROJECT`

## Development Rules for This Repo

- Put new server/API code in `service/`, not `eval/`.
- Keep `eval/server.py` and `eval/async_eval.py` as compatibility shims unless intentionally removing legacy import paths.
- For provider-mode changes, avoid breaking `generate(task, references, config)` contract in `eval/llms/*`.
- Preserve resume semantics: never re-run completed IDs when output JSONL already contains them.
- Keep metadata updates incremental; status endpoint relies on `.meta.json`.

## Testing

- Main tests are under `tests/`.
- Run with local venv only:

```bash
PYTHONPATH=$(pwd) uv run --python .venv/bin/python pytest -q
```

- There are tests for:
  - scheduler/concurrency caps
  - 223-case completion
  - fake/replay behavior
  - provider + REPL + tool cap behavior
  - async dataset fetch materialization path

## Known Boundaries

- Async path currently does not run legacy rubric judge scoring.
- Legacy path still owns full judge scoring behavior.
- Only one run is allowed at a time via API guard.


# IMPORTANT
WHEN RUNNING, DO NOT GIVE OUT RELATIVE PATHS FOR OUTPUT DIRECTORIES. 