# Evals

Evaluation harness for LLM tasks with:
- Async runner (`asyncio` + process pool)
- FastAPI service endpoints for start/status/resume
- Fake/replay/provider execution modes
- Optional S3 persistence and crash-resume support

Detailed benchmark docs and provider notes live in `eval/README.md`.

## Repository Layout

- `service/api.py`: FastAPI app (`uvicorn service.api:app`)
- `service/engine.py`: Async eval orchestration (fake/replay/provider)
- `eval/`: benchmark assets, provider modules (`eval/llms`), legacy CLI/judging logic
- `runner.py`, `clients.py`, `tools.py`: reusable async runner/core abstractions
- `setup_ec2.sh`: EC2 bootstrap script (repo root)
- `tests/`: pytest suite

## Quick Start (Local)

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r eval/requirements.txt
uv run --python .venv/bin/python uvicorn service.api:app --host 0.0.0.0 --port 8000
```

Start a fake async run (no external model API calls):

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"engine":"async","client":"fake","eval_sem":64,"cpu_sem":8}'
```

## EC2 Setup

From repo root:

```bash
chmod +x setup_ec2.sh
./setup_ec2.sh
```

This installs dependencies, syncs dataset/reference files from Hugging Face, and sets up a `systemd` service for the API server.
