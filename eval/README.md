# KWBench Eval

LLM evaluation harness with crash-resume, S3 persistence, and W&B tracking.

## EC2 Setup (one-time)

```bash
git clone <repo-url> && cd rubrics
# Create .env with API keys (see Keys table below)
./setup_ec2.sh
```

The setup script installs deps, downloads the dataset from HuggingFace (`clio-ai/kwbench`), and registers a systemd service. The server survives SSH disconnects and auto-restarts on crash.

## Keys

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Judge model + Gemini subject LLM |
| `OPENAI_API_KEY` | If using OpenAI LLM | OpenAI subject LLM |
| `ANTHROPIC_API_KEY` | If using Claude LLM | Claude subject LLM |
| `MOONSHOT_API_KEY` | If using Kimi | Kimi subject LLM |
| `DASHSCOPE_API_KEY` | If using Qwen | Qwen subject LLM |
| `DEEPSEEK_API_KEY` | If using DeepSeek | DeepSeek subject LLM |
| `MISTRAL_API_KEY` | If using Mistral | Mistral subject LLM |
| `AWS_ACCESS_KEY_ID` | If using S3 | S3 result persistence |
| `AWS_SECRET_ACCESS_KEY` | If using S3 | S3 result persistence |
| `AWS_DEFAULT_REGION` | If using S3 | S3 region |
| `WANDB_API_KEY` | If using W&B | Experiment tracking |
| `S3_BUCKET` | Server only | S3 bucket for server mode |
| `S3_PREFIX` | Server only | S3 key prefix (default: `eval`) |
| `WANDB_PROJECT` | Server only | W&B project (default: `eval`) |

Put these in `.env` at the repo root or export them.

## Available Providers

Defined in `config.yaml`. Use the provider name as the `llm` value:

| Provider | Name | Notes |
|---|---|---|
| Gemini | `gemini` | Also used as judge model |
| OpenAI | `openai` | |
| Claude | `claude` | |
| Kimi | `kimi` | OpenAI-compatible (Moonshot) |
| Qwen | `qwen` | OpenAI-compatible (DashScope) |
| DeepSeek | `deepseek` | OpenAI-compatible |
| Mistral | `mistral` | OpenAI-compatible |

## Server Management

```bash
# Start / stop / restart the server
sudo systemctl start eval-server
sudo systemctl stop eval-server
sudo systemctl restart eval-server

# Tail logs (Ctrl+C to detach, server keeps running)
journalctl -u eval-server -f

# Check if running
sudo systemctl status eval-server
```

The server runs on port 8000 (override with `EVAL_PORT` env var). It persists through SSH disconnects and reboots.

## Running Evals

### Start async eval (default, no external model API calls)

This path uses the new asyncio/process-pool runner and defaults to `FakeLLMClient`.

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "async",
    "client": "fake",
    "eval_sem": 64,
    "cpu_sem": 8
  }'
```

Use replay fixtures instead of network calls:

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "async",
    "client": "replay",
    "replay_fixtures": "fixtures/replay.json",
    "eval_sem": 64,
    "cpu_sem": 8
  }'
```

Use provider modules from `eval/llms` (or provider names from `eval/config.yaml`) with REPL/tool support:

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "async",
    "client": "provider",
    "llm": "openai",
    "eval_sem": 64,
    "cpu_sem": 8
  }'
```

Force dataset refresh from Hugging Face:

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "async",
    "client": "fake",
    "hf_repo": "clio-ai/kwbench",
    "hf_force_refresh": true
  }'
```

Provider names from config also work (example: `kimi`, `qwen`, `deepseek`, `mistral`):

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "async",
    "client": "provider",
    "llm": "kimi",
    "eval_sem": 64,
    "cpu_sem": 8
  }'
```

### Start an eval

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"engine":"legacy","llm":"gemini"}'
```

Replace `"gemini"` with any provider name from the table above.
Legacy provider mode is explicit:

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"engine":"legacy","llm":"gemini"}'
```

### Start an eval on specific tasks

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"engine":"legacy","llm":"gemini","task_ids":["kw_001","kw_030"]}'
```

### Check progress

```bash
curl http://localhost:8000/eval/status
```

### List all completed runs

```bash
curl http://localhost:8000/eval/runs
```

### Get full results for a run

```bash
curl http://localhost:8000/eval/runs/<run-name>
```

### Resume a crashed run

```bash
curl -X POST http://localhost:8000/eval/resume \
  -H 'Content-Type: application/json' \
  -d '{"engine":"legacy","llm":"gemini","output":"eval/results/<filename>.jsonl"}'
```

Use the `output` path from the original `/eval/start` response.

### Health check

```bash
curl http://localhost:8000/health
```

## API Reference

| Endpoint | Method | Body | Purpose |
|---|---|---|---|
| `/eval/start` | POST | `{"engine":"async","client":"fake"}` | Start async fake eval (no external model API calls) |
| `/eval/start` | POST | `{"engine":"async","client":"replay","replay_fixtures":"fixtures/replay.json"}` | Start async replay eval |
| `/eval/start` | POST | `{"engine":"async","client":"provider","llm":"openai"}` | Start async provider eval using `eval/llms/openai.py` |
| `/eval/start` | POST | `{"engine":"async","client":"provider","llm":"kimi"}` | Start async provider eval using provider config mapping |
| `/eval/start` | POST | `{"engine":"async","hf_repo":"clio-ai/kwbench","hf_force_refresh":true}` | Pull dataset/reference files from HF before running |
| `/eval/start` | POST | `{"engine":"legacy","llm":"<provider>"}` | Start legacy provider eval |
| `/eval/start` | POST | `{"engine":"legacy","llm":"<provider>","task_ids":["kw_001"]}` | Start legacy eval on specific tasks |
| `/eval/resume` | POST | `{"engine":"async","output":"<path>"}` or `{"engine":"legacy","llm":"<provider>","output":"<path>"}` | Resume crashed run |
| `/eval/status` | GET | — | Live progress of active run |
| `/eval/runs` | GET | — | List all runs |
| `/eval/runs/{name}` | GET | — | Full results for a run |
| `/health` | GET | — | Health check |

Only one eval runs at a time. Starting a second returns `409`.

## Run (CLI)

```bash
source .venv/bin/activate

# Basic
python run.py --llm llms/gemini.py

# With S3 + W&B
python run.py --llm llms/claude.py --s3-bucket my-bucket --wandb-project my-eval

# Specific tasks
python run.py --llm llms/openai.py --ids kw_001,kw_030

# Resume interrupted run
python run.py --llm llms/gemini.py --output eval/results/<filename>.jsonl --s3-bucket my-bucket
```

## Run (GitHub Actions)

```bash
gh workflow run eval.yml -f llm=gemini -f s3_bucket=my-bucket -f wandb_project=my-eval
```

Add all keys as repo secrets.

## Crash Recovery

Results sync to S3 after every completed task. If the machine dies:
- Use `/eval/resume` with the same output path — picks up where it left off
- Or re-run CLI with the same `--output` path
- W&B dashboard shows run as "crashed" vs "finished"

## Dataset

Downloaded automatically by `setup_ec2.sh` from [clio-ai/kwbench](https://huggingface.co/datasets/clio-ai/kwbench). Places `dataset.jsonl` and `reference_files/` in the repo root.

Async mode can also auto-fetch from Hugging Face at runtime:
- If `dataset` is omitted and local `dataset.jsonl` is missing, it pulls `clio-ai/kwbench` by default.
- Override repo with `hf_repo`.
- Set `hf_force_refresh=true` to re-download and refresh local files.
- Disable runtime fetch with `hf_fetch_if_missing=false`.

To use a custom dataset, pass `--dataset path/to/file.jsonl` (CLI) or `{"dataset": "path"}` (API).
