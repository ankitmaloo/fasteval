# FastEval Implementation Guide

A complete step-by-step guide for running benchmarks with FastEval. This guide collates all the necessary information from the repository to help you get started quickly.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation & Setup](#installation--setup)
3. [Environment Configuration](#environment-configuration)
4. [Running Your First Benchmark](#running-your-first-benchmark)
5. [Creating a Custom Benchmark](#creating-a-custom-benchmark)
6. [Advanced Configuration](#advanced-configuration)
7. [Monitoring & Results](#monitoring--results)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.9+
- `uv` package manager (automatically installed by setup script)
- API keys for your chosen LLM providers
- (Optional) AWS credentials for S3 storage
- (Optional) Weights & Biases account for experiment tracking

---

## Installation & Setup

### Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd fasteval

# Create virtual environment
uv venv .venv

# Install dependencies
uv pip install --python .venv/bin/python -r eval/requirements.txt

# Activate virtual environment
source .venv/bin/activate
```

### EC2 Setup (Production)

For production deployments on EC2:

```bash
# Make setup script executable
chmod +x setup_ec2.sh

# Run the setup script
./setup_ec2.sh
```

This script will:
- Install system dependencies (`curl`, `git`)
- Install `uv` package manager
- Create a Python virtual environment
- Install all required packages
- Download the default dataset from HuggingFace (`clio-ai/kwbench`)
- Set up a systemd service for auto-restart and persistence

---

## Environment Configuration

### Step 1: Create your `.env` file

Copy the example environment file:

```bash
cp .env.example .env
```

### Step 2: Add your API keys

Edit `.env` and add keys for the providers you'll use:

```bash
# Core providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Optional providers
MOONSHOT_API_KEY=...        # For Kimi
DASHSCOPE_API_KEY=...       # For Qwen
DEEPSEEK_API_KEY=...        # For DeepSeek
MISTRAL_API_KEY=...         # For Mistral
NEBIUS_API_KEY=...          # For Nebius-hosted models
NVIDIA_API_KEY=...          # For NVIDIA-hosted models

# Storage & tracking (optional)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=my-eval-results

WANDB_API_KEY=...
WANDB_PROJECT=my-evals

# Server configuration
EVAL_PORT=8000
```

### Step 3: Start the server

**For local development:**

```bash
# Load environment variables and start server
set -a && source .env && set +a && source .venv/bin/activate
uvicorn service.api:app --host 0.0.0.0 --port 8000
```

**For EC2/production (using systemd):**

```bash
# Start the service
sudo systemctl start eval-server

# Check status
sudo systemctl status eval-server

# View logs
journalctl -u eval-server -f

# Enable auto-start on boot
sudo systemctl enable eval-server
```

---

## Running Your First Benchmark

### Option 1: Quick Test (No External API Calls)

Perfect for testing your setup:

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

This runs a simulated benchmark with no actual LLM API calls.

### Option 2: Run GSM8K Math Benchmark

Simple Q&A benchmark with deterministic scoring:

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "client": "provider",
    "llm": "openai",
    "provider_config_path": "eval/config.yaml",
    "benchmark": "gsm8k",
    "judge_enabled": false,
    "case_max_steps": 1,
    "eval_sem": 64
  }'
```

### Option 3: Run KWBench (with Judge Scoring)

Complex benchmark with tool use and rubric-based judging:

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "client": "provider",
    "llm": "openai",
    "provider_config_path": "eval/config.yaml",
    "benchmark": "kwbench",
    "judge_enabled": true,
    "judge_provider": "gemini",
    "judge_sem": 4,
    "judge_criterion_workers": 15,
    "case_max_steps": 3,
    "eval_sem": 32,
    "cpu_sem": 4
  }'
```

### Checking Progress

```bash
# Get current status
curl http://localhost:8000/eval/status | python3 -m json.tool

# List all completed runs
curl http://localhost:8000/eval/runs | python3 -m json.tool

# Get detailed results for a specific run
curl http://localhost:8000/eval/runs/<run-name> | python3 -m json.tool
```

---

## Creating a Custom Benchmark

### Step 1: Choose Your Benchmark Pattern

FastEval supports three common patterns:

**Pattern A: Q&A with Known Answers (like GSM8K)**
- Simple question-answer pairs
- No tools needed
- Deterministic scoring (exact match, regex, etc.)

**Pattern B: Rubric-Based Judging (like KWBench)**
- Open-ended tasks
- Tool use supported (bash, python, search)
- LLM judge evaluates based on rubric criteria

**Pattern C: Custom Scoring**
- Your own scoring logic
- Test suites, file verification, API calls, etc.

### Step 2: Create the Plugin File

Create `eval/benchmarks/my_benchmark.py`:

**Example: Simple Q&A Benchmark**

```python
from __future__ import annotations
from pathlib import Path
from typing import Any
from eval.benchmarks.base import BaseBenchmarkPlugin
from eval.scorers import ScorerResult


class MyBenchmarkPlugin(BaseBenchmarkPlugin):
    name = "my_benchmark"

    def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
        """Load your dataset. Return list of task dicts."""
        # Option 1: Load from local JSONL file
        if dataset_path and dataset_path.exists():
            import json
            return [json.loads(line) for line in dataset_path.open() if line.strip()]

        # Option 2: Load from HuggingFace
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
        """Return [] to disable tools, None to use config default."""
        return []  # No tools for Q&A

    def build_prompt(self, case: dict[str, Any], context: dict[str, Any] | None) -> str | None:
        """Customize the prompt sent to the model."""
        return f"{case['task']}\n\nThink step by step and provide your answer."

    def summarize_run(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Compute summary statistics."""
        scored = [r for r in results if isinstance(r.get("scoring"), dict)]
        correct = sum(1 for r in scored if r["scoring"].get("score", 0) == 1.0)
        return {
            "total": len(results),
            "correct": correct,
            "accuracy": round(correct / max(len(scored), 1), 4),
        }
```

**Example: Rubric-Based Benchmark**

```python
from __future__ import annotations
from pathlib import Path
from typing import Any
from eval.benchmarks.base import BaseBenchmarkPlugin


class MyJudgedBenchPlugin(BaseBenchmarkPlugin):
    name = "my_judged_bench"

    def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
        """Each task has rubric with mandatory, good_to_have, and ideal criteria."""
        return [
            {
                "id": "case_001",
                "task": "Write a comprehensive marketing strategy for a new SaaS product.",
                "rubric": {
                    "mandatory": [
                        "Identifies target customer segments clearly",
                        "Proposes at least 3 distinct marketing channels",
                        "Includes budget considerations",
                    ],
                    "good_to_have": [
                        "Provides timeline with milestones",
                        "Addresses competitive landscape",
                    ],
                    "ideal": [
                        "Includes measurable KPIs for each channel",
                        "Proposes A/B testing strategy",
                    ],
                },
                "source": "my_dataset",
                "category": "business",
            },
        ]

    def summarize_run(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        scores = [
            r["eval"]["score"]
            for r in results
            if isinstance(r.get("eval"), dict)
        ]
        return {
            "avg_score": round(sum(scores) / max(len(scores), 1), 4),
            "scored_count": len(scores),
        }
```

### Step 3: Register in Config

Edit `eval/config.yaml` and add your benchmark:

```yaml
benchmarks:
  my_benchmark:
    module: eval/benchmarks/my_benchmark.py
    class: MyBenchmarkPlugin
```

### Step 4: Run Your Benchmark

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "client": "provider",
    "llm": "openai",
    "benchmark": "my_benchmark",
    "judge_enabled": false,
    "case_max_steps": 1,
    "eval_sem": 64
  }'
```

---

## Advanced Configuration

### Concurrency Control

Four semaphores control parallelism:

```json
{
  "eval_sem": 32,              // Max tasks running simultaneously
  "cpu_sem": 4,                // Max concurrent tool executions
  "judge_sem": 4,              // Max cases being judged at once
  "judge_criterion_workers": 15 // Criteria evaluated in parallel per case
}
```

**Recommended settings by benchmark type:**

| Benchmark Type | eval_sem | cpu_sem | judge_sem | criterion_workers | case_max_steps |
|---|---|---|---|---|---|
| Q&A, no tools | 64 | 1 | — | — | 1 |
| Q&A with judge | 32 | 1 | 4 | 15 | 1 |
| Agentic with tools | 32 | 4 | — | — | 3 |
| Agentic + judge (kwbench) | 32 | 4 | 4 | 15 | 3 |

### Running Specific Tasks

Filter to specific task IDs:

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "client": "provider",
    "llm": "openai",
    "benchmark": "gsm8k",
    "task_ids": ["gsm8k_0001", "gsm8k_0002", "gsm8k_0003"]
  }'
```

### Custom Output Path

Specify where results are saved:

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "client": "provider",
    "llm": "openai",
    "benchmark": "gsm8k",
    "output": "/home/runner/work/fasteval/fasteval/service/results/my_run.jsonl"
  }'
```

**IMPORTANT:** Always use absolute paths for output directories, never relative paths.

### Resuming Failed Runs

If a run crashes or is interrupted:

```bash
curl -X POST http://localhost:8000/eval/resume \
  -H 'Content-Type: application/json' \
  -d '{
    "engine": "async",
    "output": "/home/runner/work/fasteval/fasteval/service/results/my_run.jsonl"
  }'
```

The system will skip already-completed tasks and continue from where it left off.

### Adding a New Provider

Edit `eval/config.yaml`:

**For OpenAI-compatible APIs:**

```yaml
providers:
  my-custom-model:
    llm: eval/llms/oaichat.py
    base_url: https://api.example.com/v1/
    api_key: ${MY_API_KEY}
    model: org/model-name
```

**For native providers:**

```yaml
providers:
  my-provider:
    llm: eval/llms/my_provider.py
```

Then add `MY_API_KEY` to your `.env` file.

### Full Configuration Example

```bash
cat > /tmp/eval_config.json <<'JSON'
{
  "engine": "async",
  "client": "provider",
  "llm": "openai",
  "provider_config_path": "eval/config.yaml",
  "benchmark": "kwbench",

  "dataset": "dataset.jsonl",
  "task_ids": null,
  "output": "/home/runner/work/fasteval/fasteval/service/results/my-eval.jsonl",

  "eval_sem": 32,
  "cpu_sem": 4,
  "case_max_steps": 3,
  "max_retries": 3,

  "judge_enabled": true,
  "judge_provider": "gemini",
  "judge_sem": 4,
  "judge_criterion_workers": 15,

  "wandb_project": "my-evals",
  "hf_fetch_if_missing": false
}
JSON

curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  --data @/tmp/eval_config.json
```

---

## Monitoring & Results

### Live Progress Monitoring

```bash
# Check status every 5 seconds
watch -n 5 'curl -s http://localhost:8000/eval/status | python3 -m json.tool'
```

Key fields to monitor:
- `status`: `running`, `completed`, or `error`
- `completed` / `total`: progress counter
- `ok` / `failed`: success vs error count
- `current_in_flight_evals`: tasks running right now
- `current_in_flight_judges`: cases being judged right now
- `heartbeat_at`: last update timestamp (should advance every 5s)

### Results Files

Results are saved in two files:

**1. JSONL file (detailed results):** `service/results/<name>.jsonl`

Each line is one completed task:

```bash
# View first 5 results
head -5 service/results/my-eval.jsonl | python3 -m json.tool

# Quick summary
python3 -c "
import json
rows = [json.loads(l) for l in open('service/results/my-eval.jsonl')]
for r in rows[:5]:
    s = r.get('scoring', {})
    print(f\"{r['id']}: score={s.get('score', '?')} method={s.get('method', '?')}\")
"
```

**2. Meta file (run metadata):** `service/results/<name>.meta.json`

```bash
cat service/results/my-eval.meta.json | python3 -m json.tool
```

Contains:
- Run configuration
- Progress statistics
- Benchmark summary (accuracy, avg_score, etc.)
- Timing metrics

### Understanding Results

**Result row structure:**

```json
{
  "id": "task_001",
  "task": "The question or prompt",
  "llm_answer": "The model's response",
  "status": "ok",
  "scoring": {
    "method": "exact_match",
    "score": 1.0,
    "detail": {...}
  },
  "metrics": {
    "total_s": 2.5,
    "model_wait_s": 2.3,
    "tool_cpu_s": 0.2
  },
  "llm_metadata": {
    "input_tokens": 150,
    "output_tokens": 200,
    "total_cost": 0.0012
  }
}
```

**Scoring methods:**

- `exact_match`: String comparison
- `gsm8k_numeric`: Numeric extraction and comparison
- `rubric_judge`: LLM-based rubric evaluation
- `custom`: Plugin-defined scoring

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `0/0 tasks` | Plugin returned empty list or task_ids filter matched nothing | Check plugin's `load_cases()`. Verify task IDs. |
| All tasks `failed` with 401 | API key not in environment | Add key to `.env` and restart server |
| All tasks `failed` with 429 | Rate limit exceeded | Lower `eval_sem`. For judging, lower `judge_sem × judge_criterion_workers`. |
| Judging extremely slow | `judge_criterion_workers` too low | Set to 15 (matches max criteria per task) |
| `status: error` with S3 message | S3 bucket not accessible | Unset `S3_BUCKET` if not needed |
| `FileNotFoundError` on output | Run had 0 tasks | Check task loading |
| Server won't start | Port already in use | Change `EVAL_PORT` in `.env` or kill existing process |

### Debugging Tips

**Check server logs:**

```bash
# For systemd service
journalctl -u eval-server -f -n 100

# For manual server
# Logs are printed to stdout
```

**Verify environment:**

```bash
# Check if API key is loaded
source .env
echo $OPENAI_API_KEY
```

**Test provider connectivity:**

```bash
# Run a minimal fake benchmark
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"engine":"async","client":"fake","eval_sem":2}'
```

**Check dataset loading:**

```bash
# Verify dataset file exists
ls -lh dataset.jsonl

# Check first few lines
head -3 dataset.jsonl | python3 -m json.tool
```

### Getting Help

- Check existing documentation: `eval/README.md`, `AGENTS.md`, `eval/how-eval-works.md`
- Review example benchmarks: `eval/benchmarks/gsm8k.py`, `eval/benchmarks/kwbench.py`
- Check API endpoint reference in `eval/README.md`

---

## Quick Command Reference

### Server Management

```bash
# Start server locally
set -a && source .env && set +a && uvicorn service.api:app --host 0.0.0.0 --port 8000

# Start systemd service
sudo systemctl start eval-server

# Stop systemd service
sudo systemctl stop eval-server

# View logs
journalctl -u eval-server -f
```

### API Calls

```bash
# Health check
curl http://localhost:8000/health

# Start eval
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"client":"provider","llm":"openai","benchmark":"gsm8k"}'

# Check status
curl http://localhost:8000/eval/status

# List runs
curl http://localhost:8000/eval/runs

# Resume crashed run
curl -X POST http://localhost:8000/eval/resume \
  -H 'Content-Type: application/json' \
  -d '{"engine":"async","output":"service/results/my-run.jsonl"}'
```

### Dataset Management

```bash
# Download dataset from HuggingFace
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"engine":"async","client":"fake","hf_repo":"clio-ai/kwbench","hf_force_refresh":true}'
```

---

## Next Steps

1. **Run the built-in benchmarks** to familiarize yourself with the system
2. **Create a simple custom benchmark** following Pattern A (Q&A)
3. **Experiment with different providers** to compare performance
4. **Set up monitoring** with Weights & Biases or S3 persistence
5. **Build production benchmarks** with rubric scoring and tool use

For more detailed information, see:
- `eval/how-eval-works.md` - Deep dive into the evaluation pipeline
- `AGENTS.md` - Architecture and development guidelines
- `eval/benchmarks/README.md` - Benchmark plugin interface reference
