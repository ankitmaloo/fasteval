# FastEval

A production-ready evaluation harness for benchmarking LLMs with:

- **Async execution engine** - `asyncio` + process pool for high throughput
- **FastAPI service** - RESTful endpoints for starting, monitoring, and resuming evaluations
- **Multiple execution modes** - Fake (testing), replay (fixtures), and provider (real LLM calls)
- **Flexible benchmarking** - Built-in benchmarks (GSM8K, KWBench, Terminal) with easy plugin system
- **Crash recovery** - S3 persistence and automatic resume support
- **Comprehensive scoring** - Deterministic matching, LLM-based judging, and custom scorers

## 🚀 Quick Links

- **[Implementation Guide](IMPLEMENTATION_GUIDE.md)** - Complete step-by-step guide for running benchmarks
- **[Benchmark Details](eval/README.md)** - API reference and detailed configuration
- **[How It Works](eval/how-eval-works.md)** - Deep dive into the evaluation pipeline
- **[Development Guide](AGENTS.md)** - Architecture and contribution guidelines

## Repository Layout

- `service/api.py`: FastAPI app (`uvicorn service.api:app`)
- `service/engine.py`: Async eval orchestration (fake/replay/provider)
- `eval/`: benchmark assets, provider modules (`eval/llms`), legacy CLI/judging logic
- `runner.py`, `clients.py`, `tools.py`: reusable async runner/core abstractions
- `setup_ec2.sh`: EC2 bootstrap script (repo root)
- `tests/`: pytest suite

## 🏁 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
uv venv .venv

# Install packages
uv pip install --python .venv/bin/python -r eval/requirements.txt

# Activate environment
source .venv/bin/activate
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# At minimum, add one provider key (e.g., OPENAI_API_KEY, GEMINI_API_KEY)
```

### 3. Start the Server

```bash
# Load environment and start server
set -a && source .env && set +a
uvicorn service.api:app --host 0.0.0.0 --port 8000
```

### 4. Run Your First Benchmark

**Test with fake client (no API calls):**

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{"engine":"async","client":"fake","eval_sem":64,"cpu_sem":8}'
```

**Run GSM8K math benchmark:**

```bash
curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  -d '{
    "client": "provider",
    "llm": "openai",
    "benchmark": "gsm8k",
    "judge_enabled": false,
    "case_max_steps": 1,
    "eval_sem": 64
  }'
```

**Check progress:**

```bash
curl http://localhost:8000/eval/status | python3 -m json.tool
```

📖 **For detailed instructions, see the [Implementation Guide](IMPLEMENTATION_GUIDE.md)**

---

## 🖥️ Production Deployment (EC2)

For production use with auto-restart and systemd:

```bash
chmod +x setup_ec2.sh
./setup_ec2.sh
```

This script:
- Installs all dependencies
- Downloads the default dataset from HuggingFace
- Sets up a systemd service for persistence
- Enables auto-restart on failure

**Manage the service:**

```bash
# Start server
sudo systemctl start eval-server

# View logs
journalctl -u eval-server -f

# Check status
sudo systemctl status eval-server
```
