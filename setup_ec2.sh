#!/usr/bin/env bash
set -euo pipefail

# --- EC2 setup for eval server ---
# Usage: ssh into a fresh Ubuntu 22.04+ instance, clone the repo, then:
#   cd rubrics && chmod +x setup_ec2.sh && ./setup_ec2.sh

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT="${EVAL_PORT:-8000}"
HF_REPO="${HF_DATASET_REPO:-clio-ai/kwbench}"
HF_FORCE_REFRESH="${HF_FORCE_REFRESH:-0}"

echo "==> Installing system deps"
sudo apt-get update -qq
sudo apt-get install -y -qq curl git

echo "==> Installing uv"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "==> Creating venv"
cd "$REPO_ROOT"
if [ ! -d "$REPO_ROOT/.venv" ]; then
    uv venv
fi
source .venv/bin/activate

echo "==> Installing Python packages"
uv pip install -r eval/requirements.txt

echo "==> Checking .env"
if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "WARNING: No .env found at $REPO_ROOT/.env"
    echo "Create one with your API keys (see eval/README.md)"
fi

echo "==> Syncing dataset from Hugging Face: $HF_REPO"
if [ "$HF_FORCE_REFRESH" = "1" ] || [ ! -f "$REPO_ROOT/dataset.jsonl" ]; then
    "$REPO_ROOT/.venv/bin/python" - <<PY
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

repo_root = Path("$REPO_ROOT")
repo_id = "$HF_REPO"
cache_dir = repo_root / ".cache" / "hf_datasets" / repo_id.replace("/", "__")
snapshot_path = Path(
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
)

src_jsonl = snapshot_path / "dataset.jsonl"
src_json = snapshot_path / "dataset.json"
dst_jsonl = repo_root / "dataset.jsonl"
if src_jsonl.exists():
    shutil.copy2(src_jsonl, dst_jsonl)
elif src_json.exists():
    payload = json.loads(src_json.read_text(encoding="utf-8"))
    with dst_jsonl.open("w", encoding="utf-8") as handle:
        if isinstance(payload, list):
            for item in payload:
                handle.write(json.dumps(item) + "\\n")
        else:
            handle.write(json.dumps(payload) + "\\n")
else:
    raise SystemExit(f"ERROR: {repo_id} snapshot missing dataset.jsonl/dataset.json")

src_refs = snapshot_path / "reference_files"
dst_refs = repo_root / "reference_files"
if src_refs.exists():
    if dst_refs.exists():
        shutil.rmtree(dst_refs)
    shutil.copytree(src_refs, dst_refs)
PY
    echo "Dataset sync complete."
else
    echo "dataset.jsonl already exists; skipping HF sync. Set HF_FORCE_REFRESH=1 to refresh."
fi

if [ ! -f "$REPO_ROOT/dataset.jsonl" ]; then
    echo "ERROR: Failed to materialize dataset.jsonl from HF repo: $HF_REPO"
    exit 1
fi

echo "==> Setting up systemd service"
sudo tee /etc/systemd/system/eval-server.service > /dev/null <<EOF
[Unit]
Description=KWBench Eval Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$REPO_ROOT
EnvironmentFile=-$REPO_ROOT/.env
ExecStart=$REPO_ROOT/.venv/bin/uvicorn service.api:app --host 0.0.0.0 --port $PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable eval-server

echo ""
echo "==> Done! Next steps:"
echo "  1. Create $REPO_ROOT/.env with your API keys"
echo "  2. Start the server:"
echo "       sudo systemctl start eval-server"
echo "       journalctl -u eval-server -f   # tail logs"
echo ""
echo "  Or run manually:"
echo "       cd $REPO_ROOT && source .venv/bin/activate"
echo "       uvicorn service.api:app --host 0.0.0.0 --port $PORT"
echo ""
echo "  CLI mode (no server):"
echo "       python eval/run.py --llm eval/llms/gemini.py --s3-bucket YOUR_BUCKET --wandb-project YOUR_PROJECT"
echo ""
echo "  API:"
echo "       # Async fake (no external model API calls)"
echo "       curl -X POST http://localhost:$PORT/eval/start -H 'Content-Type: application/json' -d '{\"engine\":\"async\",\"client\":\"fake\",\"eval_sem\":64,\"cpu_sem\":8}'"
echo "       # Async provider (existing eval/llms modules)"
echo "       curl -X POST http://localhost:$PORT/eval/start -H 'Content-Type: application/json' -d '{\"engine\":\"async\",\"client\":\"provider\",\"llm\":\"openai\",\"eval_sem\":64,\"cpu_sem\":8}'"
echo "       # Legacy mode"
echo "       curl -X POST http://localhost:$PORT/eval/start -H 'Content-Type: application/json' -d '{\"engine\":\"legacy\",\"llm\":\"gemini\"}'"
echo "       curl http://localhost:$PORT/eval/status"
echo "       curl http://localhost:$PORT/eval/runs"
