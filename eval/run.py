#!/usr/bin/env python3
"""CLI: python eval/run.py [--provider kimi] [--model ...] [--ids kw_001] etc.

Loads defaults from eval/config.yaml. --provider selects a provider block.
CLI args (--base-url, --api-key, --model) override the selected provider.
"""

import argparse
import importlib.util
import os
import re
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.core import run_eval

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EVAL_DIR / "config.yaml"


def _expand_env(val):
    if isinstance(val, str):
        return re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), val)
    return val


def load_config(cli_args: argparse.Namespace) -> dict:
    """Load yaml, resolve provider, overlay CLI overrides."""
    config_path = Path(cli_args.config) if cli_args.config else DEFAULT_CONFIG
    raw = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}

    providers = raw.pop("providers", {})

    # Resolve provider: CLI --provider > yaml default
    provider_name = cli_args.provider or raw.get("provider")
    if provider_name and provider_name in providers:
        provider_cfg = providers[provider_name]
        raw.update(provider_cfg)
    elif provider_name and provider_name not in providers:
        print(f"Available providers: {', '.join(providers.keys())}")
        raise SystemExit(f"Unknown provider: {provider_name}")

    # CLI overrides (non-None only)
    for key, val in vars(cli_args).items():
        if key in ("config", "provider"):
            continue
        if val is not None:
            raw[key] = val

    # Expand env vars
    for key in raw:
        raw[key] = _expand_env(raw[key])

    return raw


def load_llm_module(path: str, cfg: dict):
    """Load LLM module, injecting oaichat config via env vars."""
    for key in ("base_url", "api_key", "model"):
        if cfg.get(key):
            os.environ[f"OAICHAT_{key.upper()}"] = str(cfg[key])
    spec = importlib.util.spec_from_file_location("llm_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "LLM_ID"), f"{path} must export LLM_ID"
    assert hasattr(mod, "generate"), f"{path} must export generate(task, references, config)"
    return mod


def main():
    parser = argparse.ArgumentParser(description="Run KWBench eval")
    parser.add_argument("--config", default=None, help="Path to config YAML (default: eval/config.yaml)")
    parser.add_argument("--provider", default=None, help="Provider name from config (e.g. kimi, qwen, deepseek)")
    parser.add_argument("--llm", default=None, help="Path to subject LLM module")
    parser.add_argument("--dataset", default=None, help="Path to dataset.jsonl")
    parser.add_argument("--output", default=None, help="Output JSONL path (reuse to resume)")
    parser.add_argument("--ids", default=None, help="Comma-separated task IDs (e.g. kw_001,kw_030)")
    parser.add_argument("--s3-bucket", default=None, dest="s3_bucket", help="S3 bucket for persistence")
    parser.add_argument("--s3-prefix", default=None, dest="s3_prefix", help="S3 key prefix")
    parser.add_argument("--wandb-project", default=None, dest="wandb_project", help="W&B project for tracking")
    parser.add_argument("--weave-project", default=None, dest="weave_project", help="Weave project for tracing (auto-traces OpenAI/Anthropic calls)")
    parser.add_argument("--base-url", default=None, dest="base_url", help="Override provider base URL")
    parser.add_argument("--api-key", default=None, dest="api_key", help="Override provider API key")
    parser.add_argument("--model", default=None, help="Override provider model name")
    parser.add_argument("--concurrency", type=int, default=None, help="Max parallel tasks (default: auto-detect from CPU count)")
    parser.add_argument("--log", default="terminal", help="Log destination: 'terminal' (default) or a file path")
    args = parser.parse_args()

    from eval.log import setup
    setup(args.log)

    cfg = load_config(args)

    llm_path = cfg.get("llm")
    assert llm_path, "Must specify --llm or set llm in config.yaml"
    llm = load_llm_module(llm_path, cfg)

    dataset = Path(cfg["dataset"]) if cfg.get("dataset") else None
    output = Path(cfg["output"]) if cfg.get("output") else None
    task_ids = set(cfg["ids"].split(",")) if cfg.get("ids") else None

    storage = None
    if cfg.get("s3_bucket"):
        from eval.storage import S3Storage
        storage = S3Storage(cfg["s3_bucket"], cfg.get("s3_prefix", "eval"))

    run_eval(llm, dataset, output, task_ids=task_ids, storage=storage,
             wandb_project=cfg.get("wandb_project"), weave_project=cfg.get("weave_project"),
             concurrency=cfg.get("concurrency"))


if __name__ == "__main__":
    main()
