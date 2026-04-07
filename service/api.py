"""Eval server. Run: uvicorn service.api:app --host 0.0.0.0 --port 8000"""

import importlib.util
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from service.engine import AsyncRunConfig, RESULTS_DIR, run_async_eval
from eval.storage import S3Storage
from eval.log import setup as setup_logging

# Server always logs to file
setup_logging(os.environ.get("EVAL_LOG", "eval/results/server.log"))

app = FastAPI(title="Eval Server")

LLMS_DIR = Path(__file__).resolve().parent / "llms"

_lock = threading.Lock()
_active: dict | None = None  # {thread, meta_path, output}
_last_meta_path: Path | None = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_storage() -> S3Storage | None:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return None
    return S3Storage(bucket, os.environ.get("S3_PREFIX", "eval"))


class EvalRequest(BaseModel):
    engine: Literal["async", "legacy"] = "async"
    llm: str | None = None
    dataset: str | None = None
    task_ids: list[str] | None = None
    output: str | None = None
    benchmark: str | None = None
    wandb_project: str | None = os.environ.get("WANDB_PROJECT")
    weave_project: str | None = os.environ.get("WEAVE_PROJECT")
    client: Literal["fake", "replay", "provider"] = "fake"
    replay_fixtures: str | None = None
    provider_config_path: str | None = None
    runtime_type: Literal["local", "daytona"] | None = None
    repl_mode: Literal["local", "daytona"] = "local"
    sandbox_template: str | None = None
    sandbox_concurrency: int | None = None
    eval_sem: int = 64
    cpu_sem: int | None = None
    max_retries: int = 2
    retry_base_s: float = 0.05
    fake_base_latency_s: float = 0.01
    fake_jitter_s: float = 0.04
    fake_tool_ratio: float = 0.5
    case_tool_mode: Literal["sleep", "cpu"] = "sleep"
    case_tool_payload: int | float | str = 0.02
    case_max_steps: int = 3
    judge_enabled: bool = False
    judge_provider: str | None = None
    judge_sem: int = 4
    judge_criterion_workers: int = 4
    hf_results_upload: bool = _env_bool("HF_RESULTS_UPLOAD", False)
    hf_results_repo: str = os.environ.get("HF_RESULTS_REPO", "clio-ai/kwbresults")
    hf_results_token: str | None = os.environ.get("HF_RESULTS_TOKEN")
    hf_repo: str = "clio-ai/kwbench"
    hf_fetch_if_missing: bool = True
    hf_force_refresh: bool = False


def _load_llm(name: str):
    path = LLMS_DIR / f"{name}.py"
    if not path.exists():
        raise HTTPException(404, f"LLM module not found: {name}")
    spec = importlib.util.spec_from_file_location(f"llm_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clear_active(run_token: str) -> None:
    global _active
    with _lock:
        if _active and _active.get("token") == run_token:
            _active = None


def _run_bg_legacy(llm_mod, dataset, output, task_ids, storage, wandb_project, weave_project, run_token: str):
    global _active
    try:
        from eval.core import run_eval
        run_eval(llm_mod, dataset, output, task_ids=task_ids, storage=storage, wandb_project=wandb_project, weave_project=weave_project)
    except Exception as exc:  # noqa: BLE001
        meta_path = output.with_suffix(".meta.json")
        payload: dict[str, object] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        if meta_path.exists():
            try:
                payload.update(json.loads(meta_path.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
        payload["status"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["output"] = str(output)
        meta_path.write_text(json.dumps(payload))
    finally:
        _clear_active(run_token)


def _run_bg_async(
    config: AsyncRunConfig,
    dataset: Path | None,
    output: Path,
    task_ids: set[str] | None,
    storage: S3Storage | None,
    run_token: str,
) -> None:
    global _active
    try:
        run_async_eval(config, dataset_path=dataset, output_path=output, task_ids=task_ids, storage=storage)
    except Exception as exc:  # noqa: BLE001
        meta_path = output.with_suffix(".meta.json")
        payload: dict[str, object] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        if meta_path.exists():
            try:
                payload.update(json.loads(meta_path.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
        payload["status"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["output"] = str(output)
        meta_path.write_text(json.dumps(payload))
    finally:
        _clear_active(run_token)


def _validate_request(req: EvalRequest) -> None:
    if req.eval_sem < 1:
        raise HTTPException(400, "eval_sem must be >= 1")
    if req.cpu_sem is not None and req.cpu_sem < 1:
        raise HTTPException(400, "cpu_sem must be >= 1")
    if req.sandbox_concurrency is not None and req.sandbox_concurrency < 1:
        raise HTTPException(400, "sandbox_concurrency must be >= 1")
    if req.max_retries < 0:
        raise HTTPException(400, "max_retries must be >= 0")
    if req.case_max_steps < 1:
        raise HTTPException(400, "case_max_steps must be >= 1")
    if req.judge_sem < 1:
        raise HTTPException(400, "judge_sem must be >= 1")
    if req.judge_criterion_workers < 1:
        raise HTTPException(400, "judge_criterion_workers must be >= 1")
    if req.hf_results_upload and not req.hf_results_repo.strip():
        raise HTTPException(400, "hf_results_repo must be non-empty when hf_results_upload=true")
    if req.engine == "legacy" and not req.llm:
        raise HTTPException(400, "llm is required when engine='legacy'")
    if req.engine == "async" and req.client == "replay" and not req.replay_fixtures:
        raise HTTPException(400, "replay_fixtures is required when client='replay'")
    if req.engine == "async" and req.client == "provider" and not req.llm:
        raise HTTPException(400, "llm is required when client='provider'")


def _resolve_output(req: EvalRequest, ts: str) -> Path:
    if req.output:
        return Path(req.output)
    if req.engine == "legacy":
        llm_mod = _load_llm(req.llm or "")
        return RESULTS_DIR / f"{llm_mod.LLM_ID}_{ts}.jsonl"
    suffix = req.client if req.client != "provider" else (req.llm or "provider")
    safe_suffix = suffix.replace("/", "_")
    return RESULTS_DIR / f"async-{safe_suffix}_{ts}.jsonl"


def _start(req: EvalRequest, out: Path) -> dict:
    global _active
    global _last_meta_path
    _validate_request(req)

    dataset = Path(req.dataset) if req.dataset else None
    task_ids = set(req.task_ids) if req.task_ids else None
    storage = _get_storage()
    meta_path = out.with_suffix(".meta.json")
    run_token = uuid.uuid4().hex

    if req.engine == "legacy":
        llm_mod = _load_llm(req.llm or "")
        args = (
            llm_mod,
            dataset,
            out,
            task_ids,
            storage,
            req.wandb_project,
            req.weave_project,
            run_token,
        )
        t = threading.Thread(target=_run_bg_legacy, args=args, daemon=True)
    else:
        config = AsyncRunConfig(
            client=req.client,
            replay_fixtures=req.replay_fixtures,
            provider=req.llm,
            provider_config_path=req.provider_config_path,
            benchmark=req.benchmark,
            runtime_type=req.runtime_type,
            repl_mode=req.repl_mode,
            sandbox_template=req.sandbox_template,
            sandbox_concurrency=req.sandbox_concurrency,
            eval_sem=req.eval_sem,
            cpu_sem=req.cpu_sem,
            max_retries=req.max_retries,
            retry_base_s=req.retry_base_s,
            fake_base_latency_s=req.fake_base_latency_s,
            fake_jitter_s=req.fake_jitter_s,
            fake_tool_ratio=req.fake_tool_ratio,
            case_tool_mode=req.case_tool_mode,
            case_tool_payload=req.case_tool_payload,
            case_max_steps=req.case_max_steps,
            wandb_project=req.wandb_project,
            weave_project=req.weave_project,
            judge_enabled=req.judge_enabled,
            judge_provider=req.judge_provider,
            judge_sem=req.judge_sem,
            judge_criterion_workers=req.judge_criterion_workers,
            hf_results_upload=req.hf_results_upload,
            hf_results_repo=req.hf_results_repo,
            hf_results_token=req.hf_results_token,
            hf_repo=req.hf_repo,
            hf_fetch_if_missing=req.hf_fetch_if_missing,
            hf_force_refresh=req.hf_force_refresh,
        )
        args = (config, dataset, out, task_ids, storage, run_token)
        t = threading.Thread(target=_run_bg_async, args=args, daemon=True)

    with _lock:
        if _active and _active["thread"].is_alive():
            raise HTTPException(409, "Eval already running")
        _active = {"thread": t, "meta_path": meta_path, "output": out, "token": run_token}
        _last_meta_path = meta_path
        try:
            t.start()
        except Exception:
            if _active and _active.get("token") == run_token:
                _active = None
            raise
    return {
        "status": "started",
        "engine": req.engine,
        "client": req.client if req.engine == "async" else req.llm,
        "llm": req.llm,
        "output": str(out),
        "meta": str(meta_path),
    }


@app.post("/eval/start")
def start_eval(req: EvalRequest):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = _resolve_output(req, ts)
    return _start(req, out)


@app.post("/eval/resume")
def resume_eval(req: EvalRequest):
    if not req.output:
        raise HTTPException(400, "output path required to resume")
    out = Path(req.output)
    # If no local file, try S3
    storage = _get_storage()
    if not out.exists() and storage:
        if not storage.get(out.name, out):
            raise HTTPException(404, f"No results found for: {out.name}")
    return _start(req, out)


@app.get("/eval/status")
def get_status():
    global _last_meta_path
    with _lock:
        if _active and _active["thread"].is_alive() and _active["meta_path"].exists():
            return json.loads(_active["meta_path"].read_text())
        if _active and _active["meta_path"].exists():
            return json.loads(_active["meta_path"].read_text())
        if _last_meta_path and _last_meta_path.exists():
            return json.loads(_last_meta_path.read_text())
    return {"status": "idle"}


@app.get("/eval/runs")
def list_runs():
    """List all runs. Checks S3 if configured, falls back to local."""
    storage = _get_storage()
    if storage:
        keys = storage.list()
        metas = [k for k in keys if k.endswith(".meta.json")]
        runs = []
        for k in sorted(metas, reverse=True):
            local = RESULTS_DIR / k
            if not local.exists():
                storage.get(k, local)
            try:
                runs.append(json.loads(local.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return runs

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        json.loads(m.read_text())
        for m in sorted(RESULTS_DIR.glob("*.meta.json"), reverse=True)
        if m.stat().st_size > 0
    ]


@app.get("/eval/runs/{name}")
def get_run(name: str):
    stem = name.removesuffix(".jsonl").removesuffix(".meta.json")
    jsonl = RESULTS_DIR / f"{stem}.jsonl"
    meta_path = RESULTS_DIR / f"{stem}.meta.json"

    # Try S3 if not local
    storage = _get_storage()
    if not jsonl.exists() and storage:
        storage.get(f"{stem}.jsonl", jsonl)
        storage.get(f"{stem}.meta.json", meta_path)

    if not jsonl.exists():
        raise HTTPException(404, f"Run not found: {stem}")

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    results = []
    for line in open(jsonl):
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    scores = []
    for row in results:
        scoring = row.get("scoring")
        if isinstance(scoring, dict) and isinstance(scoring.get("score"), (int, float)):
            scores.append(float(scoring["score"]))
            continue
        eval_payload = row.get("eval")
        if isinstance(eval_payload, dict) and isinstance(eval_payload.get("score"), (int, float)):
            scores.append(float(eval_payload["score"]))
    ok_count = sum(1 for r in results if r.get("status") == "ok" or "scoring" in r or "eval" in r)
    failed_count = sum(1 for r in results if r.get("status") == "error")
    async_durations = [
        r.get("metrics", {}).get("total_s")
        for r in results
        if isinstance(r.get("metrics"), dict)
    ]
    async_durations = [d for d in async_durations if isinstance(d, (int, float))]
    return {
        **meta,
        "count": len(results),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "ok_count": ok_count,
        "failed_count": failed_count,
        "avg_total_s": round(sum(async_durations) / len(async_durations), 4) if async_durations else 0,
        "results": results,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
