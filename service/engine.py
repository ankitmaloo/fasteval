from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from clients import (
    FakeLLMClient,
    LLMClient,
    ProviderCaseContext,
    ProviderLLMClient,
    ReplayLLMClient,
)
from runner import CaseResult, EvalCase, Runner

from eval.benchmarks.base import ExecutionProfile
from eval.log import log
from eval.scorers import RubricJudgeScorer, ScorerResult, resolve_scorer
from eval.storage import Storage, fire_and_forget

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
HEARTBEAT_INTERVAL_S = 5.0


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AsyncRunConfig:
    client: Literal["fake", "replay", "provider"] = "fake"
    replay_fixtures: str | None = None
    provider: str | None = None
    provider_config_path: str | None = None
    benchmark: str | None = None
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
    wandb_project: str | None = None
    weave_project: str | None = None
    judge_enabled: bool = False
    judge_provider: str | None = None
    judge_sem: int = 4
    judge_criterion_workers: int = 4
    hf_results_upload: bool = field(default_factory=lambda: _env_bool("HF_RESULTS_UPLOAD", False))
    hf_results_repo: str = field(
        default_factory=lambda: os.environ.get("HF_RESULTS_REPO", "clio-ai/kwbresults")
    )
    hf_results_token: str | None = field(default_factory=lambda: os.environ.get("HF_RESULTS_TOKEN"))
    hf_repo: str = "clio-ai/kwbench"
    hf_fetch_if_missing: bool = True
    hf_force_refresh: bool = False


def load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    dataset_path = path or DATASET_PATH
    if dataset_path.suffix.lower() == ".json":
        with dataset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must be a list of objects.")
        return [dict(item) for item in payload]

    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} in {dataset_path}") from exc
    return rows


def _download_hf_snapshot(repo_id: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic dataset download. "
            "Install it or provide --dataset explicitly."
        ) from exc

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return Path(snapshot_path)


def _to_jsonl_file(src_json: Path, dst_jsonl: Path) -> None:
    payload = json.loads(src_json.read_text(encoding="utf-8"))
    with dst_jsonl.open("w", encoding="utf-8") as handle:
        if isinstance(payload, list):
            for item in payload:
                handle.write(json.dumps(item) + "\n")
        else:
            handle.write(json.dumps(payload) + "\n")


def _materialize_hf_dataset(
    *,
    snapshot_dir: Path,
    target_dataset: Path,
    target_refs: Path,
    force_refresh: bool,
) -> None:
    src_jsonl = snapshot_dir / "dataset.jsonl"
    src_json = snapshot_dir / "dataset.json"
    if not src_jsonl.exists() and not src_json.exists():
        raise ValueError(
            f"Dataset snapshot at {snapshot_dir} must include dataset.jsonl or dataset.json"
        )

    target_dataset.parent.mkdir(parents=True, exist_ok=True)
    if force_refresh or not target_dataset.exists():
        if src_jsonl.exists():
            shutil.copy2(src_jsonl, target_dataset)
        else:
            _to_jsonl_file(src_json, target_dataset)

    src_refs = snapshot_dir / "reference_files"
    if src_refs.exists():
        if force_refresh and target_refs.exists():
            shutil.rmtree(target_refs)
        if not target_refs.exists():
            shutil.copytree(src_refs, target_refs)


def _ensure_dataset_materialized(
    *,
    dataset_path: Path | None,
    config: AsyncRunConfig,
) -> Path:
    if dataset_path is not None:
        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_path}")
        return dataset_path

    if DATASET_PATH.exists() and not config.hf_force_refresh:
        return DATASET_PATH
    if not config.hf_fetch_if_missing and not DATASET_PATH.exists():
        raise ValueError(
            f"Dataset not found at {DATASET_PATH}. "
            "Set hf_fetch_if_missing=true or pass dataset explicitly."
        )

    cache_root = BASE_DIR / ".cache" / "hf_datasets" / config.hf_repo.replace("/", "__")
    snapshot_dir = _download_hf_snapshot(config.hf_repo, cache_root)
    _materialize_hf_dataset(
        snapshot_dir=snapshot_dir,
        target_dataset=DATASET_PATH,
        target_refs=BASE_DIR / "reference_files",
        force_refresh=config.hf_force_refresh,
    )
    return DATASET_PATH


def _extract_completed(path: Path) -> tuple[set[str], int, int]:
    if not path.exists():
        return set(), 0, 0

    ids: set[str] = set()
    ok_count = 0
    failed_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = payload.get("id") or payload.get("case_id")
            if cid is not None:
                ids.add(str(cid))

            status = payload.get("status")
            if status == "ok":
                ok_count += 1
            elif status == "error":
                failed_count += 1
            elif "scoring" in payload or "eval" in payload:
                ok_count += 1
    return ids, ok_count, failed_count


def _sync_to_storage(storage: Storage, out: Path, meta_path: Path) -> None:
    try:
        storage.put(out.name, out)
        storage.put(meta_path.name, meta_path)
    except Exception as exc:  # noqa: BLE001
        log.error(f"[ASYNC SYNC ERROR] {exc}")


def _upload_snapshot_and_cleanup(
    storage: Storage,
    out_name: str,
    out_snapshot: Path,
    meta_name: str,
    meta_snapshot: Path,
) -> None:
    try:
        storage.put(out_name, out_snapshot)
        storage.put(meta_name, meta_snapshot)
    except Exception as exc:  # noqa: BLE001
        log.error(f"[ASYNC SYNC ERROR] {exc}")
    finally:
        for path in (out_snapshot, meta_snapshot):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _schedule_sync_snapshot(storage: Storage, out: Path, meta_path: Path) -> None:
    """Snapshot current files before async upload to avoid read/write races."""
    token = uuid.uuid4().hex
    sync_dir = out.parent / ".sync_tmp"
    sync_dir.mkdir(parents=True, exist_ok=True)
    out_snapshot = sync_dir / f"{out.name}.{token}.snap"
    meta_snapshot = sync_dir / f"{meta_path.name}.{token}.snap"
    shutil.copy2(out, out_snapshot)
    shutil.copy2(meta_path, meta_snapshot)
    fire_and_forget(
        _upload_snapshot_and_cleanup,
        storage,
        out.name,
        out_snapshot,
        meta_path.name,
        meta_snapshot,
    )


def _write_meta(path: Path, **updates: Any) -> None:
    meta: dict[str, Any] = {}
    if path.exists():
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
    meta.update(updates)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.rename(path)


def _expand_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return re.sub(
        r"\$\{(\w+)\}",
        lambda match: os.environ.get(match.group(1), match.group(0)),
        value,
    )


def _load_provider_map(config_path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_yaml_payload(config_path)
    providers = payload.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError(f"Invalid providers map in {config_path}")
    expanded: dict[str, dict[str, Any]] = {}
    for name, cfg in providers.items():
        if isinstance(cfg, dict):
            expanded[str(name)] = {str(k): _expand_env(v) for k, v in cfg.items()}
    return expanded


def _load_yaml_payload(config_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in provider mode only
        raise RuntimeError(
            "PyYAML is required for provider resolution from config.yaml"
        ) from exc

    if not config_path.exists():
        raise ValueError(f"Provider config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config payload in {config_path}")
    return payload


def _load_provider_module(path: Path, cfg: dict[str, Any]) -> ModuleType:
    for key in ("OAICHAT_BASE_URL", "OAICHAT_API_KEY", "OAICHAT_MODEL", "OAICHAT_EXTRA_BODY"):
        os.environ.pop(key, None)

    if path.stem == "oaichat":
        for key in ("base_url", "api_key", "model"):
            value = cfg.get(key)
            if value:
                os.environ[f"OAICHAT_{key.upper()}"] = str(value)
        if cfg.get("extra_body"):
            os.environ["OAICHAT_EXTRA_BODY"] = json.dumps(cfg["extra_body"])

    if path.stem == "openai":
        if cfg.get("model"):
            os.environ["OPENAI_MODEL"] = str(cfg["model"])
        else:
            os.environ.pop("OPENAI_MODEL", None)
        if cfg.get("reasoning_effort"):
            os.environ["OPENAI_REASONING_EFFORT"] = str(cfg["reasoning_effort"])
        else:
            os.environ.pop("OPENAI_REASONING_EFFORT", None)

    if path.stem == "gemini":
        if cfg.get("model"):
            os.environ["GEMINI_MODEL"] = str(cfg["model"])
        else:
            os.environ.pop("GEMINI_MODEL", None)

    if path.stem == "ant_compat":
        for key in ("ANTCOMPAT_BASE_URL", "ANTCOMPAT_API_KEY", "ANTCOMPAT_MODEL"):
            os.environ.pop(key, None)
        for key in ("base_url", "api_key", "model"):
            value = cfg.get(key)
            if value:
                os.environ[f"ANTCOMPAT_{key.upper()}"] = str(value)

    spec: ModuleSpec | None = importlib.util.spec_from_file_location(
        f"async_llm_{path.stem}", str(path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "generate"):
        raise ValueError(f"{path} must export generate(task, references, config)")
    if not hasattr(module, "LLM_ID"):
        raise ValueError(f"{path} must export LLM_ID")
    return module


def _resolve_provider_module(
    provider: str, provider_config_path: str | None
) -> tuple[ModuleType, str]:
    candidate = Path(provider)
    if candidate.exists():
        module = _load_provider_module(candidate, {})
        return module, provider

    llm_path = EVAL_DIR / "llms" / f"{provider}.py"
    if llm_path.exists():
        module = _load_provider_module(llm_path, {})
        return module, provider

    config_path = (
        Path(provider_config_path)
        if provider_config_path
        else (EVAL_DIR / "config.yaml")
    )
    providers = _load_provider_map(config_path)
    provider_cfg = providers.get(provider)
    if provider_cfg is None:
        known = ", ".join(sorted(providers.keys()))
        raise ValueError(f"Unknown provider '{provider}'. Known providers: {known}")

    llm_raw = provider_cfg.get("llm")
    if not llm_raw:
        raise ValueError(f"Provider '{provider}' missing llm path in {config_path}")
    llm_file = Path(str(llm_raw))
    if not llm_file.is_absolute():
        llm_file = BASE_DIR / llm_file
    if not llm_file.exists():
        raise ValueError(f"Provider llm path not found: {llm_file}")
    module = _load_provider_module(llm_file, provider_cfg)
    return module, provider


def _resolve_runtime_config(config: AsyncRunConfig) -> AsyncRunConfig:
    config_path = (
        Path(config.provider_config_path)
        if config.provider_config_path
        else (EVAL_DIR / "config.yaml")
    )
    if not config_path.exists():
        return config

    payload = _load_yaml_payload(config_path)
    sandbox_cfg = payload.get("sandbox", {})
    if not isinstance(sandbox_cfg, dict):
        sandbox_cfg = {}
    judge_cfg = payload.get("judge", {})
    if not isinstance(judge_cfg, dict):
        judge_cfg = {}

    runtime_cfg = payload.get("runtime", {})
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}

    merged_sandbox: dict[str, Any] = dict(sandbox_cfg)
    if config.client == "provider" and config.provider:
        providers = payload.get("providers", {})
        provider_cfg = providers.get(config.provider, {}) if isinstance(providers, dict) else {}
        if not isinstance(provider_cfg, dict):
            provider_cfg = {}
        provider_runtime_cfg = provider_cfg.get("runtime", {})
        if not isinstance(provider_runtime_cfg, dict):
            provider_runtime_cfg = {}
        provider_sandbox_cfg = provider_cfg.get("sandbox", {})
        if not isinstance(provider_sandbox_cfg, dict):
            provider_sandbox_cfg = {}
        merged_sandbox.update(runtime_cfg)
        merged_sandbox.update(provider_runtime_cfg)
        merged_sandbox.update(provider_sandbox_cfg)
    else:
        merged_sandbox.update(runtime_cfg)

    resolved_mode = config.runtime_type or config.repl_mode
    if resolved_mode == "local" and merged_sandbox.get("mode") in {"local", "daytona"}:
        resolved_mode = str(merged_sandbox["mode"])
    if resolved_mode == "local" and merged_sandbox.get("type") in {"local", "daytona"}:
        resolved_mode = str(merged_sandbox["type"])

    resolved_template = config.sandbox_template
    if resolved_template is None and merged_sandbox.get("template"):
        resolved_template = str(merged_sandbox["template"])

    resolved_concurrency = config.sandbox_concurrency
    if resolved_concurrency is None and merged_sandbox.get("concurrency") is not None:
        resolved_concurrency = int(merged_sandbox["concurrency"])

    resolved_judge_provider = config.judge_provider
    if resolved_judge_provider is None and judge_cfg.get("provider"):
        resolved_judge_provider = str(judge_cfg["provider"])

    return replace(
        config,
        runtime_type=resolved_mode,
        repl_mode=resolved_mode,
        sandbox_template=resolved_template,
        sandbox_concurrency=resolved_concurrency,
        judge_provider=resolved_judge_provider,
    )


def _task_prompt(task: dict[str, Any]) -> str:
    if task.get("task"):
        return str(task["task"])
    if task.get("prompt"):
        return str(task["prompt"])
    return ""


def _task_id(task: dict[str, Any], idx: int) -> str:
    value = task.get("id", f"case-{idx:04d}")
    return str(value)


class ConvStore:
    """Thread-safe conversation store that appends only dirty task snapshots."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty: set[str] = set()
        self._load()

    def append(self, task_id: str, entry: dict) -> None:
        """Append a conversation turn for a task (in-memory only until flush)."""
        with self._lock:
            if task_id not in self._data:
                self._data[task_id] = {"conversation": [], "judge": []}
            entry["ts"] = datetime.now(timezone.utc).isoformat()
            self._data[task_id]["conversation"].append(entry)
            self._dirty.add(task_id)

    def append_judge(self, task_id: str, entry: dict) -> None:
        """Append a judge turn for a task (in-memory only until flush)."""
        with self._lock:
            if task_id not in self._data:
                self._data[task_id] = {"conversation": [], "judge": []}
            entry["ts"] = datetime.now(timezone.utc).isoformat()
            self._data[task_id]["judge"].append(entry)
            self._dirty.add(task_id)

    def flush(self) -> None:
        """Append dirty task snapshots to disk."""
        self._flush()

    def _load(self) -> None:
        """Load existing conv data from disk if present."""
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    tid = row.pop("task_id", None)
                    if tid:
                        self._data[tid] = {"conversation": row.get("conversation", []), "judge": row.get("judge", [])}
        except Exception as e:
            log.warning(f"Failed to load conv store: {e}")

    def _flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            dirty_ids = tuple(sorted(self._dirty))
            snapshot = {
                tid: {
                    "conversation": list(self._data[tid]["conversation"]),
                    "judge": list(self._data[tid]["judge"]),
                }
                for tid in dirty_ids
            }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                for tid in dirty_ids:
                    data = snapshot[tid]
                    line = {"task_id": tid, **data}
                    f.write(json.dumps(line, default=str) + "\n")
                f.flush()
            with self._lock:
                self._dirty.difference_update(dirty_ids)
        except Exception as e:
            log.warning(f"Failed to flush conv store: {e}")


def _prepare_task_output_dir(task: dict[str, Any]) -> str:
    output_dir = task.get("output_dir") or "artifacts"
    from eval.core import _clear_output_dir, _resolve_output_dir
    path = _resolve_output_dir(str(output_dir), task_id=str(task.get("id", "")))
    path.mkdir(parents=True, exist_ok=True)
    _clear_output_dir(path)
    return str(path)


def _load_references(task: dict[str, Any]) -> dict[str, Any]:
    ref_paths = task.get("reference_files")
    if not ref_paths:
        return {"inline": {}, "paths": {}}
    from eval.core import read_reference_files

    return read_reference_files(ref_paths)


def _resolve_reference_source_path(rel_path: str) -> Path | None:
    candidate = BASE_DIR / rel_path
    if candidate.exists():
        return candidate
    candidate = BASE_DIR / "reference_files" / rel_path
    if candidate.exists():
        return candidate
    return None


def _reference_var_base(rel_path: str) -> str:
    stem = Path(rel_path).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "reference"


def _extract_docx_tables(path: Path) -> list[list[list[str]]]:
    from docx import Document

    doc = Document(str(path))
    tables: list[list[list[str]]] = []
    for table in doc.tables:
        rows: list[list[str]] = []
        for row in table.rows:
            rows.append([cell.text.strip() for cell in row.cells])
        tables.append(rows)
    return tables


def _reference_seed_globals(task: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    ref_paths = task.get("reference_files")
    if not ref_paths:
        return {}, None

    seed: dict[str, Any] = {}
    parsed_refs: dict[str, dict[str, Any]] = {}
    convenience_vars: list[str] = []

    from eval.core import _parse_docx

    for rel_path in ref_paths:
        resolved = _resolve_reference_source_path(str(rel_path))
        if resolved is None:
            continue

        entry: dict[str, Any] = {"path": str(resolved.resolve())}
        var_base = _reference_var_base(str(rel_path))
        ext = resolved.suffix.lower()

        if ext == ".docx":
            try:
                text = _parse_docx(resolved)
                tables = _extract_docx_tables(resolved)
            except Exception:
                continue
            entry["text"] = text
            entry["tables"] = tables
            seed[f"{var_base}_text"] = text
            seed[f"{var_base}_tables"] = tables
            convenience_vars.extend([f"{var_base}_text", f"{var_base}_tables"])

        if entry.keys() != {"path"}:
            parsed_refs[str(rel_path)] = entry

    if not parsed_refs:
        note = (
            "Reference documents are already parsed below. "
            "Do not try to reopen the original reference files from disk."
        )
        return {}, note

    seed["_reference_files"] = parsed_refs
    note = (
        "Reference documents are already parsed below. "
        "Do not try to reopen the original reference files from disk. "
        "In the Python REPL, parsed reference data is available via `_reference_files`."
    )
    if convenience_vars:
        preview = ", ".join(f"`{name}`" for name in convenience_vars[:4])
        note += f" Convenience variables are also available: {preview}."
    note += " Prefer those preloaded values over rediscovering file paths."
    return seed, note


def _apply_allowed_tools(case_config: dict[str, Any], allowed_tools: list[str] | None) -> None:
    if allowed_tools is None:
        return
    allowed = {str(tool).strip().lower() for tool in allowed_tools}
    case_config["enable_code"] = "code" in allowed or "python" in allowed
    case_config["enable_bash"] = "bash" in allowed or "shell" in allowed
    case_config["enable_search"] = "search" in allowed or "web_search" in allowed


def _resolve_execution_profile(
    task: dict[str, Any],
    *,
    benchmark_plugin: Any | None,
    runtime_type: str,
    local_output_dir: str,
    prompt_repl_note: str | None,
) -> dict[str, Any]:
    profile = ExecutionProfile()
    if benchmark_plugin is not None and hasattr(benchmark_plugin, "execution_profile"):
        plugin_profile = benchmark_plugin.execution_profile(task)
        if plugin_profile is not None:
            profile = plugin_profile

    task_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(task.get("id", ""))).strip("-") or "case"
    prompt_output_dir: str | None = None
    bash_cwd: str | None = None
    python_cwd: str | None = None
    write_redirect_dir: str | None = None
    write_mode: str | None = None
    remote_workspace_root: str | None = None
    remote_write_redirect_dir: str | None = None
    remote_write_mode: str | None = None
    remote_sync_dir: str | None = None

    prompt_parts = [part for part in (prompt_repl_note, profile.prompt_extra_instructions) if part]

    if runtime_type == "daytona":
        remote_root = f".kwbench/tasks/{task_slug}"
        remote_artifact_dir = f"{remote_root}/output"
        remote_workspace_root = profile.workspace_root or remote_artifact_dir
        bash_cwd = profile.bash_cwd or remote_workspace_root
        python_cwd = profile.python_cwd or remote_workspace_root
        if profile.python_write_policy == "artifact_dir":
            remote_write_redirect_dir = remote_artifact_dir
            remote_write_mode = "redirect"
        elif profile.python_write_policy == "cwd":
            remote_write_redirect_dir = python_cwd
            remote_write_mode = "strict"
        if profile.sync_policy == "artifact_dir":
            remote_sync_dir = remote_artifact_dir
        if profile.prompt_output_policy == "artifact_dir":
            prompt_output_dir = remote_artifact_dir
            prompt_parts.append(
                "Python and bash tools run inside a Daytona sandbox for this task. "
                f"Use the sandbox paths listed above and write outputs only inside `{remote_artifact_dir}`. "
                f"Artifacts are synced back to the local task directory after execution: `{local_output_dir}`."
            )
        else:
            prompt_parts.append(
                "Python and bash tools run inside a Daytona sandbox for this task. "
                f"Use the sandbox workspace `{remote_workspace_root}` and create files at the benchmark-specified paths. "
                "Do not relocate benchmark outputs into the harness artifact directory."
            )
    else:
        workspace_root = profile.workspace_root or local_output_dir
        bash_cwd = profile.bash_cwd or workspace_root
        python_cwd = profile.python_cwd or workspace_root
        if profile.python_write_policy == "artifact_dir":
            write_redirect_dir = local_output_dir
            write_mode = "redirect"
        elif profile.python_write_policy == "cwd":
            write_redirect_dir = python_cwd
            write_mode = "strict"
        if profile.prompt_output_policy == "artifact_dir":
            prompt_output_dir = local_output_dir

    return {
        "profile": profile,
        "task_slug": task_slug,
        "prompt_output_dir": prompt_output_dir,
        "prompt_note": " ".join(prompt_parts) if prompt_parts else None,
        "bash_cwd": bash_cwd,
        "python_cwd": python_cwd,
        "write_redirect_dir": write_redirect_dir,
        "write_mode": write_mode,
        "remote_workspace_root": remote_workspace_root,
        "remote_write_redirect_dir": remote_write_redirect_dir,
        "remote_write_mode": remote_write_mode,
        "remote_sync_dir": remote_sync_dir,
    }


def _provider_case_context(
    task: dict[str, Any],
    *,
    conv_store: ConvStore | None = None,
    benchmark_plugin: Any | None = None,
) -> ProviderCaseContext:
    from eval.tools import make_tool_runtime, make_tool_session

    cfg = task.get("config", {})
    case_config = dict(cfg) if isinstance(cfg, dict) else {}
    task_id = str(task.get("id", ""))
    case_config["_task_id"] = task_id
    try:
        case_config["_max_turns"] = int(task.get("max_steps", case_config.get("case_max_steps", 0)))
    except (TypeError, ValueError):
        pass
    if benchmark_plugin is not None:
        allowed_tools = benchmark_plugin.allowed_tools(task)
        _apply_allowed_tools(case_config, allowed_tools)

    output_dir = _prepare_task_output_dir(task)
    plugin_context = None
    if benchmark_plugin is not None:
        plugin_context = benchmark_plugin.build_case_context(task)
    references = plugin_context if plugin_context is not None else _load_references(task)
    seed_globals, prompt_repl_note = _reference_seed_globals(task)
    prompt_text = _task_prompt(task)
    runtime_type = str(case_config.get("runtime_type") or case_config.get("repl_mode") or "local")
    if benchmark_plugin is not None:
        prompt_override = benchmark_plugin.build_prompt(task, references)
        if prompt_override is not None:
            prompt_text = prompt_override
    profile_ctx = _resolve_execution_profile(
        task,
        benchmark_plugin=benchmark_plugin,
        runtime_type=runtime_type,
        local_output_dir=output_dir,
        prompt_repl_note=prompt_repl_note,
    )

    runtime = make_tool_runtime(runtime_type)
    case_config["_tool_runtime"] = runtime
    case_config["_runtime_type"] = runtime_type

    if runtime_type == "daytona":
        task_slug = profile_ctx["task_slug"]
        remote_root = f".kwbench/tasks/{task_slug}"
        remote_reference_files: dict[str, str] = {}
        prompt_reference_paths: dict[str, str] = {}
        for ref_name, local_path in references.get("paths", {}).items():
            remote_path = f"{remote_root}/refs/{Path(ref_name).as_posix()}"
            remote_reference_files[str(local_path)] = remote_path
            prompt_reference_paths[str(ref_name)] = remote_path

        references = {
            **references,
            "paths": prompt_reference_paths,
        }
        tool_session = make_tool_session(
            runtime_type="daytona",
            task_id=task_slug,
            seed_globals=seed_globals or None,
            output_dir=output_dir,
            remote_output_dir=profile_ctx["prompt_output_dir"],
            working_dir=profile_ctx["python_cwd"],
            write_redirect_dir=profile_ctx["write_redirect_dir"],
            write_mode=profile_ctx["write_mode"],
            remote_working_dir=profile_ctx["remote_workspace_root"],
            remote_write_redirect_dir=profile_ctx["remote_write_redirect_dir"],
            remote_write_mode=profile_ctx["remote_write_mode"],
            remote_sync_dir=profile_ctx["remote_sync_dir"],
            remote_reference_files=remote_reference_files,
            sandbox_template=case_config.get("sandbox_template"),
        )
    else:
        tool_session = make_tool_session(
            runtime_type="local",
            task_id=task_id,
            seed_globals=seed_globals or None,
            output_dir=output_dir,
            working_dir=profile_ctx["python_cwd"],
            write_redirect_dir=profile_ctx["write_redirect_dir"],
            write_mode=profile_ctx["write_mode"],
        )

    case_config["_tool_session"] = tool_session
    case_config["_repl"] = tool_session
    case_config["_output_dir"] = profile_ctx["prompt_output_dir"]
    case_config["_local_output_dir"] = output_dir
    case_config["_bash_cwd"] = profile_ctx["bash_cwd"]
    case_config["_python_cwd"] = profile_ctx["python_cwd"]
    case_config["_write_redirect_dir"] = profile_ctx["write_redirect_dir"]
    case_config["_write_mode"] = profile_ctx["write_mode"]
    case_config["_remote_workspace_root"] = profile_ctx["remote_workspace_root"]
    case_config["_remote_write_redirect_dir"] = profile_ctx["remote_write_redirect_dir"]
    case_config["_remote_write_mode"] = profile_ctx["remote_write_mode"]
    case_config["_remote_sync_dir"] = profile_ctx["remote_sync_dir"]
    if profile_ctx["prompt_note"]:
        case_config["_prompt_repl_note"] = profile_ctx["prompt_note"]
    if conv_store:
        case_config["_conv_store"] = conv_store

    return ProviderCaseContext(
        task_text=prompt_text,
        references=references,
        config=case_config,
    )


def _to_case(task: dict[str, Any], idx: int, config: AsyncRunConfig) -> EvalCase:
    task_config = task.get("config", {})
    if not isinstance(task_config, dict):
        task_config = {}

    tool_mode = str(task.get("tool_mode", config.case_tool_mode))
    tool_payload = task.get("tool_payload", config.case_tool_payload)
    max_steps = int(task.get("max_steps", config.case_max_steps))

    metadata: dict[str, Any] = {
        "source": task.get("source"),
        "category": task.get("category"),
        "task_config": task_config,
    }

    # Extended schema fields (all optional, absent = current behavior)
    for key in (
        "ground_truth", "match_type", "case_sensitive", "normalize",
        "verifier", "setup", "environment", "session",
        "allowed_tools", "artifact_paths", "timeout",
    ):
        value = task.get(key)
        if value is not None:
            metadata[key] = value

    return EvalCase(
        case_id=_task_id(task, idx),
        prompt=_task_prompt(task),
        tool_mode=tool_mode,
        tool_payload=tool_payload,
        max_steps=max_steps,
        metadata=metadata,
    )


def _result_row(
    *,
    task: dict[str, Any],
    result: CaseResult,
    llm_id: str,
    llm_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(task)
    row.update(
        {
            "id": str(task.get("id") or result.case_id),
            "case_id": result.case_id,
            "task": task.get("task", ""),
            "source": task.get("source"),
            "category": task.get("category"),
            "llm_answer": result.output_text,
            "status": result.status,
            "error": result.error,
            "metrics": {
                "model_wait_s": result.model_wait_s,
                "tool_cpu_s": result.tool_cpu_s,
                "total_s": result.total_s,
                "model_calls": result.model_calls,
                "tool_calls": result.tool_calls,
                "peak_in_flight_evals": result.peak_in_flight_evals,
                "peak_in_flight_tools": result.peak_in_flight_tools,
                "started_at_s": result.started_at_s,
                "finished_at_s": result.finished_at_s,
            },
            "runner": "async",
            "llm_id": llm_id,
            "eval_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    if llm_metadata is not None:
        row["llm_metadata"] = llm_metadata
    thinking = _extract_thinking(llm_metadata)
    if thinking is not None:
        row["thinking"] = thinking
    # scoring field is populated later by scorer dispatch
    return row


def _extract_thinking(llm_metadata: dict[str, Any] | None) -> Any | None:
    if not isinstance(llm_metadata, dict):
        return None

    for key in ("thinking", "reasoning"):
        value = llm_metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value:
            return value

    turns = llm_metadata.get("turns")
    if not isinstance(turns, list):
        return None

    extracted: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        value = turn.get("thinking") or turn.get("reasoning")
        if not value:
            continue
        turn_idx = turn.get("turn")
        if turn_idx is None:
            extracted.append({"thinking": value})
        else:
            extracted.append({"turn": turn_idx, "thinking": value})
    return extracted or None


def _build_judged_answer(task: dict[str, Any], llm_answer: str) -> str:
    output_dir_name = task.get("output_dir")
    if not output_dir_name:
        return llm_answer

    from eval.core import build_judge_artifact_bundle

    bundle = build_judge_artifact_bundle(str(output_dir_name), task_id=str(task.get("id", "")))
    prompt_context = bundle.get("prompt_context")
    if not prompt_context:
        return llm_answer
    return f"{llm_answer}\n\n--- ARTIFACT CONTEXT ---\n{prompt_context}"


def _judge_task_with_rubric(
    task: dict[str, Any],
    llm_answer: str,
    *,
    criterion_workers: int,
    rubric: dict[str, list[Any]] | None = None,
    conv_store: Any = None,
    judge_provider: str | None = None,
    provider_config_path: str | None = None,
    repl_mode: str = "local",
    sandbox_template: str | None = None,
) -> dict[str, Any] | None:
    if rubric is None:
        rubric = _sanitize_rubric(task.get("rubric"))
    if rubric is None:
        return None

    from eval.core import build_judge_artifact_bundle, judge_rubric, score_rubric

    task_id = str(task.get("id", ""))
    output_dir_name = task.get("output_dir") or "artifacts"
    artifact_bundle = build_judge_artifact_bundle(str(output_dir_name), task_id=task_id)
    eval_results = judge_rubric(
        str(task.get("task", "")),
        llm_answer,
        rubric,
        criterion_workers=criterion_workers,
        repl_seed=artifact_bundle["repl_seed"] or None,
        output_dir=artifact_bundle["artifact_root"],
        artifact_context=artifact_bundle["prompt_context"],
        conv_store=conv_store,
        task_id=task_id,
        judge_provider=judge_provider,
        provider_config_path=provider_config_path,
        repl_mode=repl_mode,
        sandbox_template=sandbox_template,
    )
    task_score = score_rubric(
        eval_results["mandatory"],
        eval_results["good_to_have"],
        eval_results["ideal"],
    )
    return {**eval_results, "score": task_score}


def _sanitize_rubric(raw_rubric: Any) -> dict[str, list[Any]] | None:
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    if not isinstance(raw_rubric, dict):
        return None
    mandatory = _as_list(raw_rubric.get("mandatory"))
    good_to_have = _as_list(raw_rubric.get("good_to_have"))
    ideal = _as_list(raw_rubric.get("ideal"))
    if not any((mandatory, good_to_have, ideal)):
        return None
    return {
        "mandatory": mandatory,
        "good_to_have": good_to_have,
        "ideal": ideal,
    }


def _cpu_limit(cpu_sem: int | None) -> int:
    return min(8, os.cpu_count() or 1) if cpu_sem is None else cpu_sem


def _runtime_type(config: AsyncRunConfig) -> str:
    return config.runtime_type or config.repl_mode


def _tool_limit(config: AsyncRunConfig) -> int:
    if _runtime_type(config) == "daytona" and config.sandbox_concurrency is not None:
        return config.sandbox_concurrency
    return _cpu_limit(config.cpu_sem)


def _resolve_weave_project(config: AsyncRunConfig) -> str | None:
    return config.weave_project or os.environ.get("WEAVE_PROJECT")


def _init_weave(config: AsyncRunConfig) -> Any:
    project = _resolve_weave_project(config)
    if not project:
        return None
    try:
        import weave  # type: ignore
    except ImportError:
        log.warning("[ASYNC] WEAVE_PROJECT is set but weave is not installed; skipping Weave tracing")
        return None
    try:
        entity = os.environ.get("WANDB_ENTITY") or os.environ.get("WEAVE_ENTITY")
        if entity and "/" not in project:
            project = f"{entity}/{project}"
        return weave.init(project)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[ASYNC] Failed to initialize Weave: {type(exc).__name__}: {exc}")
        return None


def _resolve_wandb_project(config: AsyncRunConfig) -> str | None:
    return config.wandb_project or os.environ.get("WANDB_PROJECT")


def _init_wandb(
    *,
    run_id: str,
    llm_id: str,
    total_tasks: int,
    dataset_path: Path,
    config: AsyncRunConfig,
):
    project = _resolve_wandb_project(config)
    if not project:
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        log.warning("[ASYNC] WANDB_PROJECT is set but wandb is not installed; skipping W&B logging")
        return None

    kwargs: dict[str, Any] = {
        "project": project,
        "name": run_id,
        "id": run_id,
        "resume": "allow",
        "config": {
            "runner": "async",
            "llm_id": llm_id,
            "total_tasks": total_tasks,
            "dataset": str(dataset_path),
            "client": config.client,
            "eval_sem": config.eval_sem,
            "cpu_sem": config.cpu_sem,
            "runtime_type": _runtime_type(config),
            "repl_mode": config.repl_mode,
            "sandbox_concurrency": config.sandbox_concurrency,
            "judge_enabled": config.judge_enabled,
            "judge_provider": config.judge_provider,
            "judge_sem": config.judge_sem,
            "judge_criterion_workers": config.judge_criterion_workers,
        },
    }
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        kwargs["entity"] = entity
    mode = os.environ.get("WANDB_MODE")
    if mode:
        kwargs["mode"] = mode
    group = os.environ.get("WANDB_RUN_GROUP")
    if group:
        kwargs["group"] = group
    tags = os.environ.get("WANDB_TAGS")
    if tags:
        kwargs["tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]

    try:
        return wandb.init(**kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[ASYNC] Failed to initialize W&B: {type(exc).__name__}: {exc}")
        return None


def _wandb_usage_tokens(llm_metadata: dict[str, Any] | None) -> int | None:
    if not isinstance(llm_metadata, dict):
        return None
    usage = llm_metadata.get("usage")
    if not isinstance(usage, dict):
        return None
    if isinstance(usage.get("total_tokens"), (int, float)):
        return int(usage["total_tokens"])
    in_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    out_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    if isinstance(in_tokens, (int, float)) and isinstance(out_tokens, (int, float)):
        total = int(in_tokens + out_tokens)
        return total if total > 0 else None
    return None


def _hf_results_token(config: AsyncRunConfig) -> str | None:
    return (
        config.hf_results_token
        or os.environ.get("HF_RESULTS_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


def _provider_slug(config: AsyncRunConfig, llm_id: str) -> str:
    raw = (
        config.provider
        if config.client == "provider" and config.provider
        else llm_id
    )
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw)).strip("-").lower()
    return slug or "unknown-provider"


def _write_run_meta_jsonl(
    *,
    path: Path,
    meta_path: Path,
    run_id: str,
    llm_id: str,
    provider: str,
    output_path: Path,
) -> None:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "provider": provider,
        "llm_id": llm_id,
        "output_file": output_path.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if meta_path.exists():
        try:
            payload["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload["meta"] = {"error": "meta_read_failed"}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _upload_results_to_hf_dataset(
    *,
    repo_id: str,
    token: str,
    provider: str,
    date_key: str,
    results_path: Path,
    run_meta_jsonl_path: Path,
) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for result uploads; install it to enable hf_results_upload"
        ) from exc

    api = HfApi(token=token)
    folder = f"{provider}/{date_key}"
    paths = {
        "results_jsonl": f"{folder}/results.jsonl",
        "run_meta_jsonl": f"{folder}/run_meta.jsonl",
    }
    api.upload_file(
        path_or_fileobj=str(results_path),
        path_in_repo=paths["results_jsonl"],
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Add eval results {provider}/{date_key}",
    )
    api.upload_file(
        path_or_fileobj=str(run_meta_jsonl_path),
        path_in_repo=paths["run_meta_jsonl"],
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Add eval run meta {provider}/{date_key}",
    )
    return {
        "repo_id": repo_id,
        "folder": folder,
        "files": paths,
    }


def _build_client(
    config: AsyncRunConfig,
    *,
    task_by_id: dict[str, dict[str, Any]],
    conv_store: ConvStore | None = None,
    benchmark_plugin: Any | None = None,
) -> tuple[str, LLMClient]:
    if config.client == "fake":
        return (
            "async-fake",
            FakeLLMClient(
                base_latency_s=config.fake_base_latency_s,
                jitter_s=config.fake_jitter_s,
                tool_ratio=config.fake_tool_ratio,
            ),
        )

    if config.client == "replay":
        if not config.replay_fixtures:
            raise ValueError("replay_fixtures is required when client='replay'")
        return ("async-replay", ReplayLLMClient.from_json(config.replay_fixtures))

    if config.client == "provider":
        if not config.provider:
            raise ValueError("provider is required when client='provider'")
        module, provider_name = _resolve_provider_module(
            config.provider, config.provider_config_path
        )
        llm_id = str(getattr(module, "LLM_ID", f"provider-{provider_name}"))
        generate_fn = getattr(module, "generate_async", None) or module.generate
        mode = "async" if getattr(module, "generate_async", None) else "sync"
        log.info(f"[ASYNC] provider={provider_name} llm_id={llm_id} generate_mode={mode}")

        def _context_factory(case_id: str) -> ProviderCaseContext:
            task = task_by_id.get(case_id)
            if task is None:
                raise KeyError(f"Missing provider task for case_id={case_id!r}")
            task_cfg = task.get("config", {})
            if not isinstance(task_cfg, dict):
                task_cfg = {}
            task_with_runtime = dict(task)
            task_with_runtime["config"] = {
                **task_cfg,
                "runtime_type": _runtime_type(config),
                "repl_mode": config.repl_mode,
                "sandbox_template": config.sandbox_template,
                "case_max_steps": int(task.get("max_steps", config.case_max_steps)),
            }
            context_kwargs: dict[str, Any] = {"conv_store": conv_store}
            if benchmark_plugin is not None:
                context_kwargs["benchmark_plugin"] = benchmark_plugin
            context = _provider_case_context(task_with_runtime, **context_kwargs)
            if benchmark_plugin is not None:
                async def _prepare_case_once() -> None:
                    repl = context.config.get("_tool_session") or context.config.get("_repl")
                    await benchmark_plugin.prepare_case(task, repl)

                context.config["_prepare_case"] = _prepare_case_once
            return context

        return (
            llm_id,
            ProviderLLMClient(
                generate_fn=generate_fn,
                case_context_factory=_context_factory,
            ),
        )

    raise ValueError(f"Unsupported client: {config.client}")


def run_async_eval(
    config: AsyncRunConfig,
    *,
    dataset_path: Path | None = None,
    output_path: Path | None = None,
    task_ids: set[str] | None = None,
    storage: Storage | None = None,
) -> Path:
    return asyncio.run(
        _run_async_eval(
            config,
            dataset_path=dataset_path,
            output_path=output_path,
            task_ids=task_ids,
            storage=storage,
        )
    )


async def _run_async_eval(
    config: AsyncRunConfig,
    *,
    dataset_path: Path | None,
    output_path: Path | None,
    task_ids: set[str] | None,
    storage: Storage | None,
) -> Path:
    config = _resolve_runtime_config(config)

    # Resolve benchmark plugin (if specified)
    benchmark_plugin = None
    if config.benchmark:
        from eval.benchmarks import get_plugin
        benchmark_plugin = get_plugin(config.benchmark)

    resolved_dataset = _ensure_dataset_materialized(
        dataset_path=dataset_path,
        config=config,
    )

    if benchmark_plugin is not None:
        # Plugins own row conversion, but they should still receive the materialized
        # local dataset path when the engine fetched or restored it.
        tasks = benchmark_plugin.load_cases(resolved_dataset)
    else:
        tasks = load_dataset(resolved_dataset)

    if resolved_dataset is None:
        resolved_dataset = DATASET_PATH
    if task_ids:
        tasks = [task for task in tasks if str(task.get("id")) in task_ids]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = output_path or (RESULTS_DIR / f"async-{config.client}_{ts}.jsonl")
    meta_path = out.with_suffix(".meta.json")
    run_id = out.stem

    if storage and not out.exists():
        if storage.get(out.name, out):
            log.info(f"[ASYNC] Restored {out.name} from remote storage")
        storage.get(meta_path.name, meta_path)

    completed_ids, ok_count, failed_count = _extract_completed(out)
    all_task_count = len(tasks)
    remaining = [
        (idx, task)
        for idx, task in enumerate(tasks)
        if _task_id(task, idx) not in completed_ids
    ]
    remaining_cases = [_to_case(task, idx, config) for idx, task in remaining]
    task_by_id = {
        case.case_id: task for case, (_, task) in zip(remaining_cases, remaining, strict=True)
    }
    conv_store = ConvStore(out.with_suffix(".conv.jsonl"))
    llm_id, client = _build_client(
        config,
        task_by_id=task_by_id,
        conv_store=conv_store,
        benchmark_plugin=benchmark_plugin,
    )
    weave_client = _init_weave(config)
    wb_run = _init_wandb(
        run_id=run_id,
        llm_id=llm_id,
        total_tasks=all_task_count,
        dataset_path=resolved_dataset,
        config=config,
    )

    provider_tool_getter = None
    provider_tool_resetter = None
    provider_tool_setter = None
    if config.client == "provider":
        from eval.tools import (
            get_tool_metrics,
            reset_tool_metrics,
            set_tool_concurrency,
        )

        provider_tool_getter = get_tool_metrics
        provider_tool_resetter = reset_tool_metrics
        provider_tool_setter = set_tool_concurrency
        provider_tool_setter(_tool_limit(config))
        provider_tool_resetter()

    judge_case_limit = max(1, config.judge_sem) if config.judge_enabled else 1
    judge_criterion_workers = max(1, config.judge_criterion_workers)
    log.info(
        f"[ASYNC] Eval: {llm_id} ({config.client}) | {len(remaining_cases)}/{all_task_count} tasks "
        f"| eval_sem={config.eval_sem} cpu_sem={config.cpu_sem} runtime={_runtime_type(config)}"
        f" judge={'on' if config.judge_enabled else 'off'}"
        f" judge_provider={config.judge_provider or 'default'}"
        f" judge_sem={judge_case_limit} judge_criterion_workers={judge_criterion_workers} | -> {out}"
    )

    started_at = datetime.now(timezone.utc).isoformat()
    _write_meta(
        meta_path,
        run_id=run_id,
        runner="async",
        llm_id=llm_id,
        status="running",
        output=str(out),
        total=all_task_count,
        completed=len(completed_ids),
        ok=ok_count,
        failed=failed_count,
        started_at=started_at,
        last_progress_at=started_at,
        heartbeat_at=started_at,
        current_in_flight_evals=0,
        current_in_flight_tools=0,
        current_in_flight_judges=0,
        oldest_in_flight_eval_case_id=None,
        oldest_in_flight_eval_age_s=None,
        wandb_project=_resolve_wandb_project(config),
        config=asdict(config),
    )

    write_lock = asyncio.Lock()
    completed_count = len(completed_ids)
    total_model_wait_s = 0.0
    total_tool_cpu_s = 0.0
    total_elapsed_s = 0.0
    judge_lock = asyncio.Lock()
    in_flight_judges = 0
    peak_in_flight_judges = 0
    last_progress_at = started_at
    active_eval_started_at: dict[str, float] = {}

    runner = Runner(
        llm_client=client,
        eval_sem=config.eval_sem,
        cpu_sem=config.cpu_sem,
        max_retries=config.max_retries,
        retry_base_s=config.retry_base_s,
    )

    def _peak_tools() -> int:
        peak = runner.peak_in_flight_tools
        if provider_tool_getter is not None:
            metrics = provider_tool_getter()
            peak = max(peak, int(metrics.get("peak_in_flight_tools", 0)))
        return peak

    def _sync() -> None:
        if storage:
            _schedule_sync_snapshot(storage, out, meta_path)

    def _live_meta_updates(*, include_current_task: bool = False) -> dict[str, Any]:
        now_wall = datetime.now(timezone.utc).isoformat()
        now_perf = time.perf_counter()
        oldest_case_id = None
        oldest_age_s = None
        if active_eval_started_at:
            oldest_case_id, oldest_started = min(
                active_eval_started_at.items(),
                key=lambda item: item[1],
            )
            oldest_age_s = round(now_perf - oldest_started, 3)
        updates: dict[str, Any] = {
            "heartbeat_at": now_wall,
            "last_progress_at": last_progress_at,
            "current_in_flight_evals": runner.current_in_flight_evals,
            "current_in_flight_tools": runner.current_in_flight_tools,
            "current_in_flight_judges": in_flight_judges,
            "oldest_in_flight_eval_case_id": oldest_case_id,
            "oldest_in_flight_eval_age_s": oldest_age_s,
        }
        if include_current_task and oldest_case_id is not None:
            updates["current_task"] = oldest_case_id
        return updates

    metadata_popper = getattr(client, "pop_case_metadata", None)
    metadata_getter = getattr(client, "get_case_metadata", None)
    case_context_getter = getattr(client, "get_case_context", None)
    case_releaser = getattr(client, "release_case", None)
    result_queue: asyncio.Queue[CaseResult | None] = asyncio.Queue()
    judge_sem = asyncio.Semaphore(judge_case_limit)
    persist_error: Exception | None = None
    stop_heartbeat = asyncio.Event()
    async def _score_case(
        task: dict[str, Any],
        result: CaseResult,
        row: dict[str, Any],
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> None:
        """Resolve scorer and apply it. Rubric scoring uses judge_sem; deterministic scorers run inline."""
        nonlocal in_flight_judges, peak_in_flight_judges

        if result.status != "ok":
            return

        if benchmark_plugin is not None:
            plugin_score = await asyncio.to_thread(
                benchmark_plugin.score_case, task, result.output_text, artifacts,
            )
            if plugin_score is not None:
                row["scoring"] = plugin_score.to_dict()
                return

        def _rubric_scorer_factory() -> RubricJudgeScorer:
            return RubricJudgeScorer(
                criterion_workers=judge_criterion_workers,
                conv_store=conv_store,
                judge_provider=config.judge_provider,
                provider_config_path=config.provider_config_path,
                repl_mode=config.repl_mode,
                sandbox_template=config.sandbox_template,
            )

        scorer_task = task
        if not config.judge_enabled and task.get("ground_truth") is not None:
            scorer_task = dict(task)
            scorer_task.pop("rubric", None)

        scorer = resolve_scorer(
            scorer_task,
            rubric_scorer_factory=_rubric_scorer_factory if config.judge_enabled else None,
        )

        # Deterministic / custom scorers: off the event loop
        if scorer.method != "rubric_judge":
            scorer_result = await asyncio.to_thread(
                scorer.score, scorer_task, result.output_text, artifacts,
            )
            row["scoring"] = scorer_result.to_dict()
            return

        # Rubric judge scorer: gate with judge_sem, run in thread
        if not config.judge_enabled:
            return
        rubric = _sanitize_rubric(task.get("rubric"))
        if rubric is None:
            return

        async with judge_sem:
            async with judge_lock:
                in_flight_judges += 1
                if in_flight_judges > peak_in_flight_judges:
                    peak_in_flight_judges = in_flight_judges
            try:
                eval_payload = await asyncio.to_thread(
                    _judge_task_with_rubric,
                    task,
                    result.output_text,
                    criterion_workers=judge_criterion_workers,
                    rubric=rubric,
                    conv_store=conv_store,
                    judge_provider=config.judge_provider,
                    provider_config_path=config.provider_config_path,
                    repl_mode=config.repl_mode,
                    sandbox_template=config.sandbox_template,
                )
            finally:
                async with judge_lock:
                    in_flight_judges -= 1

        if eval_payload is not None:
            row["eval"] = eval_payload
            row["scoring"] = {
                "method": "rubric_judge",
                "score": eval_payload.get("score", 0.0),
                "detail": eval_payload,
                "judge_provider": config.judge_provider,
            }

    async def _persist_result(result: CaseResult) -> None:
        nonlocal completed_count, ok_count, failed_count, last_progress_at
        nonlocal total_model_wait_s, total_tool_cpu_s, total_elapsed_s

        task = task_by_id[result.case_id]
        llm_metadata = None
        if callable(metadata_popper):
            llm_metadata = metadata_popper(result.case_id)
        elif callable(metadata_getter):
            llm_metadata = metadata_getter(result.case_id)

        row = _result_row(
            task=task,
            result=result,
            llm_id=llm_id,
            llm_metadata=llm_metadata,
        )
        case_context = case_context_getter(result.case_id) if callable(case_context_getter) else None
        repl = case_context.config.get("_repl") if case_context is not None else None
        tool_session = case_context.config.get("_tool_session") if case_context is not None else repl
        tool_runtime = case_context.config.get("_tool_runtime") if case_context is not None else None
        local_output_dir = None
        if case_context is not None:
            local_output_dir = case_context.config.get("_local_output_dir") or case_context.config.get("_output_dir")
        if tool_session is not None and local_output_dir and hasattr(tool_session, "sync_outputs"):
            await asyncio.to_thread(tool_session.sync_outputs, str(local_output_dir))
        artifacts: dict[str, Any] = {}
        if local_output_dir:
            artifacts["output_dir"] = str(local_output_dir)
        if repl is not None:
            artifacts["repl"] = repl
        if tool_session is not None:
            artifacts["tool_session"] = tool_session
        if tool_runtime is not None:
            artifacts["tool_runtime"] = tool_runtime
        try:
            await _score_case(task, result, row, artifacts=artifacts)
        except Exception as exc:  # noqa: BLE001
            row["judge_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            task_by_id.pop(result.case_id, None)
            if callable(case_releaser):
                await asyncio.to_thread(case_releaser, result.case_id)

        # Append judge summary to conv store
        if conv_store and row.get("eval"):
            conv_store.append_judge(str(task.get("id", "")), {"summary": row["eval"]})
        elif conv_store and row.get("judge_error"):
            conv_store.append_judge(str(task.get("id", "")), {"error": row["judge_error"]})

        async with write_lock:
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            if conv_store:
                await asyncio.to_thread(conv_store.flush)

            completed_count += 1
            total_model_wait_s += result.model_wait_s
            total_tool_cpu_s += result.tool_cpu_s
            total_elapsed_s += result.total_s
            if result.status == "ok":
                ok_count += 1
            else:
                failed_count += 1
            last_progress_at = datetime.now(timezone.utc).isoformat()

            _write_meta(
                meta_path,
                completed=completed_count,
                ok=ok_count,
                failed=failed_count,
                current_task=result.case_id,
                peak_in_flight_evals=runner.peak_in_flight_evals,
                peak_in_flight_tools=_peak_tools(),
                peak_in_flight_judges=peak_in_flight_judges,
                total_model_wait_s=round(total_model_wait_s, 4),
                total_tool_cpu_s=round(total_tool_cpu_s, 4),
                total_elapsed_s=round(total_elapsed_s, 4),
                **_live_meta_updates(),
            )
            _sync()

            if wb_run is not None:
                wb_data: dict[str, Any] = {
                    "completed": completed_count,
                    "ok_count": ok_count,
                    "failed_count": failed_count,
                    "status_ok": 1 if result.status == "ok" else 0,
                    "model_wait_s": result.model_wait_s,
                    "tool_cpu_s": result.tool_cpu_s,
                    "total_s": result.total_s,
                }
                scoring_payload = row.get("scoring")
                if isinstance(scoring_payload, dict) and isinstance(scoring_payload.get("score"), (int, float)):
                    wb_data["score"] = float(scoring_payload["score"])
                eval_payload = row.get("eval")
                if isinstance(eval_payload, dict):
                    mandatory = eval_payload.get("mandatory")
                    if isinstance(mandatory, list) and mandatory:
                        mandatory_true = sum(1 for x in mandatory if bool(x))
                        wb_data["mandatory_pass_rate"] = mandatory_true / len(mandatory)

                tokens = _wandb_usage_tokens(llm_metadata)
                if tokens is not None:
                    wb_data["tokens"] = tokens
                if isinstance(llm_metadata, dict):
                    if isinstance(llm_metadata.get("total_turns"), (int, float)):
                        wb_data["turns"] = int(llm_metadata["total_turns"])
                    if isinstance(llm_metadata.get("cost_usd"), (int, float)):
                        wb_data["cost_usd"] = float(llm_metadata["cost_usd"])

                try:
                    wb_run.log(wb_data, step=completed_count)
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"[ASYNC] W&B log error: {type(exc).__name__}: {exc}")

    async def _result_worker() -> None:
        nonlocal persist_error
        while True:
            item = await result_queue.get()
            if item is None:
                result_queue.task_done()
                break
            try:
                if persist_error is not None:
                    continue
                await _persist_result(item)
            except Exception as exc:  # noqa: BLE001
                if persist_error is None:
                    persist_error = exc
                log.error(f"[ASYNC WRITE ERROR] {type(exc).__name__}: {exc}")
            finally:
                result_queue.task_done()

    async def _heartbeat_worker() -> None:
        while not stop_heartbeat.is_set():
            try:
                await asyncio.wait_for(stop_heartbeat.wait(), timeout=HEARTBEAT_INTERVAL_S)
            except asyncio.TimeoutError:
                _write_meta(
                    meta_path,
                    **_live_meta_updates(),
                )

    async def _on_case_start(case: EvalCase, started_perf_s: float) -> None:
        active_eval_started_at[case.case_id] = started_perf_s

    async def _on_case_finish(case: EvalCase, _finished_perf_s: float) -> None:
        active_eval_started_at.pop(case.case_id, None)

    worker_count = 0
    if remaining_cases:
        scoring_worker_limit = _tool_limit(config) if benchmark_plugin is not None else 1
        worker_count = min(
            len(remaining_cases),
            max(judge_case_limit if config.judge_enabled else 1, scoring_worker_limit),
        )
    result_workers = [asyncio.create_task(_result_worker()) for _ in range(worker_count)]
    heartbeat_task = asyncio.create_task(_heartbeat_worker())

    async def _on_case_complete(result: CaseResult) -> None:
        await result_queue.put(result)

    try:
        await runner.run_cases(
            remaining_cases,
            on_case_complete=_on_case_complete,
            on_case_start=_on_case_start,
            on_case_finish=_on_case_finish,
        )
        await result_queue.join()
        if persist_error is not None:
            raise persist_error

        _write_meta(
            meta_path,
            status="completed",
            completed=all_task_count,
            ok=ok_count,
            failed=failed_count,
            peak_in_flight_evals=runner.peak_in_flight_evals,
            peak_in_flight_tools=_peak_tools(),
            peak_in_flight_judges=peak_in_flight_judges,
            finished_at=datetime.now(timezone.utc).isoformat(),
            current_in_flight_evals=0,
            current_in_flight_tools=0,
            current_in_flight_judges=0,
            oldest_in_flight_eval_case_id=None,
            oldest_in_flight_eval_age_s=None,
            heartbeat_at=datetime.now(timezone.utc).isoformat(),
            last_progress_at=last_progress_at,
        )
        if benchmark_plugin is not None and out.exists():
            try:
                benchmark_summary = benchmark_plugin.summarize_run(load_dataset(out))
                if benchmark_summary is not None:
                    _write_meta(meta_path, benchmark_summary=benchmark_summary)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[ASYNC] benchmark summarize_run failed: {type(exc).__name__}: {exc}")
        if storage:
            _sync_to_storage(storage, out, meta_path)

        if config.hf_results_upload:
            provider = _provider_slug(config, llm_id)
            date_key = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_meta_jsonl_path = out.with_suffix(".run_meta.jsonl")
            _write_run_meta_jsonl(
                path=run_meta_jsonl_path,
                meta_path=meta_path,
                run_id=run_id,
                llm_id=llm_id,
                provider=provider,
                output_path=out,
            )
            token = _hf_results_token(config)
            if not token:
                log.warning("[ASYNC] hf_results_upload is enabled but no HF token is configured")
                _write_meta(
                    meta_path,
                    hf_results_upload={
                        "status": "skipped",
                        "reason": "missing_token",
                        "repo_id": config.hf_results_repo,
                    },
                )
            else:
                try:
                    upload_info = _upload_results_to_hf_dataset(
                        repo_id=config.hf_results_repo,
                        token=token,
                        provider=provider,
                        date_key=date_key,
                        results_path=out,
                        run_meta_jsonl_path=run_meta_jsonl_path,
                    )
                    _write_meta(
                        meta_path,
                        hf_results_upload={
                            "status": "ok",
                            **upload_info,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error(f"[ASYNC] HF result upload failed: {type(exc).__name__}: {exc}")
                    _write_meta(
                        meta_path,
                        hf_results_upload={
                            "status": "error",
                            "repo_id": config.hf_results_repo,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
    finally:
        stop_heartbeat.set()
        await heartbeat_task
        for _ in result_workers:
            await result_queue.put(None)
        if result_workers:
            await asyncio.gather(*result_workers)
        if wb_run is not None:
            try:
                wb_run.finish()
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[ASYNC] W&B finish error: {type(exc).__name__}: {exc}")
        if weave_client is not None:
            try:
                import weave  # type: ignore
                weave.finish()
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[ASYNC] Weave finish error: {type(exc).__name__}: {exc}")
        if provider_tool_setter is not None:
            provider_tool_setter(None)

    log.info(f"[ASYNC] Done: ok={ok_count} failed={failed_count} out={out}")
    return out
