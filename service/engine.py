from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
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

from eval.log import log
from eval.storage import Storage, fire_and_forget

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


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
            elif "eval" in payload:
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
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in provider mode only
        raise RuntimeError(
            "PyYAML is required for provider resolution from config.yaml"
        ) from exc

    if not config_path.exists():
        raise ValueError(f"Provider config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    providers = payload.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError(f"Invalid providers map in {config_path}")
    expanded: dict[str, dict[str, Any]] = {}
    for name, cfg in providers.items():
        if isinstance(cfg, dict):
            expanded[str(name)] = {str(k): _expand_env(v) for k, v in cfg.items()}
    return expanded


def _load_provider_module(path: Path, cfg: dict[str, Any]) -> ModuleType:
    for key in ("base_url", "api_key", "model"):
        value = cfg.get(key)
        if value:
            os.environ[f"OAICHAT_{key.upper()}"] = str(value)

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


def _task_prompt(task: dict[str, Any]) -> str:
    if task.get("task"):
        return str(task["task"])
    if task.get("prompt"):
        return str(task["prompt"])
    return ""


def _task_id(task: dict[str, Any], idx: int) -> str:
    value = task.get("id", f"case-{idx:04d}")
    return str(value)


def _prepare_task_output_dir(task: dict[str, Any]) -> str | None:
    output_dir = task.get("output_dir")
    if not output_dir:
        return None
    from eval.core import _resolve_output_dir
    path = _resolve_output_dir(str(output_dir), task_id=str(task.get("id", "")))
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
    return str(path)


def _load_references(task: dict[str, Any]) -> dict[str, Any]:
    ref_paths = task.get("reference_files")
    if not ref_paths:
        return {"inline": {}, "paths": {}}
    from eval.core import read_reference_files

    return read_reference_files(ref_paths)


def _provider_case_context(task: dict[str, Any]) -> ProviderCaseContext:
    from eval.tools import PythonREPL

    cfg = task.get("config", {})
    case_config = dict(cfg) if isinstance(cfg, dict) else {}
    case_config["_repl"] = PythonREPL()
    case_config["_task_id"] = str(task.get("id", ""))

    output_dir = _prepare_task_output_dir(task)
    if output_dir:
        case_config["_output_dir"] = output_dir

    return ProviderCaseContext(
        task_text=_task_prompt(task),
        references=_load_references(task),
        config=case_config,
    )


def _to_case(task: dict[str, Any], idx: int, config: AsyncRunConfig) -> EvalCase:
    task_config = task.get("config", {})
    if not isinstance(task_config, dict):
        task_config = {}

    tool_mode = str(task.get("tool_mode", config.case_tool_mode))
    tool_payload = task.get("tool_payload", config.case_tool_payload)
    max_steps = int(task.get("max_steps", config.case_max_steps))

    metadata = {
        "source": task.get("source"),
        "category": task.get("category"),
        "task_config": task_config,
    }
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

    from eval.core import _collect_output_files, _resolve_output_dir

    task_out_dir = _resolve_output_dir(str(output_dir_name), task_id=str(task.get("id", "")))
    output = _collect_output_files(task_out_dir)

    judged_answer = llm_answer
    if output["inline"]:
        judged_answer = f"{judged_answer}\n\n--- OUTPUT FILES ---\n{output['inline']}"
    if output["paths"]:
        # Excel files are pre-loaded into the REPL — just note their names
        xlsx = [p for p in output["paths"] if p.lower().endswith((".xlsx", ".xls"))]
        other = [p for p in output["paths"] if p not in xlsx]
        if xlsx:
            from pathlib import Path as _P
            names = [_P(p).stem.replace(" ", "_").replace("-", "_") for p in xlsx]
            judged_answer += (
                "\n\n--- EXCEL FILES (pre-loaded in REPL) ---\n"
                + "\n".join(
                    f"- {_P(p).name}: use `{n}_data` (cached values) or `{n}_formulas` (raw formulas)"
                    for p, n in zip(xlsx, names)
                )
            )
        if other:
            judged_answer += (
                "\n\n--- DATA FILES (use REPL to read) ---\n"
                + "\n".join(f"- {path}" for path in other)
            )
    return judged_answer


def _judge_task_with_rubric(
    task: dict[str, Any],
    llm_answer: str,
    *,
    criterion_workers: int,
    rubric: dict[str, list[Any]] | None = None,
) -> dict[str, Any] | None:
    if rubric is None:
        rubric = _sanitize_rubric(task.get("rubric"))
    if rubric is None:
        return None

    from eval.core import judge_rubric, score_rubric, _preload_output_files

    judged_answer = _build_judged_answer(task, llm_answer)
    repl_seed = _preload_output_files(task.get("output_dir"), task_id=str(task.get("id", "")))
    eval_results = judge_rubric(
        str(task.get("task", "")),
        judged_answer,
        rubric,
        criterion_workers=criterion_workers,
        repl_seed=repl_seed or None,
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
            "judge_enabled": config.judge_enabled,
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
    remaining_cases: list[EvalCase],
    remaining_tasks: list[dict[str, Any]],
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
        contexts: dict[str, ProviderCaseContext] = {}
        for case, task in zip(remaining_cases, remaining_tasks, strict=True):
            contexts[case.case_id] = _provider_case_context(task)
        llm_id = str(getattr(module, "LLM_ID", f"provider-{provider_name}"))
        generate_fn = getattr(module, "generate_async", None) or module.generate
        mode = "async" if getattr(module, "generate_async", None) else "sync"
        log.info(f"[ASYNC] provider={provider_name} llm_id={llm_id} generate_mode={mode}")
        return (
            llm_id,
            ProviderLLMClient(generate_fn=generate_fn, case_contexts=contexts),
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
    if config.judge_enabled and not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is required when judge_enabled=True")

    resolved_dataset = _ensure_dataset_materialized(
        dataset_path=dataset_path,
        config=config,
    )
    tasks = load_dataset(resolved_dataset)
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
    remaining_tasks = [task for _, task in remaining]
    remaining_cases = [_to_case(task, idx, config) for idx, task in remaining]
    task_by_id = {
        case.case_id: task for case, task in zip(remaining_cases, remaining_tasks, strict=True)
    }
    llm_id, client = _build_client(
        config, remaining_cases=remaining_cases, remaining_tasks=remaining_tasks
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
        provider_tool_setter(_cpu_limit(config.cpu_sem))
        provider_tool_resetter()

    judge_case_limit = max(1, config.judge_sem) if config.judge_enabled else 1
    judge_criterion_workers = max(1, config.judge_criterion_workers)
    log.info(
        f"[ASYNC] Eval: {llm_id} ({config.client}) | {len(remaining_cases)}/{all_task_count} tasks "
        f"| eval_sem={config.eval_sem} cpu_sem={config.cpu_sem}"
        f" judge={'on' if config.judge_enabled else 'off'}"
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

    metadata_getter = getattr(client, "get_case_metadata", None)
    result_queue: asyncio.Queue[CaseResult | None] = asyncio.Queue()
    judge_sem = asyncio.Semaphore(judge_case_limit)
    persist_error: Exception | None = None

    async def _maybe_judge(task: dict[str, Any], result: CaseResult, row: dict[str, Any]) -> None:
        nonlocal in_flight_judges, peak_in_flight_judges
        if not config.judge_enabled or result.status != "ok":
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
                )
            finally:
                async with judge_lock:
                    in_flight_judges -= 1

        if eval_payload is not None:
            row["eval"] = eval_payload

    async def _persist_result(result: CaseResult) -> None:
        nonlocal completed_count, ok_count, failed_count
        nonlocal total_model_wait_s, total_tool_cpu_s, total_elapsed_s

        task = task_by_id[result.case_id]
        llm_metadata = (
            metadata_getter(result.case_id)
            if callable(metadata_getter)
            else None
        )
        row = _result_row(
            task=task,
            result=result,
            llm_id=llm_id,
            llm_metadata=llm_metadata,
        )
        try:
            await _maybe_judge(task, result, row)
        except Exception as exc:  # noqa: BLE001
            row["judge_error"] = f"{type(exc).__name__}: {exc}"

        async with write_lock:
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()

            completed_count += 1
            total_model_wait_s += result.model_wait_s
            total_tool_cpu_s += result.tool_cpu_s
            total_elapsed_s += result.total_s
            if result.status == "ok":
                ok_count += 1
            else:
                failed_count += 1

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
                eval_payload = row.get("eval")
                if isinstance(eval_payload, dict):
                    if isinstance(eval_payload.get("score"), (int, float)):
                        wb_data["score"] = float(eval_payload["score"])
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

    worker_count = 0
    if remaining_cases:
        worker_count = min(
            len(remaining_cases),
            judge_case_limit if config.judge_enabled else 1,
        )
    result_workers = [asyncio.create_task(_result_worker()) for _ in range(worker_count)]

    async def _on_case_complete(result: CaseResult) -> None:
        await result_queue.put(result)

    try:
        await runner.run_cases(remaining_cases, on_case_complete=_on_case_complete)
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
        )
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
