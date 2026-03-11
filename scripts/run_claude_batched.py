#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _http_json_with_backoff(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    retries: int,
    base_sleep_s: float,
    timeout_s: int,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if payload is None:
                req = Request(url=url, method=method)
            else:
                data = json.dumps(payload).encode("utf-8")
                req = Request(url=url, data=data, method=method)
                req.add_header("Content-Type", "application/json")
            with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError) as exc:  # noqa: PERF203
            last_exc = exc
            if attempt >= retries:
                break
            sleep_s = base_sleep_s * (2**attempt)
            print(
                f"[http-retry] {method} {url} failed "
                f"(attempt {attempt + 1}/{retries + 1}): {exc}; sleep {sleep_s:.2f}s"
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _http_get_json(url: str, *, retries: int, base_sleep_s: float) -> dict[str, Any]:
    return _http_json_with_backoff(
        method="GET",
        url=url,
        payload=None,
        retries=retries,
        base_sleep_s=base_sleep_s,
        timeout_s=30,
    )


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    retries: int,
    base_sleep_s: float,
) -> dict[str, Any]:
    return _http_json_with_backoff(
        method="POST",
        url=url,
        payload=payload,
        retries=retries,
        base_sleep_s=base_sleep_s,
        timeout_s=120,
    )


def _load_task_ids(dataset_path: Path) -> list[str]:
    ids: list[str] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(str(obj["id"]))
    return ids


def _summarize_output_delta(
    output_path: Path,
    seen_ids: set[str],
) -> tuple[int, float, int, int]:
    rows = 0
    cost = 0.0
    in_tokens = 0
    out_tokens = 0
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows += 1
            obj = json.loads(line)
            case_id = str(obj.get("id") or obj.get("case_id") or "")
            if not case_id or case_id in seen_ids:
                continue
            seen_ids.add(case_id)
            rows += 1
            md = obj.get("llm_metadata") or {}
            usage = md.get("usage") if isinstance(md, dict) else {}
            if isinstance(md.get("cost_usd"), (int, float)):
                cost += float(md["cost_usd"])
            if isinstance(usage, dict):
                in_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                out_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return rows, cost, in_tokens, out_tokens


def _parse_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _truthy_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Claude eval in gated batches.")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--provider-config-path", default="eval/config.yaml")
    parser.add_argument("--llm", default="claude")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=1, help="1-based")
    parser.add_argument(
        "--output",
        default=None,
        help="Final cumulative output JSONL path. If omitted, auto-generates in service/results/.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume into an existing output file (uses /eval/resume for first batch).",
    )
    parser.add_argument("--eval-sem", type=int, default=64)
    parser.add_argument("--cpu-sem", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3, help="Model-call retries per case.")
    parser.add_argument("--retry-base-s", type=float, default=0.05, help="Base retry sleep; engine uses exponential backoff.")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--http-retries", type=int, default=3, help="HTTP retries for API calls.")
    parser.add_argument("--http-backoff-base-s", type=float, default=1.0, help="HTTP exponential backoff base.")
    parser.add_argument(
        "--judge-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rubric judging for each case (default: enabled).",
    )
    parser.add_argument("--judge-sem", type=int, default=4)
    parser.add_argument("--judge-criterion-workers", type=int, default=4)
    parser.add_argument(
        "--hf-results-upload",
        action=argparse.BooleanOptionalAction,
        default=_truthy_env("HF_RESULTS_UPLOAD", True),
    )
    parser.add_argument("--hf-results-repo", default=os.environ.get("HF_RESULTS_REPO", "clio-ai/kwbresults"))
    parser.add_argument("--hf-repo", default=os.environ.get("HF_DATASET_REPO", "clio-ai/kwbench"))
    parser.add_argument(
        "--hf-fetch-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For final benchmark, keep false so the local dataset is authoritative.",
    )
    parser.add_argument(
        "--hf-force-refresh",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)
    _parse_env_file(root_dir / ".env")

    wandb_project = os.environ.get("WANDB_PROJECT", "")
    hf_results_token = (os.environ.get("HF_RESULTS_TOKEN") or "").strip()
    if not hf_results_token:
        hf_results_token = (os.environ.get("HF_TOKEN") or "").strip()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"service/results/final-claude-benchmark_{ts}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        health = _http_get_json(
            f"{args.api_url}/health",
            retries=args.http_retries,
            base_sleep_s=args.http_backoff_base_s,
        )
    except (URLError, HTTPError) as exc:
        print(f"Server not reachable at {args.api_url}: {exc}")
        return 1
    if health.get("status") != "ok":
        print(f"Server health not ok: {health}")
        return 1

    ids = _load_task_ids(dataset_path)
    total = len(ids)
    if args.start_index < 1 or args.start_index > total:
        print(f"start-index must be in [1, {total}]")
        return 1
    if args.batch_size < 1:
        print("batch-size must be >= 1")
        return 1

    output_exists = output_path.exists()
    if output_exists and args.start_index == 1 and not args.resume_existing:
        print(f"Output already exists: {output_path}")
        print("Refusing to overwrite implicitly. Re-run with --resume-existing or choose --output <new_path>.")
        return 1

    idx = args.start_index - 1
    batch_no = 1
    grand_rows = 0
    grand_cost = 0.0
    grand_in_tokens = 0
    grand_out_tokens = 0
    seen_case_ids: set[str] = set()
    if output_exists:
        _, _, _, _ = _summarize_output_delta(output_path, seen_case_ids)

    print(f"Total tasks: {total}")
    print(f"Starting at index: {args.start_index}")
    print(f"Batch size: {args.batch_size}")
    print(f"Provider: {args.llm}")
    print(f"Output file: {output_path}")
    print(f"Dataset: {dataset_path}")
    print(
        f"Judge enabled: {args.judge_enabled} "
        f"(judge_sem={args.judge_sem}, judge_criterion_workers={args.judge_criterion_workers})"
    )
    print(f"Model retries: max_retries={args.max_retries}, retry_base_s={args.retry_base_s} (exponential backoff)")
    print(f"HF upload: {args.hf_results_upload} repo={args.hf_results_repo} token_source={'HF_RESULTS_TOKEN' if (os.environ.get('HF_RESULTS_TOKEN') or '').strip() else ('HF_TOKEN' if hf_results_token else 'none')}")
    print()

    use_resume = output_exists or args.resume_existing or args.start_index > 1

    while idx < total:
        end = min(idx + args.batch_size, total)
        batch_ids = ids[idx:end]
        print(f"=== Batch {batch_no} | tasks {idx + 1}-{end} ({len(batch_ids)} items) ===")

        payload = {
            "engine": "async",
            "client": "provider",
            "llm": args.llm,
            "dataset": str(dataset_path),
            "output": str(output_path),
            "provider_config_path": args.provider_config_path,
            "eval_sem": args.eval_sem,
            "cpu_sem": args.cpu_sem,
            "max_retries": args.max_retries,
            "retry_base_s": args.retry_base_s,
            "judge_enabled": args.judge_enabled,
            "judge_sem": args.judge_sem,
            "judge_criterion_workers": args.judge_criterion_workers,
            "wandb_project": wandb_project,
            "hf_results_upload": args.hf_results_upload,
            "hf_results_repo": args.hf_results_repo,
            "hf_results_token": hf_results_token,
            "hf_repo": args.hf_repo,
            "hf_fetch_if_missing": args.hf_fetch_if_missing,
            "hf_force_refresh": args.hf_force_refresh,
            "task_ids": batch_ids,
        }

        endpoint = "/eval/resume" if use_resume else "/eval/start"
        try:
            start_resp = _http_post_json(
                f"{args.api_url}{endpoint}",
                payload,
                retries=args.http_retries,
                base_sleep_s=args.http_backoff_base_s,
            )
        except (URLError, HTTPError) as exc:
            print(f"Failed to start batch: {exc}")
            return 1
        print(f"endpoint={endpoint}")
        print(json.dumps(start_resp, indent=2))
        use_resume = True

        while True:
            try:
                status = _http_get_json(
                    f"{args.api_url}/eval/status",
                    retries=args.http_retries,
                    base_sleep_s=args.http_backoff_base_s,
                )
            except (URLError, HTTPError) as exc:
                print(f"Status check failed: {exc}")
                return 1

            state = status.get("status")
            done = int(status.get("completed", 0))
            batch_total = int(status.get("total", 0))
            print(f"status={state} completed={done}/{batch_total}")

            if state in {"completed", "done"}:
                break
            if state == "error":
                print("Batch failed:")
                print(json.dumps(status, indent=2))
                return 1
            time.sleep(args.poll_seconds)

        output_path = Path(str(status.get("output", "")))
        if not output_path.exists():
            print(f"Output file not found: {output_path}")
            return 1

        rows, cost, in_toks, out_toks = _summarize_output_delta(output_path, seen_case_ids)
        grand_rows += rows
        grand_cost += cost
        grand_in_tokens += in_toks
        grand_out_tokens += out_toks

        print(f"Batch output: {output_path}")
        print(f"Batch rows: {rows}")
        print(f"Batch cost_usd: {cost:.6f}")
        print(f"Batch tokens in/out: {in_toks} / {out_toks}")
        print(
            "Running totals -> "
            f"rows: {grand_rows}, cost_usd: {grand_cost:.6f}, "
            f"tokens in/out: {grand_in_tokens} / {grand_out_tokens}"
        )
        print()

        idx = end
        if idx >= total:
            break

        if args.non_interactive:
            batch_no += 1
            continue

        answer = input("Continue to next batch? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print(f"Stopped by user after batch {batch_no}.")
            print(f"Resume with --start-index {idx + 1} --output {output_path} --resume-existing")
            return 0

        batch_no += 1

    print("All requested batches completed.")
    print(
        "Final totals -> "
        f"rows: {grand_rows}, cost_usd: {grand_cost:.6f}, "
        f"tokens in/out: {grand_in_tokens} / {grand_out_tokens}"
    )
    print(f"Final output artifact: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
