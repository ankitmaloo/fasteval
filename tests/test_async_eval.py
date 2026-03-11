from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from service import engine as async_eval_mod
from service.engine import AsyncRunConfig, run_async_eval


def _write_dataset(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(count):
            row = {"id": f"kw_{idx:03d}", "task": f"Task {idx}"}
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_async_eval_fake_client_writes_results_and_meta(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "results.jsonl"
    meta = output.with_suffix(".meta.json")
    _write_dataset(dataset, 30)

    config = AsyncRunConfig(
        client="fake",
        eval_sem=16,
        cpu_sem=4,
        fake_base_latency_s=0.001,
        fake_jitter_s=0.003,
        fake_tool_ratio=1.0,
        case_tool_mode="sleep",
        case_tool_payload=0.01,
        case_max_steps=2,
    )
    result_path = run_async_eval(config, dataset_path=dataset, output_path=output)

    assert result_path == output
    rows = _read_jsonl(output)
    assert len(rows) == 30
    assert all("metrics" in row for row in rows)
    assert all(row["runner"] == "async" for row in rows)

    meta_payload = json.loads(meta.read_text(encoding="utf-8"))
    assert meta_payload["status"] == "completed"
    assert meta_payload["total"] == 30
    assert meta_payload["completed"] == 30
    assert meta_payload["peak_in_flight_tools"] <= 4
    assert meta_payload["peak_in_flight_evals"] <= 16
    assert meta_payload["ok"] + meta_payload["failed"] == 30


def test_async_eval_resume_skips_already_completed_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, 12)

    with output.open("w", encoding="utf-8") as handle:
        for idx in range(3):
            handle.write(
                json.dumps(
                    {
                        "id": f"kw_{idx:03d}",
                        "case_id": f"kw_{idx:03d}",
                        "status": "ok",
                        "runner": "async",
                    }
                )
                + "\n"
            )

    config = AsyncRunConfig(
        client="fake",
        eval_sem=8,
        cpu_sem=2,
        fake_base_latency_s=0.001,
        fake_jitter_s=0.002,
        fake_tool_ratio=0.0,
        case_max_steps=1,
    )
    run_async_eval(config, dataset_path=dataset, output_path=output)

    rows = _read_jsonl(output)
    ids = [row["id"] for row in rows]
    assert len(rows) == 12
    assert len(set(ids)) == 12


def test_async_eval_replay_client(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    fixtures = tmp_path / "fixtures.json"
    output = tmp_path / "results.jsonl"

    _write_dataset(dataset, 2)
    fixtures.write_text(
        json.dumps(
            {
                "kw_000": {"0": {"text": "alpha", "tool_needed": False}},
                "kw_001": {"0": {"text": "beta", "tool_needed": False}},
            }
        ),
        encoding="utf-8",
    )

    config = AsyncRunConfig(
        client="replay",
        replay_fixtures=str(fixtures),
        eval_sem=4,
        cpu_sem=2,
        case_max_steps=1,
    )
    run_async_eval(config, dataset_path=dataset, output_path=output)

    rows = _read_jsonl(output)
    assert {row["id"] for row in rows} == {"kw_000", "kw_001"}
    assert {row["llm_answer"] for row in rows} == {"alpha", "beta"}
    assert all(row["status"] == "ok" for row in rows)


def test_async_eval_fetches_dataset_from_hf_snapshot_when_missing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "reference_files").mkdir(parents=True, exist_ok=True)
    (snapshot / "reference_files" / "doc.txt").write_text("hello", encoding="utf-8")
    (snapshot / "dataset.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "kw_hf_001", "task": "HF task 1"}),
                json.dumps({"id": "kw_hf_002", "task": "HF task 2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    base_dir = tmp_path / "repo_root"
    base_dir.mkdir(parents=True, exist_ok=True)
    dataset_target = base_dir / "dataset.jsonl"
    output = tmp_path / "out.jsonl"

    monkeypatch.setattr(async_eval_mod, "BASE_DIR", base_dir)
    monkeypatch.setattr(async_eval_mod, "DATASET_PATH", dataset_target)
    monkeypatch.setattr(async_eval_mod, "_download_hf_snapshot", lambda repo_id, local_dir: snapshot)

    config = AsyncRunConfig(
        client="fake",
        eval_sem=4,
        cpu_sem=2,
        case_max_steps=1,
        fake_base_latency_s=0.001,
        fake_jitter_s=0.001,
        fake_tool_ratio=0.0,
        hf_repo="clio-ai/kwbench",
        hf_fetch_if_missing=True,
        hf_force_refresh=False,
    )
    run_async_eval(config, dataset_path=None, output_path=output)

    assert dataset_target.exists()
    assert (base_dir / "reference_files" / "doc.txt").exists()
    rows = _read_jsonl(output)
    assert len(rows) == 2


def test_async_eval_preserves_original_task_row_and_appends_runner_fields(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    fixtures = tmp_path / "fixtures.json"
    output = tmp_path / "results.jsonl"

    dataset.write_text(
        json.dumps(
            {
                "id": "kw_full_001",
                "task": "Do thing",
                "source": "kwbench",
                "category": "ops",
                "custom_field": {"x": 1, "y": "z"},
                "rubric": {"mandatory": ["must pass"], "good_to_have": [], "ideal": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fixtures.write_text(
        json.dumps({"kw_full_001": {"0": {"text": "answer", "tool_needed": False}}}),
        encoding="utf-8",
    )

    config = AsyncRunConfig(
        client="replay",
        replay_fixtures=str(fixtures),
        eval_sem=2,
        cpu_sem=1,
        case_max_steps=1,
    )
    run_async_eval(config, dataset_path=dataset, output_path=output)

    rows = _read_jsonl(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "kwbench"
    assert row["category"] == "ops"
    assert row["custom_field"] == {"x": 1, "y": "z"}
    assert row["llm_answer"] == "answer"
    assert row["runner"] == "async"
    assert "metrics" in row


def test_async_eval_judge_pipeline_runs_in_parallel_and_is_capped(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, 12)

    # Add a minimal rubric to each case so async judging is exercised.
    rows = _read_jsonl(dataset)
    with dataset.open("w", encoding="utf-8") as handle:
        for row in rows:
            row["rubric"] = {"mandatory": ["criterion"], "good_to_have": [], "ideal": []}
            handle.write(json.dumps(row) + "\n")

    state = {"in_flight": 0, "peak": 0}
    lock = threading.Lock()

    def fake_judge(
        task: dict[str, Any],
        llm_answer: str,
        *,
        criterion_workers: int,
        rubric: dict[str, list[Any]] | None = None,
    ) -> dict[str, Any]:
        assert criterion_workers == 2
        assert llm_answer
        assert rubric is not None
        with lock:
            state["in_flight"] += 1
            if state["in_flight"] > state["peak"]:
                state["peak"] = state["in_flight"]
        time.sleep(0.02)
        with lock:
            state["in_flight"] -= 1
        return {
            "mandatory": [True],
            "good_to_have": [],
            "ideal": [],
            "score": 1.0,
        }

    monkeypatch.setattr(async_eval_mod, "_judge_task_with_rubric", fake_judge)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    config = AsyncRunConfig(
        client="fake",
        eval_sem=8,
        cpu_sem=4,
        fake_base_latency_s=0.001,
        fake_jitter_s=0.001,
        fake_tool_ratio=0.0,
        case_max_steps=1,
        judge_enabled=True,
        judge_sem=3,
        judge_criterion_workers=2,
    )
    run_async_eval(config, dataset_path=dataset, output_path=output)

    out_rows = _read_jsonl(output)
    assert len(out_rows) == 12
    assert all("eval" in row for row in out_rows)
    assert state["peak"] <= 3
    assert state["peak"] > 1


def test_async_eval_judge_skips_cases_without_rubric(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "results.jsonl"
    rows = [
        {"id": "kw_000", "task": "no rubric"},
        {"id": "kw_001", "task": "bad rubric", "rubric": {"mandatory": None, "good_to_have": None, "ideal": None}},
        {"id": "kw_002", "task": "has rubric", "rubric": {"mandatory": ["criterion"]}},
    ]
    with dataset.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    calls = {"count": 0}

    def fake_judge(
        task: dict[str, Any],
        llm_answer: str,
        *,
        criterion_workers: int,
        rubric: dict[str, list[Any]] | None = None,
    ) -> dict[str, Any]:
        calls["count"] += 1
        assert task["id"] == "kw_002"
        assert rubric is not None
        return {"mandatory": [True], "good_to_have": [], "ideal": [], "score": 1.0}

    monkeypatch.setattr(async_eval_mod, "_judge_task_with_rubric", fake_judge)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    config = AsyncRunConfig(
        client="fake",
        eval_sem=4,
        cpu_sem=2,
        fake_base_latency_s=0.001,
        fake_jitter_s=0.001,
        fake_tool_ratio=0.0,
        case_max_steps=1,
        judge_enabled=True,
        judge_sem=2,
        judge_criterion_workers=2,
    )
    run_async_eval(config, dataset_path=dataset, output_path=output)

    out_rows = _read_jsonl(output)
    assert len(out_rows) == 3
    assert calls["count"] == 1
    by_id = {row["id"]: row for row in out_rows}
    assert "eval" not in by_id["kw_000"]
    assert "eval" not in by_id["kw_001"]
    assert by_id["kw_002"]["eval"]["score"] == 1.0


def test_async_eval_judge_requires_gemini_key(tmp_path: Path, monkeypatch: Any) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, 1)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    config = AsyncRunConfig(
        client="fake",
        eval_sem=2,
        cpu_sem=1,
        fake_base_latency_s=0.001,
        fake_jitter_s=0.001,
        fake_tool_ratio=0.0,
        case_max_steps=1,
        judge_enabled=True,
        judge_sem=1,
        judge_criterion_workers=1,
    )
    try:
        run_async_eval(config, dataset_path=dataset, output_path=output)
    except ValueError as exc:
        assert "GEMINI_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected ValueError when judge_enabled=True without GEMINI_API_KEY")


def test_async_eval_uploads_results_to_hf_dataset(tmp_path: Path, monkeypatch: Any) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "results.jsonl"
    meta_path = output.with_suffix(".meta.json")
    _write_dataset(dataset, 3)

    captured: dict[str, Any] = {}

    def fake_upload(
        *,
        repo_id: str,
        token: str,
        provider: str,
        date_key: str,
        results_path: Path,
        run_meta_jsonl_path: Path,
    ) -> dict[str, Any]:
        captured["repo_id"] = repo_id
        captured["token"] = token
        captured["provider"] = provider
        captured["date_key"] = date_key
        captured["results_path"] = results_path
        captured["run_meta_jsonl_path"] = run_meta_jsonl_path
        assert results_path.exists()
        assert run_meta_jsonl_path.exists()
        run_meta_rows = _read_jsonl(run_meta_jsonl_path)
        assert len(run_meta_rows) == 1
        assert run_meta_rows[0]["provider"] == provider
        return {
            "repo_id": repo_id,
            "folder": f"{provider}/{date_key}",
            "files": {
                "results_jsonl": f"{provider}/{date_key}/results.jsonl",
                "run_meta_jsonl": f"{provider}/{date_key}/run_meta.jsonl",
            },
        }

    monkeypatch.setattr(async_eval_mod, "_upload_results_to_hf_dataset", fake_upload)

    config = AsyncRunConfig(
        client="fake",
        eval_sem=4,
        cpu_sem=2,
        fake_base_latency_s=0.001,
        fake_jitter_s=0.001,
        fake_tool_ratio=0.0,
        case_max_steps=1,
        hf_results_upload=True,
        hf_results_repo="clio-ai/kwbresults",
        hf_results_token="hf-test-token",
    )
    run_async_eval(config, dataset_path=dataset, output_path=output)

    assert captured["repo_id"] == "clio-ai/kwbresults"
    assert captured["token"] == "hf-test-token"
    assert captured["provider"] == "async-fake"
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta_payload["hf_results_upload"]["status"] == "ok"
