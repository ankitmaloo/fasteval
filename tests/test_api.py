from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

import service.api as api_mod


def test_clear_active_only_clears_matching_run() -> None:
    original_active = api_mod._active
    try:
        api_mod._active = {
            "thread": SimpleNamespace(is_alive=lambda: True),
            "meta_path": Path("/tmp/meta.json"),
            "output": Path("/tmp/out.jsonl"),
            "token": "new-token",
        }

        api_mod._clear_active("old-token")
        assert api_mod._active is not None

        api_mod._clear_active("new-token")
        assert api_mod._active is None
    finally:
        api_mod._active = original_active


def test_start_rejects_when_live_run_exists(monkeypatch) -> None:
    original_active = api_mod._active
    original_last_meta_path = api_mod._last_meta_path
    try:
        api_mod._active = {
            "thread": SimpleNamespace(is_alive=lambda: True),
            "meta_path": Path("/tmp/existing.meta.json"),
            "output": Path("/tmp/existing.jsonl"),
            "token": "existing-token",
        }
        req = api_mod.EvalRequest(engine="async", client="fake")

        with monkeypatch.context() as m:
            m.setattr(api_mod, "_get_storage", lambda: None)
            try:
                api_mod._start(req, Path("/tmp/new.jsonl"))
            except HTTPException as exc:
                assert exc.status_code == 409
            else:
                raise AssertionError("Expected HTTPException for concurrent start")
    finally:
        api_mod._active = original_active
        api_mod._last_meta_path = original_last_meta_path


def test_get_run_prefers_scoring_for_summary(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_path = results_dir / "demo.jsonl"
    meta_path = results_dir / "demo.meta.json"

    run_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "status": "ok", "scoring": {"score": 1.0}, "metrics": {"total_s": 2.0}}),
                json.dumps({"id": "b", "status": "ok", "eval": {"score": 0.25}, "metrics": {"total_s": 4.0}}),
                json.dumps({"id": "c", "status": "error"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    meta_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    monkeypatch.setattr(api_mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(api_mod, "_get_storage", lambda: None)

    payload = api_mod.get_run("demo")

    assert payload["count"] == 3
    assert payload["ok_count"] == 2
    assert payload["failed_count"] == 1
    assert payload["avg_score"] == 0.625
    assert payload["avg_total_s"] == 3.0
