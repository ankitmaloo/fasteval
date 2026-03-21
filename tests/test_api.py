from __future__ import annotations

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
