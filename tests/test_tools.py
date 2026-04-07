from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class _FakeSandbox:
    def __init__(self, sandbox_id: str = "sbx-123") -> None:
        self.id = sandbox_id


class _FakeParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeDaytona:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []

    async def create(self, params: Any = None, *, timeout: float = 60, on_snapshot_create_logs: Any = None) -> _FakeSandbox:
        self.create_calls.append({
            "params": params,
            "timeout": timeout,
            "on_snapshot_create_logs": on_snapshot_create_logs,
        })
        return _FakeSandbox()

    async def get(self, sandbox_id_or_name: str) -> _FakeSandbox:
        raise RuntimeError(f"missing sandbox {sandbox_id_or_name}")


def test_daytona_repl_create_without_template_uses_params(monkeypatch) -> None:
    import eval.tools as tools_mod

    fake_daytona = _FakeDaytona()
    monkeypatch.setattr(tools_mod, "_daytona_imports", lambda: (object, _FakeParams))

    repl = tools_mod.DaytonaPythonREPL(task_id="case-1")
    sandbox = asyncio.run(repl._ensure_sandbox(fake_daytona))

    assert sandbox.id == "sbx-123"
    assert repl._sandbox_id == "sbx-123"
    assert len(fake_daytona.create_calls) == 1
    call = fake_daytona.create_calls[0]
    assert isinstance(call["params"], _FakeParams)
    assert call["timeout"] == 120
    assert call["params"].auto_stop_interval == 10
    assert call["params"].snapshot is None


def test_daytona_repl_create_with_template_sets_snapshot(monkeypatch) -> None:
    import eval.tools as tools_mod

    fake_daytona = _FakeDaytona()
    monkeypatch.setattr(tools_mod, "_daytona_imports", lambda: (object, _FakeParams))

    repl = tools_mod.DaytonaPythonREPL(task_id="case-2", sandbox_template="snap-123")
    asyncio.run(repl._ensure_sandbox(fake_daytona))

    call = fake_daytona.create_calls[0]
    assert call["params"].auto_stop_interval == 10
    assert call["params"].snapshot == "snap-123"


def test_run_coroutine_sync_works_inside_running_loop() -> None:
    import eval.tools as tools_mod

    async def _inner() -> int:
        return 7

    async def _main() -> int:
        return tools_mod._run_coroutine_sync(_inner())

    assert asyncio.run(_main()) == 7


def test_daytona_bootstrap_script_uses_workspace_without_write_redirect() -> None:
    import eval.tools as tools_mod

    script = tools_mod._daytona_bootstrap_script(
        working_dir="/app",
        write_redirect_dir=None,
        write_mode=None,
        state_path=".kwbench/state/case.pkl",
        seed_globals=None,
    )

    assert "_kwb_working_dir = '/app'" in script
    assert "_kwb_write_redirect_dir = None" in script
    assert "os.chdir(_kwb_working_dir)" in script


def test_daytona_render_process_result_prefers_all_available_output_fields() -> None:
    import eval.tools as tools_mod

    class _Artifacts:
        stdout = "artifact-stdout\n"

    class _Response:
        result = ""
        stdout = "stdout\n"
        stderr = "stderr\n"
        output = None
        artifacts = _Artifacts()
        additional_properties = {"output": "extra-output\n"}

    rendered = tools_mod.DaytonaPythonREPL._render_process_result(_Response())
    assert rendered == "stdout\nstderr\nartifact-stdout\nextra-output\n"


def test_local_python_repl_upload_file_copies_bytes(tmp_path: Path) -> None:
    import eval.tools as tools_mod

    src = tmp_path / "src.txt"
    dst = tmp_path / "nested" / "dst.txt"
    src.write_text("payload", encoding="utf-8")

    repl = tools_mod.PythonREPL()
    repl.upload_file(str(src), str(dst))

    assert dst.read_text(encoding="utf-8") == "payload"
