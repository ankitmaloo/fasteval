"""Shared tool definitions and execution for agentic LLM modules."""

import atexit
import asyncio
import io
import json
import multiprocessing
import os
import queue
import shlex
import shutil
import subprocess
import threading
import time
import traceback
import weakref
from contextlib import asynccontextmanager, contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

MAX_TURNS = 50
MAX_OUTPUT = 50_000  # truncate output beyond this

SYSTEM_PROMPT = (
    "Follow the task's filesystem instructions exactly.\n"
    "If the prompt names a workspace or output path, use that path and do not invent alternate locations.\n"
    "Available Python libraries: pandas, numpy, openpyxl, python-docx, requests."
)

_CWD = str(Path(__file__).resolve().parent)

_tool_sem: threading.BoundedSemaphore | None = None
_metrics_lock = threading.Lock()
_in_flight_tools = 0
_peak_in_flight_tools = 0
_tool_trace_lock = threading.Lock()
_repl_instances: "weakref.WeakSet[object]" = weakref.WeakSet()


@runtime_checkable
class ToolSession(Protocol):
    """Per-case stateful tool execution session."""

    def run(self, code: str, timeout: int = 120) -> str: ...
    def run_bash(self, command: str, timeout: int = 120, cwd: str | None = None) -> str: ...
    def sync_outputs(self, local_dir: str) -> None: ...
    def upload_file(self, local_path: str, remote_path: str) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class ToolRuntime(Protocol):
    """Factory for case-local tool sessions."""

    name: str

    def create_session(
        self,
        *,
        task_id: str,
        seed_globals: dict[str, Any] | None = None,
        output_dir: str | None = None,
        working_dir: str | None = None,
        write_redirect_dir: str | None = None,
        write_mode: str | None = None,
        remote_output_dir: str | None = None,
        remote_working_dir: str | None = None,
        remote_write_redirect_dir: str | None = None,
        remote_write_mode: str | None = None,
        remote_sync_dir: str | None = None,
        remote_reference_files: dict[str, str] | None = None,
        sandbox_template: str | None = None,
        sync_local_output_on_bootstrap: bool = False,
    ) -> ToolSession: ...


def max_turns_for_config(config: dict[str, Any] | None) -> int:
    if not isinstance(config, dict):
        return MAX_TURNS
    for key in ("_max_turns", "max_turns", "case_max_steps"):
        value = config.get(key)
        try:
            turns = int(value)
        except (TypeError, ValueError):
            continue
        if turns >= 1:
            return turns
    return MAX_TURNS

BASH_SCHEMA = {
    "name": "bash",
    "description": "Execute a bash command. Use for shell operations, installing packages, git, curl, etc.",
    "parameters": {
        "type": "object",
        "title": "BashParams",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute", "title": "Command"},
        },
        "required": ["command"],
    },
}

CODE_SCHEMA = {
    "name": "execute_code",
        "description": (
            "Execute Python code in a stateful REPL. Variables, imports, and state persist across calls. "
            "Available libraries: pandas, numpy, openpyxl, python-docx, requests. "
            "Data file paths are provided in the prompt — read them with pd.read_csv(), pd.read_excel(), etc. "
            "If the prompt specifies a workspace or output path, follow those path instructions exactly. "
            "Print results to stdout."
        ),
    "parameters": {
        "type": "object",
        "title": "ExecuteCodeParams",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute", "title": "Code"},
        },
        "required": ["code"],
    },
}


def _make_sandboxed_open(output_dir: str, original_open):
    """Create a sandboxed open() that redirects file writes into output_dir."""
    return _make_sandboxed_open_with_mode(output_dir, original_open, reject_outside_writes=False)


def _make_sandboxed_open_with_mode(output_dir: str, original_open, *, reject_outside_writes: bool):
    """Create a sandboxed open() constrained to output_dir."""
    _output_dir = os.path.realpath(output_dir)

    def _sandboxed_open(file, mode="r", *args, **kwargs):
        is_write = any(c in mode for c in "wxa")
        if is_write and isinstance(file, (str, Path)):
            resolved = os.path.realpath(os.path.join(_output_dir, file)) if not os.path.isabs(str(file)) else os.path.realpath(str(file))
            if not resolved.startswith(_output_dir + os.sep) and resolved != _output_dir:
                if reject_outside_writes:
                    raise PermissionError(f"Write outside allowed root: {resolved}")
                basename = os.path.basename(resolved)
                resolved = os.path.join(_output_dir, basename)
            # Ensure parent dir exists
            parent = os.path.dirname(resolved)
            if parent:
                os.makedirs(parent, exist_ok=True)
            file = resolved
        return original_open(file, mode, *args, **kwargs)

    return _sandboxed_open


@contextmanager
def _sandbox_builtins(output_dir: str, *, reject_outside_writes: bool = False):
    """Temporarily patch builtins.open AND io.open so ALL file writes
    (including from libraries like pandas, openpyxl, zipfile) are redirected
    into output_dir."""
    import builtins
    original_builtin_open = builtins.open
    original_io_open = io.open
    sandboxed = _make_sandboxed_open_with_mode(
        output_dir,
        original_builtin_open,
        reject_outside_writes=reject_outside_writes,
    )
    builtins.open = sandboxed
    io.open = sandboxed
    try:
        yield sandboxed
    finally:
        builtins.open = original_builtin_open
        io.open = original_io_open


def _repl_mp_context(seed_globals: dict | None) -> multiprocessing.context.BaseContext:
    if seed_globals and os.name == "posix":
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _repl_worker(
    request_queue: Any,
    response_queue: Any,
    seed_globals: dict | None,
    working_dir: str | None,
    write_redirect_dir: str | None,
    write_mode: str | None,
) -> None:
    globals_dict = {"__builtins__": __builtins__}
    if seed_globals:
        globals_dict.update(seed_globals)
    if working_dir:
        try:
            os.makedirs(working_dir, exist_ok=True)
        except OSError:
            pass
    if write_redirect_dir:
        os.makedirs(write_redirect_dir, exist_ok=True)

    while True:
        code = request_queue.get()
        if code is None:
            return

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        run_cwd = working_dir or _CWD
        prev_cwd = os.getcwd()
        try:
            os.chdir(run_cwd)
            if write_redirect_dir:
                with _sandbox_builtins(
                    write_redirect_dir,
                    reject_outside_writes=(write_mode == "strict"),
                ) as sandboxed_open:
                    globals_dict["open"] = sandboxed_open
                    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                        exec(compile(code, "<repl>", "exec"), globals_dict)
            else:
                globals_dict.pop("open", None)
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(compile(code, "<repl>", "exec"), globals_dict)
        except BaseException:  # noqa: BLE001
            stderr_buf.write(traceback.format_exc())
        finally:
            os.chdir(prev_cwd)

        out = stdout_buf.getvalue() + stderr_buf.getvalue()
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n[TRUNCATED at {MAX_OUTPUT} chars]"
        response_queue.put(out)


def _shutdown_repls() -> None:
    for repl in list(_repl_instances):
        try:
            repl.close()
        except Exception:
            pass


def _run_coroutine_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, even if a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            error["exc"] = exc

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join()
    if "exc" in error:
        raise error["exc"]
    return result.get("value")


atexit.register(_shutdown_repls)


class PythonREPL:
    """Stateful Python REPL. Variables persist across exec() calls."""

    def __init__(
        self,
        seed_globals: dict | None = None,
        output_dir: str | None = None,
        *,
        working_dir: str | None = None,
        write_redirect_dir: str | None = None,
        write_mode: str | None = None,
    ):
        self._seed_globals = dict(seed_globals) if seed_globals else None
        self._output_dir = output_dir
        use_explicit_paths = working_dir is not None or write_redirect_dir is not None
        self._working_dir = working_dir if use_explicit_paths else output_dir
        self._write_redirect_dir = write_redirect_dir if use_explicit_paths else output_dir
        self._write_mode = write_mode if use_explicit_paths else ("redirect" if output_dir else None)
        if self._working_dir:
            try:
                os.makedirs(self._working_dir, exist_ok=True)
            except OSError:
                pass
        if self._write_redirect_dir:
            os.makedirs(self._write_redirect_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._proc: multiprocessing.Process | None = None
        self._request_queue: Any = None
        self._response_queue: Any = None
        _repl_instances.add(self)

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._stop_worker()
        ctx = _repl_mp_context(self._seed_globals)
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()
        self._proc = ctx.Process(
            target=_repl_worker,
            args=(
                self._request_queue,
                self._response_queue,
                self._seed_globals,
                self._working_dir,
                self._write_redirect_dir,
                self._write_mode,
            ),
            daemon=True,
        )
        self._proc.start()

    def _stop_worker(self) -> None:
        proc = self._proc
        request_queue = self._request_queue
        response_queue = self._response_queue
        self._proc = None
        self._request_queue = None
        self._response_queue = None

        if request_queue is not None:
            try:
                request_queue.put_nowait(None)
            except Exception:
                pass

        if proc is not None:
            proc.join(timeout=0.1)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)

        for q in (request_queue, response_queue):
            if q is None:
                continue
            try:
                q.close()
            except Exception:
                pass
            try:
                q.join_thread()
            except Exception:
                pass

    def run(self, code: str, timeout: int = 120) -> str:
        with self._lock:
            self._ensure_worker()
            request_queue = self._request_queue
            response_queue = self._response_queue
            if request_queue is None or response_queue is None:
                return "[ERROR: failed to start Python REPL worker]"

            try:
                request_queue.put(code)
                return response_queue.get(timeout=timeout)
            except queue.Empty:
                self._stop_worker()
                return f"[TIMEOUT after {timeout}s]"
            except Exception as exc:  # noqa: BLE001
                self._stop_worker()
                return f"[ERROR: {exc}]"

    def reset(self):
        with self._lock:
            self._stop_worker()

    def close(self) -> None:
        with self._lock:
            self._stop_worker()

    def sync_outputs(self, local_dir: str) -> None:
        del local_dir

    def upload_file(self, local_path: str, remote_path: str) -> None:
        destination = Path(remote_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, destination)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _quote_py(value: str) -> str:
    return repr(value)


def _serialize_seed_globals(seed_globals: dict[str, Any] | None) -> str:
    if not seed_globals:
        return "{}"
    serializable: dict[str, Any] = {}
    for key, value in seed_globals.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        serializable[str(key)] = value
    return json.dumps(serializable)


def _daytona_bootstrap_script(
    *,
    working_dir: str,
    write_redirect_dir: str | None,
    write_mode: str | None,
    state_path: str,
    seed_globals: dict[str, Any] | None,
) -> str:
    seed_json = _serialize_seed_globals(seed_globals)
    redirect_json = "None" if write_redirect_dir is None else _quote_py(write_redirect_dir)
    strict_json = "True" if write_mode == "strict" else "False"
    return f"""
import importlib
import io
import json
import os
import types

try:
    import cloudpickle as _kwb_pickle
except Exception:
    import pickle as _kwb_pickle

_kwb_working_dir = {_quote_py(working_dir)}
_kwb_write_redirect_dir = {redirect_json}
_kwb_reject_outside_writes = {strict_json}
_kwb_state_path = {_quote_py(state_path)}
_kwb_seed_globals = json.loads({_quote_py(seed_json)})


def _kwb_make_sandboxed_open(output_dir, original_open):
    _output_dir = os.path.realpath(output_dir)

    def _kwb_sandboxed_open(file, mode="r", *args, **kwargs):
        file_path = None
        if isinstance(file, (str, os.PathLike)):
            file_path = os.fspath(file)
        if file_path is not None:
            is_write = any(c in mode for c in "wxa+")
            resolved = (
                os.path.realpath(os.path.join(_output_dir, file_path))
                if not os.path.isabs(file_path)
                else os.path.realpath(file_path)
            )
            if is_write and not resolved.startswith(_output_dir + os.sep) and resolved != _output_dir:
                if _kwb_reject_outside_writes:
                    raise PermissionError(f"Write outside allowed root: {{resolved}}")
                resolved = os.path.join(_output_dir, os.path.basename(resolved))
            file = resolved
            parent = os.path.dirname(resolved)
            if parent:
                os.makedirs(parent, exist_ok=True)
        return original_open(file, mode, *args, **kwargs)

    return _kwb_sandboxed_open


if _kwb_write_redirect_dir and not globals().get("_kwb_open_patched"):
    import builtins

    _kwb_builtin_open = builtins.open
    _kwb_io_open = io.open
    _kwb_sandboxed_open = _kwb_make_sandboxed_open(_kwb_write_redirect_dir, _kwb_builtin_open)
    builtins.open = _kwb_sandboxed_open
    io.open = _kwb_sandboxed_open
    globals()["open"] = _kwb_sandboxed_open
    globals()["_kwb_open_patched"] = True

os.makedirs(_kwb_working_dir, exist_ok=True)
os.chdir(_kwb_working_dir)

if os.path.exists(_kwb_state_path):
    with open(_kwb_state_path, "rb") as _kwb_state_handle:
        _kwb_payload = _kwb_pickle.load(_kwb_state_handle)
    for _kwb_name, _kwb_module in _kwb_payload.get("modules", {{}}).items():
        try:
            globals()[_kwb_name] = importlib.import_module(_kwb_module)
        except Exception:
            pass
    globals().update(_kwb_payload.get("globals", {{}}))
elif _kwb_seed_globals:
    globals().update(_kwb_seed_globals)
"""


_DAYTONA_CHECKPOINT_SCRIPT = """
import os
import types

try:
    import cloudpickle as _kwb_pickle
except Exception:
    import pickle as _kwb_pickle

_kwb_payload = {"globals": {}, "modules": {}}
for _kwb_name, _kwb_value in list(globals().items()):
    if _kwb_name.startswith("_kwb_") or _kwb_name in {"__builtins__", "open"}:
        continue
    if isinstance(_kwb_value, types.ModuleType):
        _kwb_payload["modules"][_kwb_name] = _kwb_value.__name__
        continue
    try:
        _kwb_pickle.dumps(_kwb_value)
    except Exception:
        continue
    _kwb_payload["globals"][_kwb_name] = _kwb_value

os.makedirs(os.path.dirname(_kwb_state_path), exist_ok=True)
with open(_kwb_state_path, "wb") as _kwb_state_handle:
    _kwb_pickle.dump(_kwb_payload, _kwb_state_handle)
"""


def _daytona_imports():
    try:
        from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams
    except ImportError as exc:  # pragma: no cover - exercised only when remote mode requested
        raise RuntimeError(
            "Daytona support requires the 'daytona' package. Install it with `pip install daytona`."
        ) from exc
    return AsyncDaytona, CreateSandboxFromSnapshotParams


class DaytonaPythonREPL:
    """Remote Python executor backed by Daytona sandboxes.

    The sandbox is created lazily, started only while code or bash is executing,
    and stopped between turns. Python REPL state is checkpointed to the sandbox
    filesystem so the main eval loop only pays for active execution time.
    """

    def __init__(
        self,
        *,
        task_id: str,
        seed_globals: dict[str, Any] | None = None,
        output_dir: str | None = None,
        working_dir: str | None = None,
        write_redirect_dir: str | None = None,
        write_mode: str | None = None,
        remote_output_dir: str | None = None,
        remote_working_dir: str | None = None,
        remote_write_redirect_dir: str | None = None,
        remote_write_mode: str | None = None,
        remote_sync_dir: str | None = None,
        reference_files: dict[str, str] | None = None,
        remote_reference_files: dict[str, str] | None = None,
        sandbox_template: str | None = None,
        sync_local_output_on_bootstrap: bool = False,
    ) -> None:
        self._task_id = task_id or "case"
        self._seed_globals = dict(seed_globals) if seed_globals else None
        self._output_dir = output_dir
        use_explicit_paths = (
            remote_working_dir is not None
            or remote_write_redirect_dir is not None
            or remote_sync_dir is not None
        )
        self._remote_working_dir = (
            remote_working_dir if use_explicit_paths else (remote_output_dir or ".kwbench/output")
        )
        self._remote_write_redirect_dir = (
            remote_write_redirect_dir if use_explicit_paths else (remote_output_dir or ".kwbench/output")
        )
        self._remote_write_mode = remote_write_mode if use_explicit_paths else ("redirect" if remote_output_dir else None)
        self._remote_sync_dir = remote_sync_dir if use_explicit_paths else (remote_output_dir or ".kwbench/output")
        self._remote_reference_files = dict(remote_reference_files or {})
        self._sandbox_template = sandbox_template
        self._sync_local_output_on_bootstrap = sync_local_output_on_bootstrap
        self._lock = threading.Lock()
        self._sandbox_id: str | None = None
        self._remote_state_path = f".kwbench/state/{self._task_id}.pkl"
        self._bootstrapped = False
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        _repl_instances.add(self)

    def run(self, code: str, timeout: int = 120) -> str:
        with self._lock:
            return _run_coroutine_sync(self._run_code(code, timeout=timeout))

    def run_bash(self, command: str, timeout: int = 120, cwd: str | None = None) -> str:
        with self._lock:
            return _run_coroutine_sync(self._run_bash(command, timeout=timeout, cwd=cwd))

    def sync_outputs(self, local_dir: str) -> None:
        with self._lock:
            _run_coroutine_sync(self._sync_outputs(local_dir))

    def upload_file(self, local_path: str, remote_path: str) -> None:
        with self._lock:
            _run_coroutine_sync(self._upload_file(local_path, remote_path))

    def close(self) -> None:
        with self._lock:
            _run_coroutine_sync(self._close_remote())

    async def _run_code(self, code: str, *, timeout: int) -> str:
        async with self._daytona_client() as daytona:
            sandbox = await self._ensure_sandbox(daytona)
            try:
                await self._start_sandbox(sandbox)
                await self._prepare_python_context(sandbox)
                result = await sandbox.code_interpreter.run_code(code, timeout=timeout)
                output = self._render_code_result(result)
                await sandbox.code_interpreter.run_code(_DAYTONA_CHECKPOINT_SCRIPT, timeout=timeout)
                return self._truncate_output(output)
            except Exception as exc:  # noqa: BLE001
                if "timeout" in str(exc).lower():
                    return f"[TIMEOUT after {timeout}s]"
                return f"[ERROR: {exc}]"
            finally:
                await self._stop_sandbox(sandbox)

    async def _run_bash(self, command: str, *, timeout: int, cwd: str | None) -> str:
        async with self._daytona_client() as daytona:
            sandbox = await self._ensure_sandbox(daytona)
            try:
                await self._start_sandbox(sandbox)
                response = await sandbox.process.exec(
                    command,
                    cwd=cwd or self._remote_working_dir,
                    timeout=timeout,
                )
                out = self._render_process_result(response)
                return self._truncate_output(out)
            except Exception as exc:  # noqa: BLE001
                if "timeout" in str(exc).lower():
                    return f"[TIMEOUT after {timeout}s]"
                return f"[ERROR: {exc}]"
            finally:
                await self._stop_sandbox(sandbox)

    async def _sync_outputs(self, local_dir: str) -> None:
        if not self._sandbox_id or not self._remote_sync_dir:
            return
        async with self._daytona_client() as daytona:
            sandbox = await daytona.get(self._sandbox_id)
            await self._start_sandbox(sandbox)
            try:
                listing = await sandbox.process.exec(
                    f"find {shlex.quote(self._remote_sync_dir)} -type f -print",
                    timeout=120,
                )
                files = [
                    line.strip()
                    for line in (listing.result or "").splitlines()
                    if line.strip()
                ]
                for remote_file in files:
                    rel_path = os.path.relpath(remote_file, self._remote_sync_dir)
                    local_path = Path(local_dir) / rel_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    content = await sandbox.fs.download_file(remote_file)
                    local_path.write_bytes(content)
            finally:
                await self._stop_sandbox(sandbox)

    async def _upload_file(self, local_path: str, remote_path: str) -> None:
        async with self._daytona_client() as daytona:
            sandbox = await self._ensure_sandbox(daytona)
            await self._start_sandbox(sandbox)
            try:
                parent = os.path.dirname(remote_path)
                if parent:
                    await sandbox.process.exec(self._ensure_remote_dir_cmd(parent), timeout=120)
                await sandbox.fs.upload_file(local_path, remote_path)
            finally:
                await self._stop_sandbox(sandbox)

    async def _close_remote(self) -> None:
        """Stop → archive → wait for backup → delete. Frees disk for new sandboxes."""
        sandbox_id = self._sandbox_id
        self._sandbox_id = None
        self._bootstrapped = False
        if not sandbox_id:
            return
        async with self._daytona_client() as daytona:
            try:
                sandbox = await daytona.get(sandbox_id)
                # Stop if running
                state = str(getattr(sandbox, "state", "")).lower()
                if state == "started":
                    await sandbox.stop(timeout=120)
                # Archive to free disk before delete
                try:
                    await sandbox.archive()
                    # Wait for archive to complete (backup moves to cold storage)
                    for _ in range(60):
                        sandbox = await daytona.get(sandbox_id)
                        if getattr(sandbox, "backup_state", None) == "Completed":
                            break
                        await asyncio.sleep(1)
                except Exception:
                    pass  # Archive failed — still delete
                await daytona.delete(sandbox)
            except Exception:
                pass

    @asynccontextmanager
    async def _daytona_client(self):
        AsyncDaytona, _CreateSandboxFromSnapshotParams = _daytona_imports()
        client = AsyncDaytona()
        try:
            yield client
        finally:
            await client.close()

    async def _ensure_sandbox(self, daytona: Any) -> Any:
        if self._sandbox_id:
            try:
                return await daytona.get(self._sandbox_id)
            except Exception:
                self._sandbox_id = None
                self._bootstrapped = False

        _AsyncDaytona, CreateSandboxFromSnapshotParams = _daytona_imports()
        params = CreateSandboxFromSnapshotParams(
            snapshot=self._sandbox_template,
            auto_stop_interval=10,
        )
        if self._sandbox_template:
            sandbox = await daytona.create(params, timeout=120)
        else:
            sandbox = await daytona.create(params, timeout=120)
        self._sandbox_id = sandbox.id
        self._bootstrapped = False
        return sandbox

    async def _start_sandbox(self, sandbox: Any) -> None:
        state = str(getattr(sandbox, "state", "")).lower()
        if state != "started":
            await sandbox.start(timeout=120)

    async def _stop_sandbox(self, sandbox: Any) -> None:
        state = str(getattr(sandbox, "state", "")).lower()
        if state == "started":
            try:
                await sandbox.stop(timeout=120)
            except Exception:
                pass

    async def _prepare_python_context(self, sandbox: Any) -> None:
        if not self._bootstrapped:
            await self._prepare_filesystem(sandbox)
            self._bootstrapped = True
        bootstrap = _daytona_bootstrap_script(
            working_dir=self._remote_working_dir,
            write_redirect_dir=self._remote_write_redirect_dir,
            write_mode=self._remote_write_mode,
            state_path=self._remote_state_path,
            seed_globals=self._seed_globals,
        )
        await sandbox.code_interpreter.run_code(bootstrap, timeout=120)

    async def _prepare_filesystem(self, sandbox: Any) -> None:
        prep_dirs = {
            self._remote_working_dir,
            self._remote_write_redirect_dir,
            self._remote_sync_dir,
            os.path.dirname(self._remote_state_path) or ".kwbench/state",
        }
        prep_cmds = [self._ensure_remote_dir_cmd(path) for path in sorted(path for path in prep_dirs if path)]
        for remote_path in self._remote_reference_files.values():
            parent = os.path.dirname(remote_path)
            if parent:
                prep_cmds.append(self._ensure_remote_dir_cmd(parent))
        await sandbox.process.exec(" && ".join(prep_cmds), timeout=120)
        await sandbox.process.exec(
            "python -c \"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('cloudpickle') else 1)\" "
            "|| python -m pip install cloudpickle >/dev/null 2>&1 || true",
            timeout=300,
        )
        if self._sync_local_output_on_bootstrap and self._output_dir and self._remote_sync_dir:
            output_root = Path(self._output_dir)
            if output_root.exists():
                for local_file in sorted(path for path in output_root.rglob("*") if path.is_file()):
                    rel_path = local_file.relative_to(output_root).as_posix()
                    await sandbox.fs.upload_file(str(local_file), f"{self._remote_sync_dir}/{rel_path}")
        for local_path, remote_path in self._remote_reference_files.items():
            await sandbox.fs.upload_file(local_path, remote_path)

    @staticmethod
    def _render_code_result(result: Any) -> str:
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        error = getattr(result, "error", None)
        if error:
            name = getattr(error, "name", "ExecutionError")
            value = getattr(error, "value", "")
            tb = getattr(error, "traceback", "")
            parts = [stdout, stderr, f"{name}: {value}".strip()]
            if tb:
                parts.append(tb)
            return "".join(parts)
        return stdout + stderr

    @staticmethod
    def _render_process_result(result: Any) -> str:
        parts: list[str] = []
        for value in (
            getattr(result, "result", None),
            getattr(result, "stdout", None),
            getattr(result, "stderr", None),
            getattr(result, "output", None),
        ):
            if isinstance(value, str) and value and value not in parts:
                parts.append(value)
        artifacts = getattr(result, "artifacts", None)
        if artifacts is not None:
            for value in (
                getattr(artifacts, "stdout", None),
                getattr(artifacts, "stderr", None),
                getattr(artifacts, "output", None),
            ):
                if isinstance(value, str) and value and value not in parts:
                    parts.append(value)
        additional = getattr(result, "additional_properties", None)
        if isinstance(additional, dict):
            for key in ("stdout", "stderr", "output", "result"):
                value = additional.get(key)
                if isinstance(value, str) and value and value not in parts:
                    parts.append(value)
        exit_code = getattr(result, "exit_code", None)
        if isinstance(exit_code, int) and exit_code != 0:
            parts.append(f"\n[exit_code={exit_code}]")
        return "".join(parts).lstrip("\n")

    @staticmethod
    def _ensure_remote_dir_cmd(path: str) -> str:
        quoted = shlex.quote(path)
        return (
            f"(mkdir -p {quoted} || sudo mkdir -p {quoted}) >/dev/null 2>&1 && "
            f"((chmod 777 {quoted} >/dev/null 2>&1) || (sudo chmod 777 {quoted} >/dev/null 2>&1) || true)"
        )

    @staticmethod
    def _truncate_output(out: str) -> str:
        if len(out) > MAX_OUTPUT:
            return out[:MAX_OUTPUT] + f"\n[TRUNCATED at {MAX_OUTPUT} chars]"
        return out

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class LocalToolRuntime:
    name = "local"

    def create_session(
        self,
        *,
        task_id: str,
        seed_globals: dict[str, Any] | None = None,
        output_dir: str | None = None,
        working_dir: str | None = None,
        write_redirect_dir: str | None = None,
        write_mode: str | None = None,
        remote_output_dir: str | None = None,
        remote_working_dir: str | None = None,
        remote_write_redirect_dir: str | None = None,
        remote_write_mode: str | None = None,
        remote_sync_dir: str | None = None,
        remote_reference_files: dict[str, str] | None = None,
        sandbox_template: str | None = None,
        sync_local_output_on_bootstrap: bool = False,
    ) -> ToolSession:
        del task_id, remote_output_dir, remote_working_dir, remote_write_redirect_dir, remote_write_mode, remote_sync_dir
        del remote_reference_files, sandbox_template, sync_local_output_on_bootstrap
        return PythonREPL(
            seed_globals=seed_globals,
            output_dir=output_dir,
            working_dir=working_dir,
            write_redirect_dir=write_redirect_dir,
            write_mode=write_mode,
        )


class DaytonaToolRuntime:
    name = "daytona"

    def create_session(
        self,
        *,
        task_id: str,
        seed_globals: dict[str, Any] | None = None,
        output_dir: str | None = None,
        working_dir: str | None = None,
        write_redirect_dir: str | None = None,
        write_mode: str | None = None,
        remote_output_dir: str | None = None,
        remote_working_dir: str | None = None,
        remote_write_redirect_dir: str | None = None,
        remote_write_mode: str | None = None,
        remote_sync_dir: str | None = None,
        remote_reference_files: dict[str, str] | None = None,
        sandbox_template: str | None = None,
        sync_local_output_on_bootstrap: bool = False,
    ) -> ToolSession:
        return DaytonaPythonREPL(
            task_id=task_id,
            seed_globals=seed_globals,
            output_dir=output_dir,
            working_dir=working_dir,
            write_redirect_dir=write_redirect_dir,
            write_mode=write_mode,
            remote_output_dir=remote_output_dir,
            remote_working_dir=remote_working_dir,
            remote_write_redirect_dir=remote_write_redirect_dir,
            remote_write_mode=remote_write_mode,
            remote_sync_dir=remote_sync_dir,
            remote_reference_files=remote_reference_files,
            sandbox_template=sandbox_template,
            sync_local_output_on_bootstrap=sync_local_output_on_bootstrap,
        )


# Singleton per task — call reset() between tasks
repl = PythonREPL()


def make_tool_runtime(runtime_type: str = "local") -> ToolRuntime:
    if runtime_type == "daytona":
        return DaytonaToolRuntime()
    return LocalToolRuntime()


def make_tool_session(
    *,
    runtime_type: str = "local",
    task_id: str,
    seed_globals: dict[str, Any] | None = None,
    output_dir: str | None = None,
    working_dir: str | None = None,
    write_redirect_dir: str | None = None,
    write_mode: str | None = None,
    remote_output_dir: str | None = None,
    remote_working_dir: str | None = None,
    remote_write_redirect_dir: str | None = None,
    remote_write_mode: str | None = None,
    remote_sync_dir: str | None = None,
    remote_reference_files: dict[str, str] | None = None,
    sandbox_template: str | None = None,
    sync_local_output_on_bootstrap: bool = False,
) -> ToolSession:
    runtime = make_tool_runtime(runtime_type)
    return runtime.create_session(
        task_id=task_id,
        seed_globals=seed_globals,
        output_dir=output_dir,
        working_dir=working_dir,
        write_redirect_dir=write_redirect_dir,
        write_mode=write_mode,
        remote_output_dir=remote_output_dir,
        remote_working_dir=remote_working_dir,
        remote_write_redirect_dir=remote_write_redirect_dir,
        remote_write_mode=remote_write_mode,
        remote_sync_dir=remote_sync_dir,
        remote_reference_files=remote_reference_files,
        sandbox_template=sandbox_template,
        sync_local_output_on_bootstrap=sync_local_output_on_bootstrap,
    )


def make_repl(
    *,
    repl_mode: str = "local",
    task_id: str,
    seed_globals: dict[str, Any] | None = None,
    output_dir: str | None = None,
    working_dir: str | None = None,
    write_redirect_dir: str | None = None,
    write_mode: str | None = None,
    remote_output_dir: str | None = None,
    remote_working_dir: str | None = None,
    remote_write_redirect_dir: str | None = None,
    remote_write_mode: str | None = None,
    remote_sync_dir: str | None = None,
    remote_reference_files: dict[str, str] | None = None,
    sandbox_template: str | None = None,
    sync_local_output_on_bootstrap: bool = False,
) -> Any:
    """Compatibility wrapper. Prefer make_tool_session(runtime_type=...)."""
    return make_tool_session(
        runtime_type=repl_mode,
        task_id=task_id,
        seed_globals=seed_globals,
        output_dir=output_dir,
        working_dir=working_dir,
        write_redirect_dir=write_redirect_dir,
        write_mode=write_mode,
        remote_output_dir=remote_output_dir,
        remote_working_dir=remote_working_dir,
        remote_write_redirect_dir=remote_write_redirect_dir,
        remote_write_mode=remote_write_mode,
        remote_sync_dir=remote_sync_dir,
        remote_reference_files=remote_reference_files,
        sandbox_template=sandbox_template,
        sync_local_output_on_bootstrap=sync_local_output_on_bootstrap,
    )


def set_tool_concurrency(limit: int | None) -> None:
    """Set shared tool concurrency limit across threads. None disables gating."""
    global _tool_sem
    if limit is None:
        _tool_sem = None
        return
    if limit < 1:
        raise ValueError("tool concurrency limit must be >= 1")
    _tool_sem = threading.BoundedSemaphore(limit)


def reset_tool_metrics() -> None:
    global _in_flight_tools
    global _peak_in_flight_tools
    with _metrics_lock:
        _in_flight_tools = 0
        _peak_in_flight_tools = 0


def get_tool_metrics() -> dict[str, int]:
    with _metrics_lock:
        return {
            "in_flight_tools": _in_flight_tools,
            "peak_in_flight_tools": _peak_in_flight_tools,
        }


@contextmanager
def _track_tool():
    global _in_flight_tools
    global _peak_in_flight_tools
    with _metrics_lock:
        _in_flight_tools += 1
        if _in_flight_tools > _peak_in_flight_tools:
            _peak_in_flight_tools = _in_flight_tools
    try:
        yield
    finally:
        with _metrics_lock:
            _in_flight_tools -= 1


@contextmanager
def _tool_slot():
    if _tool_sem is None:
        with _track_tool():
            yield
        return
    with _tool_sem:
        with _track_tool():
            yield


def _tool_trace_enabled() -> bool:
    return os.environ.get("TOOL_TRACE_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tool_trace_preview_chars() -> int:
    raw = os.environ.get("TOOL_TRACE_PREVIEW_CHARS", "400")
    try:
        return max(0, int(raw))
    except ValueError:
        return 400


def _tool_trace_path() -> Path:
    raw = os.environ.get("TOOL_TRACE_PATH", "eval/results/tool_calls.jsonl")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _preview(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[TRUNCATED preview at {limit} chars]"


def _write_tool_trace(
    *,
    tool: str,
    started_s: float,
    timeout: int,
    input_text: str,
    output_text: str,
    error: str | None = None,
    return_code: int | None = None,
) -> None:
    if not _tool_trace_enabled():
        return
    preview_chars = _tool_trace_preview_chars()
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "timeout_s": timeout,
        "duration_s": round(time.perf_counter() - started_s, 6),
        "input_chars": len(input_text),
        "output_chars": len(output_text),
        "input_preview": _preview(input_text, preview_chars),
        "output_preview": _preview(output_text, preview_chars),
        "error": error,
        "return_code": return_code,
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
    }
    trace_path = _tool_trace_path()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with _tool_trace_lock:
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


def execute_code(code: str, timeout: int = 120, repl_instance: ToolSession | None = None) -> str:
    """Run Python code in stateful REPL, return stdout+stderr."""
    started_s = time.perf_counter()
    with _tool_slot():
        out = (repl_instance or repl).run(code, timeout)
    _write_tool_trace(
        tool="execute_code",
        started_s=started_s,
        timeout=timeout,
        input_text=code,
        output_text=out,
    )
    return out


def execute_bash(
    command: str,
    timeout: int = 120,
    cwd: str | None = None,
    repl_instance: Any | None = None,
) -> str:
    """Run a shell command, return stdout+stderr."""
    started_s = time.perf_counter()
    return_code: int | None = None
    error: str | None = None
    run_cwd = cwd or _CWD
    if repl_instance is not None and hasattr(repl_instance, "run_bash"):
        with _tool_slot():
            out = repl_instance.run_bash(command, timeout=timeout, cwd=run_cwd)
        _write_tool_trace(
            tool="bash",
            started_s=started_s,
            timeout=timeout,
            input_text=command,
            output_text=out,
        )
        return out
    with _tool_slot():
        try:
            r = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=run_cwd,
            )
            return_code = r.returncode
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            error = "timeout"
            out = f"[TIMEOUT after {timeout}s]"
        except Exception as e:
            error = str(e)
            out = f"[ERROR: {e}]"
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n[TRUNCATED at {MAX_OUTPUT} chars]"
    _write_tool_trace(
        tool="bash",
        started_s=started_s,
        timeout=timeout,
        input_text=command,
        output_text=out,
        error=error,
        return_code=return_code,
    )
    return out
