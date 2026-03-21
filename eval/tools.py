"""Shared tool definitions and execution for agentic LLM modules."""

import atexit
import io
import json
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import weakref
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_TURNS = 50
MAX_OUTPUT = 50_000  # truncate output beyond this

SYSTEM_PROMPT = (
    "If the task requires file output, save files only inside the output path specified at the end of the prompt.\n"
    "Do not write files anywhere else in the repository or filesystem.\n"
    "Available Python libraries: pandas, numpy, openpyxl, python-docx, requests."
)

_CWD = str(Path(__file__).resolve().parent)

_tool_sem: threading.BoundedSemaphore | None = None
_metrics_lock = threading.Lock()
_in_flight_tools = 0
_peak_in_flight_tools = 0
_tool_trace_lock = threading.Lock()
_repl_instances: "weakref.WeakSet[PythonREPL]" = weakref.WeakSet()

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
        "If an output directory is provided in the prompt, write files only inside that directory. "
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
    _output_dir = os.path.realpath(output_dir)

    def _sandboxed_open(file, mode="r", *args, **kwargs):
        is_write = any(c in mode for c in "wxa")
        if is_write and isinstance(file, (str, Path)):
            resolved = os.path.realpath(os.path.join(_output_dir, file)) if not os.path.isabs(str(file)) else os.path.realpath(str(file))
            if not resolved.startswith(_output_dir + os.sep) and resolved != _output_dir:
                # Redirect: write to output_dir/basename
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
def _sandbox_builtins(output_dir: str):
    """Temporarily patch builtins.open AND io.open so ALL file writes
    (including from libraries like pandas, openpyxl, zipfile) are redirected
    into output_dir."""
    import builtins
    original_builtin_open = builtins.open
    original_io_open = io.open
    sandboxed = _make_sandboxed_open(output_dir, original_builtin_open)
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
    output_dir: str | None,
) -> None:
    globals_dict = {"__builtins__": __builtins__}
    if seed_globals:
        globals_dict.update(seed_globals)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    while True:
        code = request_queue.get()
        if code is None:
            return

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        run_cwd = output_dir or _CWD
        prev_cwd = os.getcwd()
        try:
            os.chdir(run_cwd)
            if output_dir:
                with _sandbox_builtins(output_dir) as sandboxed_open:
                    globals_dict["open"] = sandboxed_open
                    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                        exec(compile(code, "<repl>", "exec"), globals_dict)
            else:
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


atexit.register(_shutdown_repls)


class PythonREPL:
    """Stateful Python REPL. Variables persist across exec() calls."""

    def __init__(self, seed_globals: dict | None = None, output_dir: str | None = None):
        self._seed_globals = dict(seed_globals) if seed_globals else None
        self._output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
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
                self._output_dir,
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

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# Singleton per task — call reset() between tasks
repl = PythonREPL()


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


def execute_code(code: str, timeout: int = 120, repl_instance: PythonREPL | None = None) -> str:
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


def execute_bash(command: str, timeout: int = 120, cwd: str | None = None) -> str:
    """Run a shell command, return stdout+stderr."""
    started_s = time.perf_counter()
    return_code: int | None = None
    error: str | None = None
    run_cwd = cwd or _CWD
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
