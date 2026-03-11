"""Shared tool definitions and execution for agentic LLM modules."""

import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

MAX_TURNS = 50
MAX_OUTPUT = 50_000  # truncate output beyond this

SYSTEM_PROMPT = (
    "If the task requires file output, save to the path specified at the end of the prompt.\n"
    "For multiple files, create a folder at that path and save files inside it.\n"
    "Available Python libraries: pandas, numpy, openpyxl, python-docx, requests."
)

_CWD = str(Path(__file__).resolve().parent)

_tool_sem: threading.BoundedSemaphore | None = None
_metrics_lock = threading.Lock()
_in_flight_tools = 0
_peak_in_flight_tools = 0
_repl_run_lock = threading.Lock()
_tool_trace_lock = threading.Lock()

BASH_SCHEMA = {
    "name": "bash",
    "description": "Execute a bash command. Use for shell operations, installing packages, git, curl, etc.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute"},
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
        "Write output files to the artifacts/ directory. "
        "Print results to stdout."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
        "required": ["code"],
    },
}


class PythonREPL:
    """Stateful Python REPL. Variables persist across exec() calls."""

    def __init__(self, seed_globals: dict | None = None):
        self._globals = {"__builtins__": __builtins__}
        if seed_globals:
            self._globals.update(seed_globals)

    def run(self, code: str, timeout: int = 120) -> str:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with _repl_run_lock:
            prev_cwd = os.getcwd()
            try:
                os.chdir(_CWD)
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(compile(code, "<repl>", "exec"), self._globals)
            except Exception:
                stderr_buf.write(traceback.format_exc())
            finally:
                os.chdir(prev_cwd)
        out = stdout_buf.getvalue() + stderr_buf.getvalue()
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n[TRUNCATED at {MAX_OUTPUT} chars]"
        return out

    def reset(self):
        self._globals = {"__builtins__": __builtins__}


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


def execute_bash(command: str, timeout: int = 120) -> str:
    """Run a shell command in eval/ dir, return stdout+stderr."""
    started_s = time.perf_counter()
    return_code: int | None = None
    error: str | None = None
    with _tool_slot():
        try:
            r = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=_CWD,
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
