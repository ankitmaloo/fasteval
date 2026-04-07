"""Terminal-Bench plugin — terminal/CLI agent tasks with pytest verification in Daytona sandboxes.

Source: https://github.com/harbor-framework/terminal-bench
241 tasks across security, sysadmin, data science, ML, software engineering, etc.

Each task has:
  - instruction: what the agent must do
  - Dockerfile: container environment (not used directly — Daytona sandbox replaces it)
  - tests/test_outputs.py: pytest verification of final container state

Execution flow:
  1. Plugin loads task metadata from the cloned repo
  2. Engine creates a Daytona sandbox per task
  3. Agent gets bash + code tools in the sandbox
  4. After agent finishes, score_case() runs pytest in the SAME sandbox
  5. Score = pytest exit code (0 = pass, else fail)
  6. Sandbox is torn down by release_case()

Configuration (all on the task dict, all optional):
  - work_dir: where the agent should work (default: /app)
  - setup_cmd: commands to run before agent starts (default: create work_dir)
  - verify_cmd: how to run tests (default: pytest)
  - verify_timeout: seconds for verification (default: max_test_timeout_sec)

Requires:
  - Clone: git clone https://github.com/harbor-framework/terminal-bench.git /tmp/terminal-bench
  - Daytona: repl_mode=daytona in the run config
"""

from __future__ import annotations

import asyncio
import base64
import posixpath
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from eval.benchmarks.base import BaseBenchmarkPlugin, ExecutionProfile
from eval.scorers import ScorerResult
from eval.tools import execute_bash

_DEFAULT_REPO = Path("/tmp/terminal-bench")

# Defaults — override per-task or globally via config
_DEFAULT_WORK_DIR = "/app"
_DEFAULT_SETUP_CMD = "sudo mkdir -p {work_dir} && sudo chmod 777 {work_dir}"
_DEFAULT_VERIFY_CMD = "cd {work_dir} && python -m pytest {test_path} -v --tb=short 2>&1; echo \"EXIT_CODE=$?\""
_DEFAULT_PYTEST_INSTALL = "python -m pytest --version >/dev/null 2>&1 || pip install pytest -q >/dev/null 2>&1"


class TerminalBenchPlugin(BaseBenchmarkPlugin):
    name = "terminal_bench"

    def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
        repo_root = Path(dataset_path) if dataset_path else _DEFAULT_REPO
        tasks_dir = repo_root / "original-tasks"
        if not tasks_dir.exists():
            raise ValueError(
                f"Terminal-bench tasks not found at {tasks_dir}. "
                "Clone the repo: git clone https://github.com/harbor-framework/terminal-bench.git /tmp/terminal-bench"
            )

        cases = []
        for task_dir in sorted(tasks_dir.iterdir()):
            task_yaml = task_dir / "task.yaml"
            if not task_yaml.exists():
                continue
            raw = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
            if not raw or not raw.get("instruction"):
                continue

            test_content = _read_test_file(task_dir)
            work_dir = raw.get("work_dir", _DEFAULT_WORK_DIR)

            cases.append({
                # Engine required
                "id": task_dir.name,
                "task": raw["instruction"],

                # Terminal-bench metadata
                "difficulty": raw.get("difficulty", "medium"),
                "category": raw.get("category", "unknown"),
                "tags": raw.get("tags", []),
                "source": "terminal-bench",

                # Execution config (all overridable)
                "work_dir": work_dir,
                "setup_cmd": raw.get("setup_cmd", _DEFAULT_SETUP_CMD.format(work_dir=work_dir)),
                "verify_cmd": raw.get("verify_cmd"),  # None = use default pytest
                "verify_timeout": int(raw.get("max_test_timeout_sec", 180)),
                "max_agent_timeout_sec": raw.get("max_agent_timeout_sec", 900),
                "run_tests_in_same_shell": raw.get("run_tests_in_same_shell", True),

                # Paths
                "task_dir": str(task_dir),
                "has_dockerfile": (task_dir / "Dockerfile").exists(),
                "has_tests": (task_dir / "tests").exists(),
                "test_content": test_content,
                "materialization_steps": _parse_dockerfile_steps(task_dir),
            })

        return cases

    def allowed_tools(self, case: dict[str, Any]) -> list[str] | None:
        return ["bash", "code"]

    def execution_profile(self, case: dict[str, Any]) -> ExecutionProfile | None:
        work_dir = str(case.get("work_dir", _DEFAULT_WORK_DIR))
        return ExecutionProfile(
            prompt_output_policy="none",
            prompt_extra_instructions=(
                f"Use `{work_dir}` as the benchmark workspace and create files exactly at the task-specified paths. "
                "Do not relocate outputs into the harness artifact directory."
            ),
            workspace_root=work_dir,
            bash_cwd=work_dir,
            python_cwd=work_dir,
            python_write_policy="cwd",
            sync_policy="none",
        )

    def build_prompt(self, case: dict[str, Any], context: dict[str, Any] | None) -> str | None:
        parts = [case["task"]]
        work_dir = case.get("work_dir", _DEFAULT_WORK_DIR)
        timeout = case.get("max_agent_timeout_sec", 900)
        parts.append(f"\nYou have a maximum of {int(timeout)} seconds to complete this task.")
        parts.append("\nYou are in a terminal environment. Use bash commands to accomplish the task.")
        parts.append(f"Work in the {work_dir} directory unless the task specifies otherwise.")
        return "\n".join(parts)

    async def prepare_case(self, case: dict[str, Any], repl: Any) -> None:
        if repl is None:
            return

        materialization_steps = case.get("materialization_steps") or []
        step_timeout = max(60, min(int(case.get("max_agent_timeout_sec", 900)), 600))

        for step in materialization_steps:
            step_type = step.get("type")
            if step_type == "upload":
                local_path = step.get("local")
                remote_path = step.get("remote")
                if local_path and remote_path:
                    await asyncio.to_thread(repl.upload_file, str(local_path), str(remote_path))
            elif step_type == "run":
                command = step.get("command")
                if command:
                    await asyncio.to_thread(
                        execute_bash,
                        command,
                        timeout=step_timeout,
                        cwd=step.get("cwd"),
                        repl_instance=repl,
                    )

        setup_cmd = case.get("setup_cmd", "")
        if setup_cmd:
            await asyncio.to_thread(
                execute_bash,
                setup_cmd,
                timeout=60,
                cwd=case.get("work_dir", _DEFAULT_WORK_DIR),
                repl_instance=repl,
            )

    def score_case(
        self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None,
    ) -> ScorerResult | None:
        """Run verification in the sandbox.

        The REPL (Daytona sandbox) is passed through artifacts["repl"].
        It's still alive — release_case() hasn't been called yet.

        Steps:
          1. Run setup_cmd (e.g. create work_dir)
          2. Upload test file
          3. Install pytest (if needed)
          4. Run verify_cmd (or default pytest)
          5. Parse exit code
        """
        repl = artifacts.get("tool_session") if artifacts else None
        if repl is None and artifacts:
            repl = artifacts.get("repl")
        test_content = case.get("test_content")

        if repl is None:
            return ScorerResult(
                score=0.0,
                detail={"error": "no_repl", "task_id": case.get("id")},
                method="terminal_bench_verifier",
            )

        if not test_content:
            return ScorerResult(
                score=0.0,
                detail={"error": "no_test_file", "task_id": case.get("id")},
                method="terminal_bench_verifier",
            )

        work_dir = case.get("work_dir", _DEFAULT_WORK_DIR)
        verify_timeout = case.get("verify_timeout", 180)
        test_path = "/tmp/tbench_tests/test_outputs.py"

        # 1. Upload test file (base64 to avoid quoting issues)
        encoded = base64.b64encode(test_content.encode()).decode()
        execute_bash(
            f"mkdir -p /tmp/tbench_tests && echo '{encoded}' | base64 -d > {test_path}",
            timeout=30,
            cwd=work_dir,
            repl_instance=repl,
        )

        # 2. Install pytest
        execute_bash(_DEFAULT_PYTEST_INSTALL, timeout=60, cwd=work_dir, repl_instance=repl)

        # 3. Run verification
        verify_cmd = case.get("verify_cmd")
        if verify_cmd is None:
            verify_cmd = _DEFAULT_VERIFY_CMD.format(work_dir=work_dir, test_path=test_path)

        pytest_output = execute_bash(verify_cmd, timeout=verify_timeout, cwd=work_dir, repl_instance=repl)

        # 5. Parse
        exit_code = _parse_exit_code(pytest_output)
        passed = exit_code == 0
        test_summary = _parse_pytest_summary(pytest_output)

        return ScorerResult(
            score=1.0 if passed else 0.0,
            detail={
                "exit_code": exit_code,
                "passed": passed,
                "pytest_output": pytest_output[-2000:] if len(pytest_output) > 2000 else pytest_output,
                **test_summary,
            },
            method="terminal_bench_verifier",
        )

    def summarize_run(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        total = len(results)
        scored = [r for r in results if isinstance(r.get("scoring"), dict)]
        passed = sum(1 for r in scored if r["scoring"].get("score", 0) == 1.0)

        by_difficulty: dict[str, dict[str, int]] = {}
        by_category: dict[str, dict[str, int]] = {}
        for r in results:
            diff = r.get("difficulty", "unknown")
            cat = r.get("category", "unknown")
            for group, key in [(by_difficulty, diff), (by_category, cat)]:
                group.setdefault(key, {"total": 0, "passed": 0})
                group[key]["total"] += 1
                if isinstance(r.get("scoring"), dict) and r["scoring"].get("score", 0) == 1.0:
                    group[key]["passed"] += 1

        return {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / max(total, 1), 4),
            "by_difficulty": by_difficulty,
            "by_category": by_category,
        }


def _read_test_file(task_dir: Path) -> str | None:
    test_dir = task_dir / "tests"
    if not test_dir.exists():
        return None
    for name in ("test_outputs.py", "test_output.py"):
        test_file = test_dir / name
        if test_file.exists():
            return test_file.read_text(encoding="utf-8")
    py_files = sorted(test_dir.glob("*.py"))
    if py_files:
        return py_files[0].read_text(encoding="utf-8")
    return None


def _parse_dockerfile_steps(task_dir: Path) -> list[dict[str, Any]]:
    dockerfile = task_dir / "Dockerfile"
    if not dockerfile.exists():
        return []

    steps: list[dict[str, Any]] = []
    work_dir = "/"
    for instruction, value in _dockerfile_instructions(dockerfile.read_text(encoding="utf-8")):
        if instruction == "WORKDIR":
            work_dir = _resolve_container_path(value, work_dir)
        elif instruction in {"COPY", "ADD"}:
            steps.extend(_expand_copy_add(task_dir, value, work_dir))
        elif instruction == "RUN":
            steps.append({"type": "run", "cwd": work_dir, "command": value})
    return steps


def _dockerfile_instructions(text: str) -> list[tuple[str, str]]:
    instructions: list[tuple[str, str]] = []
    pending = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if pending:
            stripped = f"{pending} {stripped}"
        if stripped.endswith("\\"):
            pending = stripped[:-1].rstrip()
            continue
        pending = ""
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            continue
        instructions.append((parts[0].upper(), parts[1].strip()))
    if pending:
        parts = pending.split(None, 1)
        if len(parts) == 2:
            instructions.append((parts[0].upper(), parts[1].strip()))
    return instructions


def _resolve_container_path(path: str, work_dir: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    base = work_dir or "/"
    return posixpath.normpath(posixpath.join(base, path))


def _expand_copy_add(task_dir: Path, value: str, work_dir: str) -> list[dict[str, Any]]:
    tokens = shlex.split(value)
    while tokens and tokens[0].startswith("--"):
        tokens.pop(0)
    if len(tokens) < 2:
        return []

    dest = _resolve_container_path(tokens[-1], work_dir)
    srcs = tokens[:-1]
    steps: list[dict[str, Any]] = []
    multiple = len(srcs) > 1

    for src in srcs:
        if "://" in src:
            continue
        local_src = task_dir / src
        if not local_src.exists():
            continue
        if local_src.is_dir():
            dest_root = dest.rstrip("/") or dest
            for path in sorted(p for p in local_src.rglob("*") if p.is_file()):
                rel = path.relative_to(local_src).as_posix()
                remote = posixpath.join(dest_root, rel)
                steps.append({"type": "upload", "local": str(path), "remote": remote})
            continue

        remote = dest
        if multiple or dest.endswith("/"):
            remote = posixpath.join(dest.rstrip("/"), local_src.name)
        steps.append({"type": "upload", "local": str(local_src), "remote": remote})

    return steps


def _parse_exit_code(output: str) -> int:
    m = re.search(r"EXIT_CODE=(\d+)", output)
    if m:
        return int(m.group(1))
    if "passed" in output and "failed" not in output and "error" not in output.lower():
        return 0
    return 1


def _parse_pytest_summary(output: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label in ("passed", "failed", "error", "warnings"):
        m = re.search(rf"(\d+) {label}", output)
        if m:
            summary[f"tests_{label}"] = int(m.group(1))
    return summary
