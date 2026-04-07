"""Benchmark plugin protocol and default base class.

Every benchmark adapter implements this interface. Methods with a default
implementation are optional — the engine falls back to built-in behavior
when they return None.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from eval.scorers import ScorerResult


@dataclass(frozen=True)
class ExecutionProfile:
    """Benchmark-defined execution contract for one case.

    This separates:
    - prompt/output instructions shown to the model
    - runtime working directory
    - Python write redirection policy
    - artifact sync policy
    """

    prompt_output_policy: Literal["artifact_dir", "none"] = "artifact_dir"
    prompt_extra_instructions: str | None = None
    workspace_root: str | None = None
    bash_cwd: str | None = None
    python_cwd: str | None = None
    python_write_policy: Literal["artifact_dir", "cwd", "none"] = "artifact_dir"
    sync_policy: Literal["artifact_dir", "none"] = "artifact_dir"


@runtime_checkable
class BenchmarkPlugin(Protocol):
    """Contract that benchmark adapters must satisfy."""

    name: str

    def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
        """Load task rows into the common schema."""
        ...

    def build_case_context(self, case: dict[str, Any]) -> dict[str, Any] | None:
        """Build per-case context (references, seed globals, etc.).

        Return None to let the engine use its default context builder.
        Must remain lightweight — heavy setup belongs in prepare_case.
        """
        ...

    def build_prompt(self, case: dict[str, Any], context: dict[str, Any] | None) -> str | None:
        """Build the prompt string for a case.

        Return None to let the provider module handle prompt construction.
        """
        ...

    def allowed_tools(self, case: dict[str, Any]) -> list[str] | None:
        """Return tool whitelist for this case. None → use config defaults."""
        ...

    def execution_profile(self, case: dict[str, Any]) -> ExecutionProfile | None:
        """Return the execution contract for this case.

        None means the engine should use its generic default contract.
        """
        ...

    def score_case(
        self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None,
    ) -> ScorerResult | None:
        """Score a completed case. Return None to defer to resolve_scorer."""
        ...

    def summarize_run(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Aggregate run-level summary. Return None for default aggregation."""
        ...

    # -----------------------------------------------------------------------
    # Lifecycle hooks (Phase 3)
    #
    # All hooks are optional. Default implementations are no-ops.
    # Blocking local work MUST be wrapped in asyncio.to_thread or a
    # dedicated executor. Remote sandbox work is acceptable if awaited
    # asynchronously. Tool-heavy hooks must acquire a bounded execution
    # gate (e.g. cpu_sem).
    # -----------------------------------------------------------------------

    async def prepare_case(self, case: dict[str, Any], repl: Any) -> None:
        """Called before the agent loop, after REPL is created."""
        ...

    async def seed_environment(self, case: dict[str, Any], repl: Any) -> None:
        """Called after REPL created, before first agent turn."""
        ...

    async def finalize_case(self, case: dict[str, Any], answer: str, repl: Any) -> None:
        """Called after the agent loop completes."""
        ...

    async def collect_artifacts(self, case: dict[str, Any], repl: Any) -> dict[str, Any] | None:
        """Called after finalize to gather artifacts. Return None for defaults."""
        ...

    async def cleanup_case(self, case: dict[str, Any], repl: Any) -> None:
        """Called in finally block — even on error. Best-effort."""
        ...


class BaseBenchmarkPlugin:
    """Convenience base with no-op defaults for all optional methods."""

    name: str = "base"

    def build_case_context(self, case: dict[str, Any]) -> dict[str, Any] | None:
        return None

    def build_prompt(self, case: dict[str, Any], context: dict[str, Any] | None) -> str | None:
        return None

    def allowed_tools(self, case: dict[str, Any]) -> list[str] | None:
        return None

    def execution_profile(self, case: dict[str, Any]) -> ExecutionProfile | None:
        return None

    def score_case(
        self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None,
    ) -> ScorerResult | None:
        return None

    def summarize_run(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        return None

    async def prepare_case(self, case: dict[str, Any], repl: Any) -> None:
        pass

    async def seed_environment(self, case: dict[str, Any], repl: Any) -> None:
        pass

    async def finalize_case(self, case: dict[str, Any], answer: str, repl: Any) -> None:
        pass

    async def collect_artifacts(self, case: dict[str, Any], repl: Any) -> dict[str, Any] | None:
        return None

    async def cleanup_case(self, case: dict[str, Any], repl: Any) -> None:
        pass
