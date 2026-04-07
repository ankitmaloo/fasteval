"""KWBench benchmark plugin — the current default benchmark.

Extracts existing dataset loading and reference handling into the
plugin interface. All existing behavior is preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.benchmarks.base import BaseBenchmarkPlugin, ExecutionProfile
from eval.scorers import ScorerResult


class KWBenchPlugin(BaseBenchmarkPlugin):
    name = "kwbench"

    def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
        from service.engine import load_dataset, DATASET_PATH
        return load_dataset(dataset_path or DATASET_PATH)

    def build_case_context(self, case: dict[str, Any]) -> dict[str, Any] | None:
        """Return reference data for prompt construction.

        Returns None to let the engine use its default _provider_case_context
        path, which already handles kwbench reference files correctly.
        """
        return None

    def build_prompt(self, case: dict[str, Any], context: dict[str, Any] | None) -> str | None:
        """Return None — kwbench prompt construction lives in the provider modules."""
        return None

    def allowed_tools(self, case: dict[str, Any]) -> list[str] | None:
        """Return None — kwbench tool selection is config-driven."""
        return None

    def execution_profile(self, case: dict[str, Any]) -> ExecutionProfile | None:
        del case
        # Preserve the current kwbench contract explicitly:
        # prompt points at the artifact directory, bash/python default there,
        # Python writes are redirected there, and the runtime syncs it back.
        return ExecutionProfile(
            prompt_output_policy="artifact_dir",
            python_write_policy="artifact_dir",
            sync_policy="artifact_dir",
        )

    def score_case(
        self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None,
    ) -> ScorerResult | None:
        """Return None — defer to resolve_scorer (rubric-based for kwbench tasks)."""
        return None

    def summarize_run(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Compute kwbench-specific summary stats."""
        scores = []
        ok_count = 0
        failed_count = 0
        for r in results:
            status = r.get("status")
            if status == "ok" or "scoring" in r or "eval" in r:
                ok_count += 1
            if status == "error":
                failed_count += 1
            scoring_payload = r.get("scoring")
            if isinstance(scoring_payload, dict) and isinstance(scoring_payload.get("score"), (int, float)):
                scores.append(float(scoring_payload["score"]))
                continue
            eval_payload = r.get("eval")
            if isinstance(eval_payload, dict) and isinstance(eval_payload.get("score"), (int, float)):
                scores.append(float(eval_payload["score"]))

        return {
            "ok_count": ok_count,
            "failed_count": failed_count,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "scored_count": len(scores),
        }
