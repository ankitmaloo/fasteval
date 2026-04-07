"""Scorer interface and implementations.

Scoring precedence (resolved by resolve_scorer):
  1. rubric present       -> RubricJudgeScorer
  2. ground_truth present -> deterministic scorer (exact/regex/contains/set)
  3. verifier present     -> VerifierScorer
  4. else                 -> NoopScorer
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ScorerResult:
    score: float
    detail: dict[str, Any] = field(default_factory=dict)
    method: str = "unscored"

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "score": self.score, "detail": self.detail}


@runtime_checkable
class Scorer(Protocol):
    """Protocol every scorer must satisfy."""

    method: str

    def score(self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None) -> ScorerResult: ...


# ---------------------------------------------------------------------------
# Deterministic scorers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Unicode NFKC + collapse whitespace + strip."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ExactMatchScorer:
    method = "exact_match"

    def score(self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None) -> ScorerResult:
        ground_truth = case.get("ground_truth", "")
        case_sensitive = case.get("case_sensitive", True)
        normalize = case.get("normalize", False)

        candidates = ground_truth if isinstance(ground_truth, list) else [ground_truth]

        def _prep(text: str) -> str:
            if normalize:
                text = _normalize_text(text)
            if not case_sensitive:
                text = text.lower()
            return text.strip()

        prepped_answer = _prep(answer)
        matched = any(_prep(str(gt)) == prepped_answer for gt in candidates)
        return ScorerResult(
            score=1.0 if matched else 0.0,
            detail={"matched": matched, "case_sensitive": case_sensitive, "normalize": normalize},
            method=self.method,
        )


class RegexMatchScorer:
    method = "regex"

    def score(self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None) -> ScorerResult:
        pattern = case.get("ground_truth", "")
        case_sensitive = case.get("case_sensitive", True)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            matched = bool(re.search(str(pattern), answer, flags))
        except re.error:
            matched = False
        return ScorerResult(
            score=1.0 if matched else 0.0,
            detail={"matched": matched, "pattern": str(pattern)},
            method=self.method,
        )


class ContainsScorer:
    method = "contains"

    def score(self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None) -> ScorerResult:
        ground_truth = case.get("ground_truth", "")
        case_sensitive = case.get("case_sensitive", True)
        normalize = case.get("normalize", False)

        candidates = ground_truth if isinstance(ground_truth, list) else [ground_truth]

        def _prep(text: str) -> str:
            if normalize:
                text = _normalize_text(text)
            if not case_sensitive:
                text = text.lower()
            return text

        prepped_answer = _prep(answer)
        matched_items = [str(gt) for gt in candidates if _prep(str(gt)) in prepped_answer]
        all_matched = len(matched_items) == len(candidates)
        return ScorerResult(
            score=1.0 if all_matched else len(matched_items) / max(len(candidates), 1),
            detail={"matched_count": len(matched_items), "total": len(candidates)},
            method=self.method,
        )


class SetMatchScorer:
    method = "set"

    def score(self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None) -> ScorerResult:
        ground_truth = case.get("ground_truth", [])
        case_sensitive = case.get("case_sensitive", True)
        normalize = case.get("normalize", False)

        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]

        def _prep(text: str) -> str:
            if normalize:
                text = _normalize_text(text)
            if not case_sensitive:
                text = text.lower()
            return text.strip()

        prepped_answer = _prep(answer)
        expected = {_prep(str(gt)) for gt in ground_truth}
        found = sum(1 for item in expected if item in prepped_answer)
        return ScorerResult(
            score=found / max(len(expected), 1),
            detail={"found": found, "expected": len(expected)},
            method=self.method,
        )


# ---------------------------------------------------------------------------
# Rubric judge scorer (delegates to eval.core)
# ---------------------------------------------------------------------------

class RubricJudgeScorer:
    """Wraps existing judge_rubric / score_rubric from eval.core.

    This scorer is meant to run under the judge_sem gate — callers are
    responsible for concurrency control.
    """

    method = "rubric_judge"

    def __init__(
        self,
        *,
        criterion_workers: int = 4,
        conv_store: Any = None,
        judge_provider: str | None = None,
        provider_config_path: str | None = None,
        repl_mode: str = "local",
        sandbox_template: str | None = None,
    ) -> None:
        self.criterion_workers = criterion_workers
        self.conv_store = conv_store
        self.judge_provider = judge_provider
        self.provider_config_path = provider_config_path
        self.repl_mode = repl_mode
        self.sandbox_template = sandbox_template

    def score(self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None) -> ScorerResult:
        from eval.core import build_judge_artifact_bundle, judge_rubric, score_rubric

        rubric = case.get("rubric")
        if not isinstance(rubric, dict):
            return ScorerResult(score=0.0, detail={"error": "missing_rubric"}, method=self.method)

        task_id = str(case.get("id", ""))
        output_dir_name = case.get("output_dir") or "artifacts"
        artifact_bundle = build_judge_artifact_bundle(str(output_dir_name), task_id=task_id)

        eval_results = judge_rubric(
            str(case.get("task", "")),
            answer,
            rubric,
            criterion_workers=self.criterion_workers,
            repl_seed=artifact_bundle["repl_seed"] or None,
            output_dir=artifact_bundle["artifact_root"],
            artifact_context=artifact_bundle["prompt_context"],
            conv_store=self.conv_store,
            task_id=task_id,
            judge_provider=self.judge_provider,
            provider_config_path=self.provider_config_path,
            repl_mode=self.repl_mode,
            sandbox_template=self.sandbox_template,
        )
        task_score = score_rubric(
            eval_results["mandatory"],
            eval_results["good_to_have"],
            eval_results["ideal"],
        )
        return ScorerResult(
            score=task_score,
            detail={**eval_results, "score": task_score},
            method=self.method,
        )


# ---------------------------------------------------------------------------
# Verifier scorer (placeholder — Phase 2 will add real execution)
# ---------------------------------------------------------------------------

class VerifierScorer:
    """Run a benchmark-supplied verifier command to score a case.

    IMPORTANT: verifier execution must happen under a bounded gate
    (cpu_sem or a dedicated verifier semaphore). It must NOT run
    inline in the single result-worker path when non-trivial.
    """

    method = "verifier"

    def score(self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None) -> ScorerResult:
        verifier = case.get("verifier")
        if not verifier:
            return ScorerResult(score=0.0, detail={"error": "missing_verifier"}, method=self.method)

        # Phase 2: actual verifier execution will be implemented here.
        # For now return unscored to indicate the verifier path is recognized
        # but not yet wired.
        return ScorerResult(
            score=0.0,
            detail={"verifier": str(verifier), "status": "not_implemented"},
            method=self.method,
        )


# ---------------------------------------------------------------------------
# Noop scorer
# ---------------------------------------------------------------------------

class NoopScorer:
    method = "unscored"

    def score(self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None) -> ScorerResult:
        return ScorerResult(score=0.0, detail={}, method=self.method)


# ---------------------------------------------------------------------------
# Scorer resolution
# ---------------------------------------------------------------------------

_DETERMINISTIC_SCORERS: dict[str, type] = {
    "exact": ExactMatchScorer,
    "regex": RegexMatchScorer,
    "contains": ContainsScorer,
    "set": SetMatchScorer,
}


def resolve_scorer(
    case: dict[str, Any],
    *,
    rubric_scorer_factory: Any | None = None,
) -> Scorer:
    """Resolve the appropriate scorer for a case based on its fields.

    rubric_scorer_factory: if provided, called with no args to produce a
    RubricJudgeScorer (allows caller to inject runtime config).
    """
    # 1. rubric present → rubric judge
    rubric = case.get("rubric")
    if isinstance(rubric, dict) and any(rubric.get(k) for k in ("mandatory", "good_to_have", "ideal")):
        if rubric_scorer_factory is not None:
            return rubric_scorer_factory()
        return RubricJudgeScorer()

    # 2. ground_truth present → deterministic
    if case.get("ground_truth") is not None:
        match_type = str(case.get("match_type", "exact"))
        scorer_cls = _DETERMINISTIC_SCORERS.get(match_type, ExactMatchScorer)
        return scorer_cls()

    # 3. verifier present → verifier
    if case.get("verifier") is not None:
        return VerifierScorer()

    # 4. fallback
    return NoopScorer()
