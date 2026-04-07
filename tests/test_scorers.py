"""Tests for eval/scorers.py — scorer protocol, implementations, and resolution."""

from __future__ import annotations

import pytest

from eval.scorers import (
    ContainsScorer,
    ExactMatchScorer,
    NoopScorer,
    RegexMatchScorer,
    RubricJudgeScorer,
    SetMatchScorer,
    ScorerResult,
    VerifierScorer,
    resolve_scorer,
)


# ---------------------------------------------------------------------------
# ScorerResult
# ---------------------------------------------------------------------------

class TestScorerResult:
    def test_to_dict(self) -> None:
        r = ScorerResult(score=0.75, detail={"k": "v"}, method="test")
        d = r.to_dict()
        assert d == {"method": "test", "score": 0.75, "detail": {"k": "v"}}

    def test_defaults(self) -> None:
        r = ScorerResult(score=0.0)
        assert r.method == "unscored"
        assert r.detail == {}


# ---------------------------------------------------------------------------
# ExactMatchScorer
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_match(self) -> None:
        case = {"ground_truth": "42"}
        r = ExactMatchScorer().score(case, "42")
        assert r.score == 1.0
        assert r.method == "exact_match"

    def test_no_match(self) -> None:
        case = {"ground_truth": "42"}
        r = ExactMatchScorer().score(case, "43")
        assert r.score == 0.0

    def test_case_insensitive(self) -> None:
        case = {"ground_truth": "Hello", "case_sensitive": False}
        assert ExactMatchScorer().score(case, "hello").score == 1.0

    def test_normalize(self) -> None:
        case = {"ground_truth": "hello world", "normalize": True}
        assert ExactMatchScorer().score(case, "  hello   world  ").score == 1.0

    def test_list_ground_truth(self) -> None:
        case = {"ground_truth": ["yes", "Yes"]}
        assert ExactMatchScorer().score(case, "Yes").score == 1.0
        assert ExactMatchScorer().score(case, "no").score == 0.0

    def test_whitespace_stripped(self) -> None:
        case = {"ground_truth": "42"}
        assert ExactMatchScorer().score(case, " 42 ").score == 1.0


# ---------------------------------------------------------------------------
# RegexMatchScorer
# ---------------------------------------------------------------------------

class TestRegex:
    def test_match(self) -> None:
        case = {"ground_truth": r"\d{2,}"}
        r = RegexMatchScorer().score(case, "the answer is 42")
        assert r.score == 1.0
        assert r.method == "regex"

    def test_no_match(self) -> None:
        case = {"ground_truth": r"^exact$"}
        assert RegexMatchScorer().score(case, "not exact match").score == 0.0

    def test_case_insensitive(self) -> None:
        case = {"ground_truth": "hello", "case_sensitive": False}
        assert RegexMatchScorer().score(case, "HELLO world").score == 1.0

    def test_invalid_regex(self) -> None:
        case = {"ground_truth": "[invalid"}
        assert RegexMatchScorer().score(case, "anything").score == 0.0


# ---------------------------------------------------------------------------
# ContainsScorer
# ---------------------------------------------------------------------------

class TestContains:
    def test_all_present(self) -> None:
        case = {"ground_truth": ["foo", "bar"]}
        r = ContainsScorer().score(case, "foo and bar are here")
        assert r.score == 1.0
        assert r.method == "contains"

    def test_partial(self) -> None:
        case = {"ground_truth": ["foo", "bar"]}
        r = ContainsScorer().score(case, "only foo here")
        assert r.score == 0.5

    def test_none_present(self) -> None:
        case = {"ground_truth": ["foo", "bar"]}
        r = ContainsScorer().score(case, "nothing here")
        assert r.score == 0.0

    def test_single_string(self) -> None:
        case = {"ground_truth": "needle"}
        assert ContainsScorer().score(case, "find the needle").score == 1.0

    def test_case_insensitive(self) -> None:
        case = {"ground_truth": "hello", "case_sensitive": False}
        assert ContainsScorer().score(case, "say HELLO").score == 1.0


# ---------------------------------------------------------------------------
# SetMatchScorer
# ---------------------------------------------------------------------------

class TestSetMatch:
    def test_all_found(self) -> None:
        case = {"ground_truth": ["alpha", "beta"]}
        r = SetMatchScorer().score(case, "alpha and beta")
        assert r.score == 1.0
        assert r.method == "set"

    def test_partial(self) -> None:
        case = {"ground_truth": ["alpha", "beta", "gamma"]}
        r = SetMatchScorer().score(case, "alpha is here, gamma too")
        assert abs(r.score - 2 / 3) < 0.01

    def test_empty_ground_truth(self) -> None:
        case = {"ground_truth": []}
        r = SetMatchScorer().score(case, "anything")
        assert r.score == 0.0


# ---------------------------------------------------------------------------
# VerifierScorer
# ---------------------------------------------------------------------------

class TestVerifier:
    def test_placeholder_returns_not_implemented(self) -> None:
        case = {"verifier": "run_tests.sh"}
        r = VerifierScorer().score(case, "answer")
        assert r.method == "verifier"
        assert r.detail["status"] == "not_implemented"

    def test_missing_verifier(self) -> None:
        case = {}
        r = VerifierScorer().score(case, "answer")
        assert r.detail["error"] == "missing_verifier"


# ---------------------------------------------------------------------------
# NoopScorer
# ---------------------------------------------------------------------------

class TestNoop:
    def test_returns_unscored(self) -> None:
        r = NoopScorer().score({}, "answer")
        assert r.score == 0.0
        assert r.method == "unscored"


# ---------------------------------------------------------------------------
# resolve_scorer
# ---------------------------------------------------------------------------

class TestResolveScorer:
    def test_rubric_takes_priority(self) -> None:
        case = {
            "rubric": {"mandatory": ["check"]},
            "ground_truth": "42",
        }
        s = resolve_scorer(case)
        assert s.method == "rubric_judge"

    def test_ground_truth_exact(self) -> None:
        case = {"ground_truth": "42"}
        s = resolve_scorer(case)
        assert s.method == "exact_match"

    def test_ground_truth_regex(self) -> None:
        case = {"ground_truth": r"\d+", "match_type": "regex"}
        s = resolve_scorer(case)
        assert s.method == "regex"

    def test_ground_truth_contains(self) -> None:
        case = {"ground_truth": "word", "match_type": "contains"}
        s = resolve_scorer(case)
        assert s.method == "contains"

    def test_ground_truth_set(self) -> None:
        case = {"ground_truth": ["a", "b"], "match_type": "set"}
        s = resolve_scorer(case)
        assert s.method == "set"

    def test_ground_truth_unknown_type_defaults_to_exact(self) -> None:
        case = {"ground_truth": "42", "match_type": "unknown"}
        s = resolve_scorer(case)
        assert s.method == "exact_match"

    def test_verifier(self) -> None:
        case = {"verifier": "test.sh"}
        s = resolve_scorer(case)
        assert s.method == "verifier"

    def test_noop_fallback(self) -> None:
        case = {"task": "hello"}
        s = resolve_scorer(case)
        assert s.method == "unscored"

    def test_rubric_scorer_factory_called(self) -> None:
        case = {"rubric": {"mandatory": ["check"]}}
        sentinel = ExactMatchScorer()  # just something non-None
        s = resolve_scorer(case, rubric_scorer_factory=lambda: sentinel)
        assert s is sentinel

    def test_rubric_scorer_factory_none_falls_to_default(self) -> None:
        case = {"rubric": {"mandatory": ["check"]}}
        s = resolve_scorer(case, rubric_scorer_factory=None)
        assert s.method == "rubric_judge"

    def test_empty_rubric_falls_through(self) -> None:
        """Rubric with no criteria should not trigger rubric scorer."""
        case = {
            "rubric": {"mandatory": [], "good_to_have": [], "ideal": []},
            "ground_truth": "42",
        }
        s = resolve_scorer(case)
        assert s.method == "exact_match"

    def test_judge_disabled_with_both_rubric_and_gt(self) -> None:
        """When no rubric_scorer_factory is provided (judge off), rubric still wins
        in resolve_scorer. Callers must strip rubric themselves for GT fallback."""
        case = {"rubric": {"mandatory": ["check"]}, "ground_truth": "42"}
        s = resolve_scorer(case, rubric_scorer_factory=None)
        # Without the engine's strip-rubric logic, rubric still wins
        assert s.method == "rubric_judge"

        # Simulating what _score_case does: strip rubric when judge is off
        stripped = dict(case)
        stripped.pop("rubric", None)
        s2 = resolve_scorer(stripped, rubric_scorer_factory=None)
        assert s2.method == "exact_match"
