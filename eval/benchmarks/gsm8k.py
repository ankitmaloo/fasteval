"""GSM8K benchmark plugin — grade-school math with deterministic numeric scoring.

Downloads from HuggingFace `openai/gsm8k` (main split). Each task row gets a
`ground_truth` field with the extracted numeric answer. Scoring extracts the
final number from the model's response and compares numerically.

No tools, no judge. Pure Q→A with deterministic scoring.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from eval.benchmarks.base import BaseBenchmarkPlugin
from eval.scorers import ScorerResult


def _extract_gsm8k_answer(text: str) -> str | None:
    """Extract the numeric answer from GSM8K's `#### <number>` format."""
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        return _normalize_number(m.group(1).strip())
    return None


def _normalize_number(text: str) -> str:
    """Strip commas, dollar signs, percent signs, trailing periods from a number string."""
    text = text.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    # Handle negative
    neg = text.startswith("-")
    if neg:
        text = text[1:]
    # Strip leading zeros but keep "0" and "0.x"
    if "." in text:
        integer, decimal = text.split(".", 1)
        integer = integer.lstrip("0") or "0"
        text = f"{integer}.{decimal}"
    else:
        text = text.lstrip("0") or "0"
    return f"-{text}" if neg else text


def _extract_model_number(text: str) -> str | None:
    """Extract the final numeric answer from a model's free-form response.

    Precedence:
    1. #### <number> (GSM8K convention)
    2. \\boxed{<number>} (LaTeX convention)
    3. Last standalone number in the text
    """
    # 1. #### convention
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        return _normalize_number(m.group(1).strip())

    # 2. \boxed{} convention
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return _normalize_number(m.group(1).strip())

    # 3. Last number in text (including negative, decimals, commas)
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return _normalize_number(numbers[-1])

    return None


class GSM8KPlugin(BaseBenchmarkPlugin):
    name = "gsm8k"

    def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
        """Load GSM8K from a local JSONL/JSON file or from HuggingFace."""
        if dataset_path is not None and dataset_path.exists():
            return self._load_local(dataset_path)
        return self._load_from_hf()

    def _load_local(self, path: Path) -> list[dict[str, Any]]:
        """Load from local file. Expects rows with `question` and `answer` fields."""
        import json

        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rows.append(self._to_case(raw, idx))
        return rows

    def _load_from_hf(self) -> list[dict[str, Any]]:
        """Download GSM8K test split from HuggingFace."""
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The `datasets` package is required for GSM8K. "
                "Install with: pip install datasets"
            ) from exc

        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [self._to_case(row, idx) for idx, row in enumerate(ds)]

    def _to_case(self, raw: dict[str, Any], idx: int) -> dict[str, Any]:
        question = raw.get("question") or raw.get("task") or ""
        answer_raw = raw.get("answer") or ""
        gt = _extract_gsm8k_answer(answer_raw)
        if gt is None:
            # Fallback: try the whole string as a number
            gt = _normalize_number(answer_raw) if answer_raw.strip() else ""

        case_id = raw.get("id") or f"gsm8k_{idx:04d}"
        return {
            "id": str(case_id),
            "task": question,
            "ground_truth": gt,
            "source": "gsm8k",
            "category": "math",
            "config": {},
        }

    def allowed_tools(self, case: dict[str, Any]) -> list[str] | None:
        """No tools for GSM8K — pure text Q&A."""
        return []

    def build_prompt(self, case: dict[str, Any], context: dict[str, Any] | None) -> str | None:
        """Simple math prompt."""
        return (
            f"{case['task']}\n\n"
            "Solve this step by step. "
            "At the end, write your final numeric answer after ####. "
            "For example: #### 42"
        )

    def score_case(
        self, case: dict[str, Any], answer: str, artifacts: dict[str, Any] | None = None,
    ) -> ScorerResult | None:
        """Extract the final number from the model's answer and compare to ground truth."""
        gt = case.get("ground_truth", "")
        model_number = _extract_model_number(answer)

        if model_number is None:
            return ScorerResult(
                score=0.0,
                detail={"expected": gt, "extracted": None, "reason": "no_number_found"},
                method="gsm8k_numeric",
            )

        matched = model_number == gt
        return ScorerResult(
            score=1.0 if matched else 0.0,
            detail={"expected": gt, "extracted": model_number, "matched": matched},
            method="gsm8k_numeric",
        )

    def summarize_run(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        total = len(results)
        scored = [
            r for r in results
            if isinstance(r.get("scoring"), dict)
        ]
        correct = sum(1 for r in scored if r["scoring"].get("score", 0) == 1.0)
        return {
            "total": total,
            "scored": len(scored),
            "correct": correct,
            "accuracy": round(correct / max(len(scored), 1), 4),
        }
