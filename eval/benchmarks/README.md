# Benchmark Plugins

## Running a benchmark

Set `benchmark` in the request. Everything else works the same as before.

```json
{
  "client": "provider",
  "llm": "deepseek-v3",
  "benchmark": "gsm8k",
  "judge_enabled": false,
  "case_max_steps": 1,
  "eval_sem": 16
}
```

Built-in benchmarks: `kwbench`, `gsm8k`.

## Adding a new benchmark

Two things:

### 1. Write a plugin file

Create `eval/benchmarks/my_bench.py`:

```python
from pathlib import Path
from typing import Any
from eval.benchmarks.base import BaseBenchmarkPlugin
from eval.scorers import ScorerResult


class MyBenchPlugin(BaseBenchmarkPlugin):
    name = "my_bench"

    def load_cases(self, dataset_path: Path | None) -> list[dict[str, Any]]:
        """Required. Return task rows in the common schema."""
        return [
            {
                "id": "case_001",
                "task": "What is 2+2?",
                "ground_truth": "4",       # for deterministic scoring
                # OR
                # "rubric": {...}           # for judge scoring
                # OR
                # "verifier": "test.sh"     # for verifier scoring
            }
        ]
```

### 2. Register in config.yaml

```yaml
benchmarks:
  my_bench:
    module: eval/benchmarks/my_bench.py
    class: MyBenchPlugin
```

Done. Run with `"benchmark": "my_bench"`.

## Scoring modes

Scoring is determined by what fields are on the task rows. The plugin does not need to implement scoring logic unless it wants custom behavior.

| Field on task row | Scoring mode | Needs judge LLM? |
|---|---|---|
| `rubric` | LLM judge evaluates each criterion | Yes (`judge_enabled: true`) |
| `ground_truth` | Deterministic compare (exact, regex, contains, set) | No |
| `verifier` | Run a verifier script | No |
| none of the above | Unscored | No |

When `judge_enabled` is false and both `rubric` and `ground_truth` are present, the engine falls back to `ground_truth` scoring.

### Deterministic scoring options

Set `match_type` on the task row to control comparison:

```json
{
  "ground_truth": "42",
  "match_type": "exact",
  "case_sensitive": false,
  "normalize": true
}
```

| `match_type` | Behavior |
|---|---|
| `exact` (default) | Exact string match after optional normalize/case fold |
| `regex` | `ground_truth` is a regex pattern |
| `contains` | All items in `ground_truth` (string or list) must appear in the answer |
| `set` | Fraction of `ground_truth` items found in the answer |

### Custom scoring

If the built-in modes aren't enough, implement `score_case` on the plugin:

```python
def score_case(self, case, answer, artifacts=None):
    # Return ScorerResult to override, or None to fall back to resolve_scorer
    return ScorerResult(score=1.0, detail={...}, method="my_method")
```

## Plugin interface

| Method | When called | Default if None | Required? |
|---|---|---|---|
| `load_cases(dataset_path)` | Run start | — | **Yes** |
| `build_prompt(case, context)` | Prompt construction | Sends only `task["task"]` to the model | **Yes, if your tasks have context beyond the `task` field.** Without this, custom fields like setup commands, environment details, file lists etc. never reach the model. |
| `build_case_context(case)` | Per-case context setup | Use `read_reference_files` | No |
| `allowed_tools(case)` | Tool selection | Use config flags | No |
| `score_case(case, answer, artifacts)` | After case completes | `resolve_scorer` from task fields | No |
| `summarize_run(results)` | After all cases complete | No summary | No |

## Task schema

Core fields (always present):

- `id` — unique task identifier
- `task` — the prompt/question text

Optional scoring fields:

- `rubric` — `{mandatory: [...], good_to_have: [...], ideal: [...]}`
- `ground_truth` — expected answer (string or list)
- `match_type` — `exact`, `regex`, `contains`, `set`
- `case_sensitive` — bool (default true)
- `normalize` — bool (default false, collapses whitespace + NFKC)
- `verifier` — path to verifier script

Optional execution fields:

- `reference_files` — list of paths to load as context
- `output_dir` — artifact output directory
- `config` — passed through to the provider module
- `setup` — setup command (metadata, not yet executed)
- `environment` — env vars for sandbox
- `allowed_tools` — tool whitelist override
- `timeout` — per-task timeout override
- `max_steps` — max agent turns

## Example: GSM8K (no judge)

```json
{
  "client": "provider",
  "llm": "deepseek-v3",
  "benchmark": "gsm8k",
  "judge_enabled": false,
  "case_max_steps": 1,
  "eval_sem": 16
}
```

Downloads 1319 test questions from HuggingFace. Model answers math questions. Plugin extracts the final number and compares to ground truth. Result rows get `"scoring": {"method": "gsm8k_numeric", "score": 0.0 or 1.0}`. Meta gets `benchmark_summary` with accuracy.

## Example: rubric-judged benchmark

```json
{
  "client": "provider",
  "llm": "claude",
  "benchmark": "kwbench",
  "judge_enabled": true,
  "judge_provider": "gemini",
  "judge_sem": 4,
  "eval_sem": 32
}
```

Tasks have `rubric` fields. After each case, the judge LLM evaluates each criterion (PASS/FAIL). Score = 0.40 (mandatory) + 0.35 (good_to_have fraction) + 0.25 (ideal fraction). Judge concurrency gated by `judge_sem`.
