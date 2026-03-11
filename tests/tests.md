# Test Suite

Run all tests:

```bash
.venv/bin/python -m pytest tests/ -v
```

No tests call external APIs — safe to run without tokens.

---

## test_runner.py (5 tests)

Core async runner (`runner.py`) concurrency and correctness.

| Test | What it verifies |
|---|---|
| `test_tool_concurrency_never_exceeds_cpu_sem` | 40 tool-heavy cases, `cpu_sem=4` — peak tools never exceeds 4 |
| `test_dynamic_scheduling_has_no_wave_gaps` | Variable-latency cases — fast cases from wave 2 start before slow wave-1 cases finish (no wave barrier) |
| `test_runner_completes_all_223_cases` | 223 mixed cases, `eval_sem=64`, `cpu_sem=8` — all complete, semaphore bounds respected |
| `test_retry_behavior_handles_transient_errors` | FlakyClient fails twice then succeeds — runner retries correctly, all pass after 3 attempts |
| `test_replay_client_works_with_tool_flow` | ReplayLLMClient drives a tool→final flow from fixture JSON |

## test_engine.py (83 tests)

Engine layer (`service/engine.py`), tools (`eval/tools.py`), scoring (`eval/core.py`), clients (`clients.py`), and API validation (`service/api.py`).

### Dataset loading (5)
- JSONL and JSON array formats
- Blank line skipping, bad JSON error, non-list JSON error

### Resume logic — `_extract_completed` (4)
- Missing file returns empty
- Correct ok/failed counting from status and eval fields
- Skips malformed JSON lines
- Falls back to `case_id` when `id` is absent

### Meta file — `_write_meta` (3)
- Creates new meta file
- Merges updates into existing meta
- Atomic write via tmp+rename (no leftover `.tmp`)

### Result row — `_result_row` (4)
- All standard fields populated (id, case_id, llm_answer, status, metrics, runner, llm_id)
- Preserves arbitrary task keys in output row
- Omits `llm_metadata` when None, includes when present
- Extracts `thinking` from metadata into top-level key

### Thinking extraction — `_extract_thinking` (9)
- None/non-dict metadata returns None
- Direct `thinking` string, `reasoning` list
- Whitespace-only thinking ignored
- Multi-turn extraction with/without turn index
- Empty turns or turns without thinking return None

### Scoring — `score_rubric` (5)
- All pass = 1.0
- Any mandatory fail = 0.0
- Empty mandatory = base 0.40
- Partial good_to_have and ideal contributions

### Case conversion — `_to_case` (3)
- Basic task→EvalCase mapping
- Task-level overrides (tool_mode, tool_payload, max_steps) beat config defaults
- Fallback ID generation (`case-0007`)

### Task helpers — `_task_id`, `_task_prompt` (5)
- ID from `id` field or fallback
- Prompt from `task`, `prompt`, or empty

### W&B token extraction — `_wandb_usage_tokens` (6)
- None metadata, missing usage key
- `total_tokens`, `input_tokens`+`output_tokens`, `prompt_tokens`+`completion_tokens`
- Zero tokens returns None

### Judged answer building — `_build_judged_answer` (2)
- No output_dir returns raw answer
- With output_dir appends inline output files (mocked)

### Judge pipeline — `_judge_task_with_rubric` (4)
- No rubric or empty rubric returns None
- Mocked `judge_rubric`+`score_rubric` wired correctly
- Mandatory-only rubric

### Resume index preservation (1)
- After filtering completed tasks, remaining indices match original dataset positions

### Storage snapshot sync (3)
- `_schedule_sync_snapshot` creates snapshot files in `.sync_tmp/`
- `_upload_snapshot_and_cleanup` uploads then deletes snapshots
- Upload failure still cleans up snapshots

### PythonREPL (5)
- Basic execution, state persistence across calls
- Error capture (stderr), reset clears state
- Output truncation at `MAX_OUTPUT`

### Tool concurrency tracking (3)
- `set_tool_concurrency` + `reset_tool_metrics` + `get_tool_metrics`
- `set_tool_concurrency(None)` disables gating
- `set_tool_concurrency(0)` raises ValueError

### Runner callbacks and edge cases (4)
- `on_case_complete` async callback receives all results
- `on_case_complete` sync callback also works
- `max_steps` exceeded returns error status
- Empty case list returns `[]`

### FakeLLMClient determinism (3)
- Same inputs produce same outputs across instances
- `force_tool=True` forces tool on step 0
- No tool requested on step > 0

### End-to-end `_run_async_eval` (5)
- Fake client: 8 tasks, verifies JSONL rows + meta status/counts
- Task ID filtering: only requested IDs run
- Resume: pre-seeded completed rows skipped, new rows appended
- Judge with mock: rubric tasks get `eval` in output, non-rubric tasks don't
- Replay client: fixture-driven responses written correctly

### API request validation (9)
- `eval_sem=0`, `cpu_sem=0`, `max_retries=-1` rejected
- `case_max_steps=0`, `judge_sem=0`, `judge_criterion_workers=0` rejected
- Legacy engine requires `llm`, replay requires `replay_fixtures`, provider requires `llm`
- Valid request passes

## test_async_eval.py (9 tests)

Integration tests for the full async eval pipeline.

| Test | What it verifies |
|---|---|
| `test_async_eval_fake_client_writes_results_and_meta` | 30 fake cases — JSONL + meta written, semaphore bounds respected |
| `test_async_eval_resume_skips_already_completed_ids` | Pre-seeded 3/12 completed — only remaining 9 run, no duplicates |
| `test_async_eval_replay_client` | Replay fixtures produce correct answers |
| `test_async_eval_fetches_dataset_from_hf_snapshot_when_missing` | Mocked HF download materializes dataset + reference files |
| `test_async_eval_preserves_original_task_row_and_appends_runner_fields` | Custom fields from dataset preserved in output alongside runner fields |
| `test_async_eval_judge_pipeline_runs_in_parallel_and_is_capped` | 12 tasks with rubric — judge runs concurrently, peak capped at `judge_sem=3` |
| `test_async_eval_judge_skips_cases_without_rubric` | Only rubric tasks judged, others pass through without `eval` |
| `test_async_eval_judge_requires_gemini_key` | ValueError raised when `judge_enabled=True` without `GEMINI_API_KEY` |
| `test_async_eval_uploads_results_to_hf_dataset` | Mocked HF upload called with correct args, meta records upload status |

## test_async_provider.py (2 tests)

Provider client integration (real REPL, no external API).

| Test | What it verifies |
|---|---|
| `test_provider_client_supports_repl_and_tool_cap` | 16 tasks via demo provider — REPL works, tool concurrency capped, thinking extracted |
| `test_provider_prefers_generate_async_when_available` | Provider with `generate_async` uses it instead of sync `generate` |
