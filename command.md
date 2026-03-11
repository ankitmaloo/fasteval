cat > /tmp/eval_start_full.json <<'JSON'
{
  "engine": "async",
  "llm": "openai",
  "dataset": "/Users/ankit/Documents/dev/RL/benchmarks/evals/dataset.jsonl",
  "task_ids": null,
  "output": "/Users/ankit/Documents/dev/RL/benchmarks/evals/service/results/final-openai-benchmark.jsonl",
  "wandb_project": "kwb",

  "client": "provider",
  "replay_fixtures": null,
  "provider_config_path": "/Users/ankit/Documents/dev/RL/benchmarks/evals/eval/config.yaml",

  "eval_sem": 64,
  "cpu_sem": 8,
  "max_retries": 3,
  "retry_base_s": 0.05,

  "fake_base_latency_s": 0.01,
  "fake_jitter_s": 0.04,
  "fake_tool_ratio": 0.5,
  "case_tool_mode": "sleep",
  "case_tool_payload": 0.02,
  "case_max_steps": 3,

  "judge_enabled": true,
  "judge_sem": 8,
  "judge_criterion_workers": 8,

  "hf_results_upload": true,
  "hf_results_repo": "clio-ai/kwbresults",
  "hf_results_token": "",
  "hf_repo": "clio-ai/kwbench",
  "hf_fetch_if_missing": false,
  "hf_force_refresh": false
}
JSON

curl -X POST http://localhost:8000/eval/start \
  -H 'Content-Type: application/json' \
  --data @/tmp/eval_start_full.json
