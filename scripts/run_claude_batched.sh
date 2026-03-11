#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "Missing .env in $ROOT_DIR"
  exit 1
fi

set -a
source .env
set +a

API_URL="${API_URL:-http://localhost:8000}"
LLM="${LLM:-claude}"
PROVIDER_CONFIG_PATH="${PROVIDER_CONFIG_PATH:-eval/config.yaml}"
BATCH_SIZE="${BATCH_SIZE:-64}"
START_INDEX="${START_INDEX:-1}"   # 1-based
EVAL_SEM="${EVAL_SEM:-64}"
CPU_SEM="${CPU_SEM:-8}"
POLL_S="${POLL_S:-5}"
JUDGE_ENABLED="${JUDGE_ENABLED:-true}"
JUDGE_SEM="${JUDGE_SEM:-4}"
JUDGE_CRITERION_WORKERS="${JUDGE_CRITERION_WORKERS:-4}"

if ! curl -fsS "$API_URL/health" >/dev/null; then
  echo "Server is not reachable at $API_URL"
  exit 1
fi

mapfile -t ALL_IDS < <(jq -r '.id' dataset.jsonl)
TOTAL="${#ALL_IDS[@]}"

if (( START_INDEX < 1 || START_INDEX > TOTAL )); then
  echo "START_INDEX must be between 1 and $TOTAL"
  exit 1
fi

idx=$((START_INDEX - 1))
batch_num=1
grand_cost="0"
grand_in_tokens=0
grand_out_tokens=0
grand_rows=0

echo "Total tasks: $TOTAL"
echo "Starting at index: $START_INDEX"
echo "Batch size: $BATCH_SIZE"
echo "Provider: $LLM"
echo "Judge enabled: $JUDGE_ENABLED (judge_sem=$JUDGE_SEM, judge_criterion_workers=$JUDGE_CRITERION_WORKERS)"
echo

while (( idx < TOTAL )); do
  end=$((idx + BATCH_SIZE))
  if (( end > TOTAL )); then
    end=$TOTAL
  fi
  count=$((end - idx))

  batch_json="$(
    printf "%s\n" "${ALL_IDS[@]:idx:count}" | jq -R . | jq -s .
  )"

  payload="$(
    jq -n \
      --arg engine "async" \
      --arg client "provider" \
      --arg llm "$LLM" \
      --arg provider_config_path "$PROVIDER_CONFIG_PATH" \
      --argjson eval_sem "$EVAL_SEM" \
      --argjson cpu_sem "$CPU_SEM" \
      --argjson judge_enabled "$JUDGE_ENABLED" \
      --argjson judge_sem "$JUDGE_SEM" \
      --argjson judge_criterion_workers "$JUDGE_CRITERION_WORKERS" \
      --arg wandb_project "${WANDB_PROJECT:-}" \
      --argjson task_ids "$batch_json" \
      '{
        engine: $engine,
        client: $client,
        llm: $llm,
        provider_config_path: $provider_config_path,
        eval_sem: $eval_sem,
        cpu_sem: $cpu_sem,
        judge_enabled: $judge_enabled,
        judge_sem: $judge_sem,
        judge_criterion_workers: $judge_criterion_workers,
        wandb_project: $wandb_project,
        task_ids: $task_ids
      }'
  )"

  echo "=== Batch $batch_num | tasks $((idx + 1))-$end ($count items) ==="
  start_resp="$(curl -fsS -X POST "$API_URL/eval/start" -H 'Content-Type: application/json' -d "$payload")"
  echo "$start_resp" | jq .

  while true; do
    status_resp="$(curl -fsS "$API_URL/eval/status")"
    state="$(echo "$status_resp" | jq -r '.status // "unknown"')"
    done_count="$(echo "$status_resp" | jq -r '.completed // 0')"
    total_count="$(echo "$status_resp" | jq -r '.total // 0')"
    echo "status=$state completed=$done_count/$total_count"

    if [[ "$state" == "completed" || "$state" == "done" ]]; then
      break
    fi
    if [[ "$state" == "error" ]]; then
      echo "Batch failed:"
      echo "$status_resp" | jq .
      exit 1
    fi
    sleep "$POLL_S"
  done

  out_path="$(echo "$status_resp" | jq -r '.output')"
  if [[ ! -f "$out_path" ]]; then
    echo "Missing output file: $out_path"
    exit 1
  fi

  batch_rows="$(wc -l < "$out_path" | tr -d ' ')"
  batch_cost="$(jq -s 'map(.llm_metadata.cost_usd // 0) | add // 0' "$out_path")"
  batch_in_tokens="$(jq -s 'map(.llm_metadata.usage.input_tokens // .llm_metadata.usage.prompt_tokens // 0) | add // 0' "$out_path")"
  batch_out_tokens="$(jq -s 'map(.llm_metadata.usage.output_tokens // .llm_metadata.usage.completion_tokens // 0) | add // 0' "$out_path")"

  grand_rows=$((grand_rows + batch_rows))
  grand_in_tokens=$((grand_in_tokens + batch_in_tokens))
  grand_out_tokens=$((grand_out_tokens + batch_out_tokens))
  grand_cost="$(awk -v a="$grand_cost" -v b="$batch_cost" 'BEGIN { printf "%.6f", a + b }')"

  echo "Batch output: $out_path"
  echo "Batch rows: $batch_rows"
  echo "Batch cost_usd: $batch_cost"
  echo "Batch tokens in/out: $batch_in_tokens / $batch_out_tokens"
  echo "Running totals -> rows: $grand_rows, cost_usd: $grand_cost, tokens in/out: $grand_in_tokens / $grand_out_tokens"
  echo

  idx=$end
  if (( idx >= TOTAL )); then
    break
  fi

  read -r -p "Continue to next batch? [y/N] " answer
  if [[ ! "$answer" =~ ^[Yy]$ ]]; then
    echo "Stopped by user after batch $batch_num."
    echo "Resume later with START_INDEX=$((idx + 1))"
    exit 0
  fi

  batch_num=$((batch_num + 1))
done

echo "All requested batches completed."
echo "Final totals -> rows: $grand_rows, cost_usd: $grand_cost, tokens in/out: $grand_in_tokens / $grand_out_tokens"
