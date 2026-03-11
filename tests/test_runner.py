from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from clients import (
    LLMClient,
    ProviderCaseContext,
    ProviderLLMClient,
    ReplayLLMClient,
    Response,
    TransientLLMError,
)
from runner import EvalCase, Runner


def _extract_case_and_step(messages: list[dict[str, Any]]) -> tuple[str, int]:
    for message in reversed(messages):
        if "case_id" in message and "step" in message:
            return str(message["case_id"]), int(message["step"])
    raise ValueError("missing case metadata")


class VariableLatencyClient(LLMClient):
    def __init__(self, latencies: dict[str, float]) -> None:
        self.latencies = latencies

    async def complete(self, messages: list[dict[str, Any]]) -> Response:
        case_id, step = _extract_case_and_step(messages)
        await asyncio.sleep(self.latencies.get(case_id, 0.01))
        return Response(text=f"{case_id}-step-{step}", tool_needed=False)


class FakeToolHeavyClient(LLMClient):
    async def complete(self, messages: list[dict[str, Any]]) -> Response:
        _case_id, step = _extract_case_and_step(messages)
        await asyncio.sleep(0.002)
        if step == 0:
            return Response(text="need-tool", tool_needed=True, tool_input=0.05)
        return Response(text="done", tool_needed=False)


class FlakyClient(LLMClient):
    def __init__(self, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.attempts: dict[str, int] = {}

    async def complete(self, messages: list[dict[str, Any]]) -> Response:
        case_id, step = _extract_case_and_step(messages)
        key = f"{case_id}:{step}"
        self.attempts[key] = self.attempts.get(key, 0) + 1
        if self.attempts[key] <= self.failures_before_success:
            raise TransientLLMError("temporary")
        return Response(text=f"ok-{key}", tool_needed=False)


def test_tool_concurrency_never_exceeds_cpu_sem() -> None:
    cases = [
        EvalCase(
            case_id=f"case-{i}",
            prompt="run tool",
            tool_mode="sleep",
            tool_payload=0.05,
            max_steps=2,
        )
        for i in range(40)
    ]
    runner = Runner(llm_client=FakeToolHeavyClient(), eval_sem=32, cpu_sem=4)
    results = asyncio.run(runner.run_cases(cases))

    assert len(results) == 40
    assert all(result.status == "ok" for result in results)
    assert runner.peak_in_flight_tools <= 4
    assert runner.peak_in_flight_tools == 4


def test_dynamic_scheduling_has_no_wave_gaps() -> None:
    latencies = {
        "case-0": 0.01,
        "case-1": 0.20,
        "case-2": 0.20,
        "case-3": 0.01,
        "case-4": 0.01,
        "case-5": 0.01,
    }
    cases = [EvalCase(case_id=f"case-{i}", prompt="simple", max_steps=1) for i in range(6)]
    runner = Runner(llm_client=VariableLatencyClient(latencies), eval_sem=3, cpu_sem=2)

    results = asyncio.run(runner.run_cases(cases))
    assert all(result.status == "ok" for result in results)

    wave_1_finish = max(runner.case_timings[f"case-{i}"][1] for i in range(3))
    second_wave_first_start = runner.case_timings["case-3"][0]
    assert second_wave_first_start < wave_1_finish


def test_runner_completes_all_223_cases() -> None:
    class MixedClient(LLMClient):
        async def complete(self, messages: list[dict[str, Any]]) -> Response:
            case_id, step = _extract_case_and_step(messages)
            await asyncio.sleep(0.001)
            if step == 0 and int(case_id.split("-")[1]) % 2 == 0:
                return Response(text="tool", tool_needed=True, tool_input=0.01)
            return Response(text="done", tool_needed=False)

    cases = [
        EvalCase(
            case_id=f"case-{i}",
            prompt="task",
            tool_mode="sleep",
            tool_payload=0.01,
            max_steps=2,
        )
        for i in range(223)
    ]
    runner = Runner(llm_client=MixedClient(), eval_sem=64, cpu_sem=8)
    results = asyncio.run(runner.run_cases(cases))

    assert len(results) == 223
    assert all(result.status == "ok" for result in results)
    assert runner.peak_in_flight_evals <= 64
    assert runner.peak_in_flight_tools <= 8


def test_retry_behavior_handles_transient_errors() -> None:
    client = FlakyClient(failures_before_success=2)
    cases = [EvalCase(case_id=f"retry-{i}", prompt="retry", max_steps=1) for i in range(6)]
    runner = Runner(llm_client=client, eval_sem=6, cpu_sem=2, max_retries=2, retry_base_s=0.001)
    results = asyncio.run(runner.run_cases(cases))

    assert all(result.status == "ok" for result in results)
    assert all(attempts == 3 for attempts in client.attempts.values())


def test_replay_client_works_with_tool_flow(tmp_path: Path) -> None:
    fixture_path = tmp_path / "replay.json"
    fixture_path.write_text(
        json.dumps(
            {
                "case-a": {
                    "0": {"text": "use-tool", "tool_needed": True, "tool_input": 0.01},
                    "1": {"text": "final", "tool_needed": False},
                }
            }
        ),
        encoding="utf-8",
    )
    client = ReplayLLMClient.from_json(fixture_path)
    runner = Runner(llm_client=client, eval_sem=4, cpu_sem=2)
    results = asyncio.run(
        runner.run_cases(
            [
                EvalCase(
                    case_id="case-a",
                    prompt="from fixture",
                    tool_mode="sleep",
                    tool_payload=0.01,
                    max_steps=2,
                )
            ]
        )
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].output_text == "final"
    assert results[0].tool_calls == 1


def test_provider_rate_limit_retries_same_task() -> None:
    class RateLimitError(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.status_code = 429

    attempts = {"count": 0}

    def flaky_generate(task: str, references: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise RateLimitError("rate limit, please retry")
        return {"text": f"ok::{task}", "metadata": {"attempt": attempts["count"]}}

    client = ProviderLLMClient(
        generate_fn=flaky_generate,
        case_contexts={
            "case-rate-limit": ProviderCaseContext(
                task_text="run",
                references={},
                config={},
            )
        },
    )
    runner = Runner(llm_client=client, eval_sem=1, cpu_sem=1, max_retries=3, retry_base_s=0.001)
    results = asyncio.run(
        runner.run_cases([EvalCase(case_id="case-rate-limit", prompt="run", max_steps=1)])
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].output_text == "ok::run"
    assert attempts["count"] == 3
