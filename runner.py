from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from clients import FakeLLMClient, LLMClient, ReplayLLMClient, Response, TransientLLMError
from tools import run_tool


@dataclass(slots=True)
class EvalCase:
    case_id: str
    prompt: str
    tool_mode: str = "sleep"
    tool_payload: int | float | str = 0.02
    max_steps: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCase":
        case_id = str(payload["case_id"])
        prompt = str(payload.get("prompt", ""))
        return cls(
            case_id=case_id,
            prompt=prompt,
            tool_mode=str(payload.get("tool_mode", "sleep")),
            tool_payload=payload.get("tool_payload", 0.02),
            max_steps=int(payload.get("max_steps", 3)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class CaseResult:
    case_id: str
    status: str
    output_text: str
    error: str | None
    model_wait_s: float
    tool_cpu_s: float
    total_s: float
    model_calls: int
    tool_calls: int
    peak_in_flight_evals: int
    peak_in_flight_tools: int
    started_at_s: float
    finished_at_s: float


class _ConcurrencyTracker:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.in_flight_evals = 0
        self.in_flight_tools = 0
        self.peak_in_flight_evals = 0
        self.peak_in_flight_tools = 0

    async def eval_enter(self) -> None:
        async with self._lock:
            self.in_flight_evals += 1
            self.peak_in_flight_evals = max(
                self.peak_in_flight_evals, self.in_flight_evals
            )

    async def eval_exit(self) -> None:
        async with self._lock:
            self.in_flight_evals -= 1

    async def tool_enter(self) -> None:
        async with self._lock:
            self.in_flight_tools += 1
            self.peak_in_flight_tools = max(
                self.peak_in_flight_tools, self.in_flight_tools
            )

    async def tool_exit(self) -> None:
        async with self._lock:
            self.in_flight_tools -= 1


class Runner:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        eval_sem: int = 64,
        cpu_sem: int | None = None,
        max_retries: int = 2,
        retry_base_s: float = 0.05,
    ) -> None:
        if eval_sem < 1:
            raise ValueError("eval_sem must be >= 1")
        default_cpu = min(8, os.cpu_count() or 1)
        cpu_limit = default_cpu if cpu_sem is None else cpu_sem
        if cpu_limit < 1:
            raise ValueError("cpu_sem must be >= 1")

        self.llm_client = llm_client
        self.eval_limit = eval_sem
        self.cpu_limit = cpu_limit
        self.max_retries = max_retries
        self.retry_base_s = retry_base_s

        self._eval_sem = asyncio.Semaphore(self.eval_limit)
        self._cpu_sem = asyncio.Semaphore(self.cpu_limit)
        self._tracker = _ConcurrencyTracker()
        self.case_timings: dict[str, tuple[float, float]] = {}

    @property
    def peak_in_flight_evals(self) -> int:
        return self._tracker.peak_in_flight_evals

    @property
    def peak_in_flight_tools(self) -> int:
        return self._tracker.peak_in_flight_tools

    @property
    def current_in_flight_evals(self) -> int:
        return self._tracker.in_flight_evals

    @property
    def current_in_flight_tools(self) -> int:
        return self._tracker.in_flight_tools

    async def run_cases(
        self,
        cases: Sequence[EvalCase],
        *,
        out_path: str | Path | None = None,
        on_case_complete: Callable[[CaseResult], Awaitable[None] | None] | None = None,
        on_case_start: Callable[[EvalCase, float], Awaitable[None] | None] | None = None,
        on_case_finish: Callable[[EvalCase, float], Awaitable[None] | None] | None = None,
    ) -> list[CaseResult]:
        indexed_cases = list(enumerate(cases))
        if not indexed_cases:
            return []

        queue: asyncio.Queue[tuple[int, EvalCase] | None] = asyncio.Queue()
        for item in indexed_cases:
            queue.put_nowait(item)

        results: list[CaseResult | None] = [None] * len(indexed_cases)
        run_started = time.perf_counter()
        worker_count = min(self.eval_limit, len(indexed_cases))

        with ProcessPoolExecutor(max_workers=self.cpu_limit) as pool:
            workers = [
                asyncio.create_task(
                    self._worker(
                        queue,
                        results,
                        pool,
                        run_started,
                        on_case_complete=on_case_complete,
                        on_case_start=on_case_start,
                        on_case_finish=on_case_finish,
                    )
                )
                for _ in range(worker_count)
            ]
            await queue.join()
            for _ in workers:
                queue.put_nowait(None)
            await asyncio.gather(*workers)

        typed_results = [result for result in results if result is not None]
        if len(typed_results) != len(indexed_cases):
            raise RuntimeError("Runner did not produce all results.")

        if out_path is not None:
            _write_jsonl(out_path, typed_results)
        return typed_results

    async def _worker(
        self,
        queue: asyncio.Queue[tuple[int, EvalCase] | None],
        results: list[CaseResult | None],
        pool: ProcessPoolExecutor,
        run_started: float,
        on_case_complete: Callable[[CaseResult], Awaitable[None] | None] | None,
        on_case_start: Callable[[EvalCase, float], Awaitable[None] | None] | None,
        on_case_finish: Callable[[EvalCase, float], Awaitable[None] | None] | None,
    ) -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            idx, case = item
            try:
                result = await self._run_case(
                    case,
                    pool=pool,
                    run_started=run_started,
                    on_case_start=on_case_start,
                    on_case_finish=on_case_finish,
                )
                results[idx] = result
                if on_case_complete is not None:
                    callback_result = on_case_complete(result)
                    if inspect.isawaitable(callback_result):
                        await callback_result
            finally:
                queue.task_done()

    async def _run_case(
        self,
        case: EvalCase,
        *,
        pool: ProcessPoolExecutor,
        run_started: float,
        on_case_start: Callable[[EvalCase, float], Awaitable[None] | None] | None,
        on_case_finish: Callable[[EvalCase, float], Awaitable[None] | None] | None,
    ) -> CaseResult:
        async with self._eval_sem:
            await self._tracker.eval_enter()
            case_started = time.perf_counter()
            if on_case_start is not None:
                callback_result = on_case_start(case, case_started)
                if inspect.isawaitable(callback_result):
                    await callback_result
            started_at = case_started - run_started

            model_wait_s = 0.0
            tool_cpu_s = 0.0
            model_calls = 0
            tool_calls = 0
            output_text = ""
            status = "ok"
            error: str | None = None

            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": case.prompt,
                    "case_id": case.case_id,
                    "step": 0,
                }
            ]

            try:
                for step in range(case.max_steps):
                    response, waited = await self._complete_with_retry(messages)
                    model_wait_s += waited
                    model_calls += 1

                    if not response.tool_needed:
                        output_text = response.text
                        break

                    tool_calls += 1
                    payload = (
                        response.tool_input
                        if response.tool_input is not None
                        else case.tool_payload
                    )

                    async with self._cpu_sem:
                        await self._tracker.tool_enter()
                        try:
                            tool_output, cpu_s = await run_tool(
                                pool=pool, mode=case.tool_mode, payload=payload
                            )
                        finally:
                            await self._tracker.tool_exit()

                    tool_cpu_s += cpu_s
                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_output,
                            "case_id": case.case_id,
                            "step": step,
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Tool output: {tool_output}",
                            "case_id": case.case_id,
                            "step": step + 1,
                        }
                    )
                else:
                    status = "error"
                    error = f"max_steps_exceeded ({case.max_steps})"
            except Exception as exc:  # noqa: BLE001
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
            finally:
                finished_at_raw = time.perf_counter()
                if on_case_finish is not None:
                    callback_result = on_case_finish(case, finished_at_raw)
                    if inspect.isawaitable(callback_result):
                        await callback_result
                finished_at = finished_at_raw - run_started
                total_s = finished_at_raw - case_started
                self.case_timings[case.case_id] = (started_at, finished_at)
                await self._tracker.eval_exit()

            return CaseResult(
                case_id=case.case_id,
                status=status,
                output_text=output_text,
                error=error,
                model_wait_s=model_wait_s,
                tool_cpu_s=tool_cpu_s,
                total_s=total_s,
                model_calls=model_calls,
                tool_calls=tool_calls,
                peak_in_flight_evals=self._tracker.peak_in_flight_evals,
                peak_in_flight_tools=self._tracker.peak_in_flight_tools,
                started_at_s=started_at,
                finished_at_s=finished_at,
            )

    async def _complete_with_retry(
        self, messages: Sequence[dict[str, Any]]
    ) -> tuple[Response, float]:
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = await self.llm_client.complete(messages)
                waited = time.perf_counter() - started
                return response, waited
            except TransientLLMError:
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(self.retry_base_s * (2**attempt))

        raise RuntimeError("Unreachable retry state.")


def _load_cases(path: str | Path) -> list[EvalCase]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("cases file must contain a list of objects.")
    return [EvalCase.from_dict(item) for item in payload]


def _write_jsonl(path: str | Path, results: Sequence[CaseResult]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def _build_client(args: argparse.Namespace) -> LLMClient:
    if args.client == "fake":
        return FakeLLMClient(
            base_latency_s=args.fake_base_latency_s,
            jitter_s=args.fake_jitter_s,
            tool_ratio=args.fake_tool_ratio,
        )
    if args.client == "replay":
        if not args.replay_fixtures:
            raise ValueError("--replay-fixtures is required when --client replay")
        return ReplayLLMClient.from_json(args.replay_fixtures)
    raise ValueError(f"Unsupported client type: {args.client}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM eval cases concurrently.")
    parser.add_argument("--cases", required=True, help="Path to cases.json")
    parser.add_argument("--out", required=True, help="Path to output JSONL")
    parser.add_argument("--client", choices=("fake", "replay"), default="fake")
    parser.add_argument("--replay-fixtures", default=None)
    parser.add_argument("--eval-sem", type=int, default=64)
    parser.add_argument("--cpu-sem", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-base-s", type=float, default=0.05)
    parser.add_argument("--fake-base-latency-s", type=float, default=0.01)
    parser.add_argument("--fake-jitter-s", type=float, default=0.04)
    parser.add_argument("--fake-tool-ratio", type=float, default=0.5)
    return parser.parse_args()


async def _run_from_cli(args: argparse.Namespace) -> list[CaseResult]:
    client = _build_client(args)
    runner = Runner(
        llm_client=client,
        eval_sem=args.eval_sem,
        cpu_sem=args.cpu_sem,
        max_retries=args.max_retries,
        retry_base_s=args.retry_base_s,
    )
    cases = _load_cases(args.cases)
    return await runner.run_cases(cases, out_path=args.out)


def main() -> None:
    args = _parse_args()
    results = asyncio.run(_run_from_cli(args))
    completed = sum(1 for result in results if result.status == "ok")
    failed = len(results) - completed
    print(
        f"completed={completed} failed={failed} total={len(results)} "
        f"eval_sem={args.eval_sem} cpu_sem={args.cpu_sem or min(8, os.cpu_count() or 1)}"
    )


if __name__ == "__main__":
    main()
