from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence


Message = dict[str, Any]


class TransientLLMError(RuntimeError):
    """Raised when an LLM call should be retried."""


@dataclass(slots=True)
class Response:
    text: str
    tool_needed: bool = False
    tool_input: int | float | str | None = None


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, messages: Sequence[Message]) -> Response:
        """Return the next model response for the provided messages."""


def _extract_case_and_step(messages: Sequence[Message]) -> tuple[str, int]:
    for message in reversed(messages):
        case_id = message.get("case_id")
        step = message.get("step")
        if case_id is not None and step is not None:
            return str(case_id), int(step)
    raise ValueError("Messages must include case_id and step metadata.")


def _stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


class FakeLLMClient(LLMClient):
    def __init__(
        self,
        *,
        base_latency_s: float = 0.01,
        jitter_s: float = 0.04,
        tool_ratio: float = 0.5,
        force_tool: bool | None = None,
    ) -> None:
        self.base_latency_s = base_latency_s
        self.jitter_s = jitter_s
        self.tool_ratio = tool_ratio
        self.force_tool = force_tool

    def _latency_for(self, case_id: str, step: int) -> float:
        h = _stable_hash(f"{case_id}:{step}:latency")
        bucket = (h % 1000) / 1000.0
        return self.base_latency_s + self.jitter_s * bucket

    def _needs_tool(self, case_id: str, step: int) -> bool:
        if step != 0:
            return False
        if self.force_tool is not None:
            return self.force_tool
        h = _stable_hash(f"{case_id}:tool")
        return (h % 10_000) / 10_000.0 < self.tool_ratio

    async def complete(self, messages: Sequence[Message]) -> Response:
        case_id, step = _extract_case_and_step(messages)
        await asyncio.sleep(self._latency_for(case_id, step))

        if self._needs_tool(case_id, step):
            payload = (_stable_hash(f"{case_id}:payload") % 5) + 1
            return Response(
                text=f"case={case_id} step={step} requesting tool",
                tool_needed=True,
                tool_input=payload,
            )

        return Response(
            text=f"case={case_id} step={step} final",
            tool_needed=False,
            tool_input=None,
        )


class ReplayLLMClient(LLMClient):
    def __init__(self, fixtures: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
        self._fixtures: dict[str, dict[str, dict[str, Any]]] = {
            str(case_id): {str(step): dict(payload) for step, payload in steps.items()}
            for case_id, steps in fixtures.items()
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "ReplayLLMClient":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Replay fixture must be a dict keyed by case_id.")
        return cls(data)

    async def complete(self, messages: Sequence[Message]) -> Response:
        case_id, step = _extract_case_and_step(messages)
        case_payload = self._fixtures.get(case_id)
        if case_payload is None:
            raise KeyError(f"Missing replay fixture for case_id={case_id!r}")
        step_payload = case_payload.get(str(step))
        if step_payload is None:
            raise KeyError(f"Missing replay fixture for case_id={case_id!r}, step={step}")

        return Response(
            text=str(step_payload.get("text", "")),
            tool_needed=bool(step_payload.get("tool_needed", False)),
            tool_input=step_payload.get("tool_input"),
        )


@dataclass(slots=True)
class ProviderCaseContext:
    task_text: str
    references: dict[str, Any]
    config: dict[str, Any]


class ProviderLLMClient(LLMClient):
    """Adapter to run existing eval/llms/*.py modules behind LLMClient."""

    def __init__(
        self,
        *,
        generate_fn: Callable[[str, dict[str, Any], dict[str, Any]], Any]
        | Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[Any]],
        case_contexts: Mapping[str, ProviderCaseContext],
    ) -> None:
        self._generate_fn = generate_fn
        self._generate_is_async = inspect.iscoroutinefunction(generate_fn)
        self._case_contexts = dict(case_contexts)
        self._case_metadata: dict[str, dict[str, Any] | None] = {}
        self._case_latencies_s: dict[str, float] = {}

    def get_case_metadata(self, case_id: str) -> dict[str, Any] | None:
        return self._case_metadata.get(case_id)

    def get_case_latency_s(self, case_id: str) -> float | None:
        return self._case_latencies_s.get(case_id)

    @staticmethod
    def _is_transient_provider_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True
        status = getattr(exc, "status", None)
        if isinstance(status, int) and status in {408, 409, 429, 500, 502, 503, 504}:
            return True

        transient_names = {
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "ServiceUnavailableError",
            "ResourceExhausted",
            "TooManyRequests",
            "DeadlineExceeded",
            "InternalServerError",
            "OverloadedError",
        }
        if type(exc).__name__ in transient_names:
            return True

        msg = str(exc).lower()
        transient_markers = (
            "rate limit",
            "too many requests",
            "resource exhausted",
            "service unavailable",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "connection reset",
            "try again",
            "429",
            "503",
        )
        return any(marker in msg for marker in transient_markers)

    async def complete(self, messages: Sequence[Message]) -> Response:
        case_id, _step = _extract_case_and_step(messages)
        context = self._case_contexts.get(case_id)
        if context is None:
            raise KeyError(f"Missing provider context for case_id={case_id!r}")

        started = time.perf_counter()
        try:
            if self._generate_is_async:
                raw = await self._generate_fn(
                    context.task_text, context.references, context.config
                )
            else:
                raw = await asyncio.to_thread(
                    self._generate_fn, context.task_text, context.references, context.config
                )
        except Exception as exc:  # noqa: BLE001
            if self._is_transient_provider_error(exc):
                raise TransientLLMError(str(exc)) from exc
            raise
        self._case_latencies_s[case_id] = time.perf_counter() - started

        if isinstance(raw, dict):
            text = str(raw.get("text", ""))
            metadata = raw.get("metadata")
            self._case_metadata[case_id] = (
                metadata if isinstance(metadata, dict) else None
            )
        else:
            text = str(raw)
            self._case_metadata[case_id] = None

        return Response(text=text, tool_needed=False, tool_input=None)
