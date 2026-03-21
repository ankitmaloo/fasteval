from __future__ import annotations

import asyncio
import math
import time
from concurrent.futures import ProcessPoolExecutor


def run_tool_sync(mode: str, payload: int | float | str) -> str:
    if mode == "sleep":
        seconds = float(payload)
        time.sleep(seconds)
        return f"sleep:{seconds}"

    if mode == "cpu":
        units = int(payload)
        acc = 0.0
        total = max(1, units) * 300_000
        for i in range(total):
            acc += math.sqrt((i % 1_000) + 1)
        return f"cpu:{acc:.3f}"

    raise ValueError(f"Unknown tool mode: {mode}")


async def run_tool(
    *,
    pool: ProcessPoolExecutor,
    mode: str,
    payload: int | float | str,
) -> tuple[str, float]:
    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    result = await loop.run_in_executor(pool, run_tool_sync, mode, payload)
    elapsed = time.perf_counter() - started
    return result, elapsed
