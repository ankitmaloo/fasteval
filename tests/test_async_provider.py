from __future__ import annotations

import inspect
import json
from pathlib import Path

from service.engine import AsyncRunConfig, run_async_eval


def _write_dataset(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(count):
            handle.write(json.dumps({"id": f"prov_{idx:03d}", "task": f"Task {idx}"}) + "\n")


def test_provider_client_supports_repl_and_tool_cap(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "results.jsonl"
    meta = output.with_suffix(".meta.json")
    llm_module = tmp_path / "demo_provider.py"

    llm_module.write_text(
        "\n".join(
            [
                'LLM_ID = "demo-provider"',
                "",
                "from eval.tools import execute_code",
                "",
                "def generate(task, references, config):",
                "    repl = config.get('_repl')",
                "    execute_code(\"import time; time.sleep(0.03); print('REPL_OK')\", repl_instance=repl)",
                "    out = execute_code(\"print('FINAL')\", repl_instance=repl)",
                "    return {",
                "        'text': f'{task}::{out.strip()}',",
                "        'metadata': {",
                "            'provider': 'demo',",
                "            'repl': True,",
                "            'turns': [{'turn': 0, 'thinking': 'plan first'}],",
                "        },",
                "    }",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _write_dataset(dataset, 16)
    cfg = AsyncRunConfig(
        client="provider",
        provider=str(llm_module),
        eval_sem=8,
        cpu_sem=2,
        case_max_steps=1,
    )
    run_async_eval(cfg, dataset_path=dataset, output_path=output)

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta_payload = json.loads(meta.read_text(encoding="utf-8"))

    assert len(rows) == 16
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["llm_metadata"]["repl"] is True for row in rows)
    assert all(row["thinking"][0]["thinking"] == "plan first" for row in rows)
    assert all("FINAL" in row["llm_answer"] for row in rows)
    assert meta_payload["status"] == "completed"
    assert meta_payload["peak_in_flight_tools"] <= 2
    assert meta_payload["peak_in_flight_tools"] > 0


def test_provider_prefers_generate_async_when_available(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "results_async.jsonl"
    llm_module = tmp_path / "demo_async_provider.py"

    llm_module.write_text(
        "\n".join(
            [
                'LLM_ID = "demo-async-provider"',
                "",
                "import asyncio",
                "",
                "def generate(task, references, config):",
                "    raise RuntimeError('sync generate should not be called')",
                "",
                "async def generate_async(task, references, config):",
                "    await asyncio.sleep(0.01)",
                "    return {'text': f'ASYNC::{task}', 'metadata': {'mode': 'async'}}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _write_dataset(dataset, 8)
    cfg = AsyncRunConfig(
        client="provider",
        provider=str(llm_module),
        eval_sem=8,
        cpu_sem=2,
        case_max_steps=1,
    )
    run_async_eval(cfg, dataset_path=dataset, output_path=output)

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 8
    assert all(row["llm_metadata"]["mode"] == "async" for row in rows)
    assert all(row["llm_answer"].startswith("ASYNC::") for row in rows)


def test_claude_provider_exposes_async_entrypoint() -> None:
    from eval.llms import claude

    assert hasattr(claude, "generate_async")
    assert inspect.iscoroutinefunction(claude.generate_async)
