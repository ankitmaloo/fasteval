"""Subject LLM: Anthropic-compatible API (MiniMax M2.7, etc.)

Config injected by engine.py via ANTCOMPAT_BASE_URL, ANTCOMPAT_API_KEY, ANTCOMPAT_MODEL env vars.
Defaults live in eval/config.yaml, CLI args override.
"""

import asyncio
import os
import anthropic
from dotenv import load_dotenv

from exa_py import Exa

from eval.tools import (
    CODE_SCHEMA,
    BASH_SCHEMA,
    execute_code,
    execute_bash,
    MAX_TURNS,
    SYSTEM_PROMPT,
    max_turns_for_config,
)
from eval.log import log

load_dotenv()

_exa = Exa(os.environ.get("EXA_API_KEY"))

_base_url = os.environ.get("ANTCOMPAT_BASE_URL")
_api_key = os.environ.get("ANTCOMPAT_API_KEY")
_model = os.environ.get("ANTCOMPAT_MODEL")
assert _base_url and _api_key and _model, "ANTCOMPAT_BASE_URL, ANTCOMPAT_API_KEY, ANTCOMPAT_MODEL must be set (via config.yaml + engine.py)"

_client = anthropic.Anthropic(base_url=_base_url, api_key=_api_key)
_async_client = anthropic.AsyncAnthropic(base_url=_base_url, api_key=_api_key)

LLM_ID = f"antcompat-{_model}"

_code_tool = {
    "name": CODE_SCHEMA["name"],
    "description": CODE_SCHEMA["description"],
    "input_schema": CODE_SCHEMA["parameters"],
}

_bash_tool = {
    "name": BASH_SCHEMA["name"],
    "description": BASH_SCHEMA["description"],
    "input_schema": BASH_SCHEMA["parameters"],
}

_search_tool = {
    "name": "search",
    "description": "Search the web for current information. Returns titles, URLs, and highlights.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"],
    },
}


def _exa_search(query: str) -> str:
    results = _exa.search(query=query, type="auto", num_results=10,
                          contents={"highlights": {"max_characters": 2000}})
    parts = []
    for r in results.results:
        highlights = "\n".join(r.highlights) if r.highlights else ""
        parts.append(f"{r.title}: {r.url}\n{highlights}")
    return "\n\n".join(parts)


async def _exa_search_async(query: str) -> str:
    return await asyncio.to_thread(_exa_search, query)


def _build_tools(config: dict) -> list[dict]:
    tools = []
    if config.get("enable_code"):
        tools.append(_code_tool)
    if config.get("enable_bash"):
        tools.append(_bash_tool)
    if config.get("enable_search"):
        tools.append(_search_tool)
    return tools


def _usage_counts(usage) -> dict[str, int]:
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    }


def _response_payload(answer: str, model: str, usage_total: dict[str, int], total_turns: int) -> dict:
    return {
        "text": answer,
        "metadata": {
            "model": model,
            "usage": {**usage_total, "total_tokens": usage_total["input_tokens"] + usage_total["output_tokens"]},
            "total_turns": total_turns,
        },
    }


def generate(task: str, references: dict, config: dict) -> dict:
    from eval.core import build_prompt
    _prompt_output = config.get("_output_dir")
    _bash_cwd = config.get("_bash_cwd") or _prompt_output
    prompt = build_prompt(
        task,
        references,
        output_file=_prompt_output,
        extra_instructions=config.get("_prompt_repl_note"),
    )
    task_repl = config.get("_repl")
    tools = _build_tools(config)

    messages = [{"role": "user", "content": prompt}]
    all_text = []
    usage_total = {"input_tokens": 0, "output_tokens": 0}
    total_turns = 0
    max_turns = max_turns_for_config(config)

    _tid = config.get("_task_id", "")
    for turn in range(max_turns):
        total_turns = turn + 1
        kwargs = {"model": _model, "max_tokens": 16384, "system": SYSTEM_PROMPT, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = _client.messages.create(**kwargs)

        u = resp.usage
        counts = _usage_counts(u)
        usage_total["input_tokens"] += counts["input_tokens"]
        usage_total["output_tokens"] += counts["output_tokens"]
        log.info(f"    [ANTCOMPAT] [{_tid}] turn {total_turns}/{max_turns} {counts['input_tokens']}in/{counts['output_tokens']}out stop={resp.stop_reason}")

        for b in resp.content:
            if b.type == "text" and b.text:
                all_text.append(b.text)

        if resp.stop_reason == "end_turn":
            break

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            break

        messages.append({"role": "assistant", "content": resp.content})

        tool_results = []
        for tu in tool_uses:
            if tu.name == "execute_code":
                code = tu.input.get("code", "")
                log.info(f"    [CODE]\n{code}")
                result = execute_code(code, repl_instance=task_repl)
                log.info(f"    [CODE OUT] ({len(result)} chars)\n{result}")
            elif tu.name == "bash":
                cmd = tu.input.get("command", "")
                log.info(f"    [BASH]\n{cmd}")
                result = execute_bash(cmd, cwd=_bash_cwd, repl_instance=task_repl)
                log.info(f"    [BASH OUT] ({len(result)} chars)\n{result}")
            elif tu.name == "search":
                query = tu.input.get("query", "")
                log.info(f"    [SEARCH] {query}")
                result = _exa_search(query)
                log.info(f"    [SEARCH OUT] ({len(result)} chars)")
            else:
                result = f"[Unknown tool: {tu.name}]"
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    answer = "\n".join(all_text)
    log.info(f"    [ANTCOMPAT] [{_tid}] done: {total_turns} turns, {usage_total['input_tokens']+usage_total['output_tokens']} tokens")
    return _response_payload(answer, _model, usage_total, total_turns)


async def generate_async(task: str, references: dict, config: dict) -> dict:
    from eval.core import build_prompt
    _prompt_output = config.get("_output_dir")
    _bash_cwd = config.get("_bash_cwd") or _prompt_output
    prompt = build_prompt(
        task,
        references,
        output_file=_prompt_output,
        extra_instructions=config.get("_prompt_repl_note"),
    )
    task_repl = config.get("_repl")
    tools = _build_tools(config)

    messages = [{"role": "user", "content": prompt}]
    all_text = []
    usage_total = {"input_tokens": 0, "output_tokens": 0}
    total_turns = 0
    max_turns = max_turns_for_config(config)
    _tid = config.get("_task_id", "")

    for turn in range(max_turns):
        total_turns = turn + 1
        kwargs = {"model": _model, "max_tokens": 16384, "system": SYSTEM_PROMPT, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = await _async_client.messages.create(**kwargs)

        u = resp.usage
        counts = _usage_counts(u)
        usage_total["input_tokens"] += counts["input_tokens"]
        usage_total["output_tokens"] += counts["output_tokens"]
        log.info(f"    [ANTCOMPAT] [{_tid}] turn {total_turns}/{max_turns} {counts['input_tokens']}in/{counts['output_tokens']}out stop={resp.stop_reason}")

        for b in resp.content:
            if b.type == "text" and b.text:
                all_text.append(b.text)

        if resp.stop_reason == "end_turn":
            break

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            break

        messages.append({"role": "assistant", "content": resp.content})

        tool_results = []
        for tu in tool_uses:
            if tu.name == "execute_code":
                code = tu.input.get("code", "")
                log.info(f"    [CODE]\n{code}")
                result = await asyncio.to_thread(execute_code, code, repl_instance=task_repl)
                log.info(f"    [CODE OUT] ({len(result)} chars)\n{result}")
            elif tu.name == "bash":
                cmd = tu.input.get("command", "")
                log.info(f"    [BASH]\n{cmd}")
                result = await asyncio.to_thread(execute_bash, cmd, cwd=_bash_cwd, repl_instance=task_repl)
                log.info(f"    [BASH OUT] ({len(result)} chars)\n{result}")
            elif tu.name == "search":
                query = tu.input.get("query", "")
                log.info(f"    [SEARCH] {query}")
                result = await _exa_search_async(query)
                log.info(f"    [SEARCH OUT] ({len(result)} chars)")
            else:
                result = f"[Unknown tool: {tu.name}]"
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    answer = "\n".join(all_text)
    log.info(f"    [ANTCOMPAT] [{_tid}] done: {total_turns} turns, {usage_total['input_tokens']+usage_total['output_tokens']} tokens")
    return _response_payload(answer, _model, usage_total, total_turns)
