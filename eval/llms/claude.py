"""Subject LLM: Claude Opus 4.6 with code execution + search."""

import asyncio
import os
import anthropic
from dotenv import load_dotenv

from eval.tools import CODE_SCHEMA, BASH_SCHEMA, execute_code, execute_bash, MAX_TURNS, SYSTEM_PROMPT
from eval.log import log

load_dotenv()

LLM_ID = "claude-opus-4-6"

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
_async_client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
    "description": "Search the web for current information. Returns a summary of search results.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"],
    },
}

# Pricing per million tokens (Opus 4.6)
_COST_PER_M = {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_creation": 18.75}


def _web_search(query: str) -> str:
    """Use Claude to synthesize a search answer (no actual web tool yet)."""
    resp = _client.messages.create(
        model=LLM_ID, max_tokens=4096,
        messages=[{"role": "user", "content": f"Search and summarize: {query}"}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


async def _web_search_async(query: str) -> str:
    """Use Claude to synthesize a search answer (async)."""
    resp = await _async_client.messages.create(
        model=LLM_ID, max_tokens=4096,
        messages=[{"role": "user", "content": f"Search and summarize: {query}"}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _build_tools(config: dict) -> list[dict]:
    tools = []
    if config.get("enable_code"):
        tools.append(_code_tool)
    if config.get("enable_bash"):
        tools.append(_bash_tool)
    if config.get("enable_search"):
        tools.append(_search_tool)
    return tools


def _cost_from_usage(usage_total: dict[str, int]) -> float:
    return (
        usage_total["input_tokens"] * _COST_PER_M["input"]
        + usage_total["output_tokens"] * _COST_PER_M["output"]
        + usage_total["cache_read_input_tokens"] * _COST_PER_M["cache_read"]
        + usage_total["cache_creation_input_tokens"] * _COST_PER_M["cache_creation"]
    ) / 1_000_000


def _response_payload(answer: str, usage_total: dict[str, int], total_turns: int, cost: float) -> dict:
    return {
        "text": answer,
        "metadata": {
            "model": LLM_ID,
            "usage": {**usage_total, "total_tokens": usage_total["input_tokens"] + usage_total["output_tokens"]},
            "cost_usd": round(cost, 6),
            "total_turns": total_turns,
        },
    }


def generate(task: str, references: dict, config: dict) -> dict:
    from eval.core import build_prompt
    _odir = config.get("_output_dir")
    prompt = build_prompt(
        task,
        references,
        output_file=_odir,
        extra_instructions=config.get("_prompt_repl_note"),
    )
    task_repl = config.get("_repl")
    tools = _build_tools(config)

    messages = [{"role": "user", "content": prompt}]
    all_text = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    total_turns = 0

    for turn in range(MAX_TURNS):
        total_turns = turn + 1
        kwargs = {"model": LLM_ID, "max_tokens": 16384, "system": SYSTEM_PROMPT, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = _client.messages.create(**kwargs)

        # Track usage
        u = resp.usage
        usage_total["input_tokens"] += u.input_tokens
        usage_total["output_tokens"] += u.output_tokens
        usage_total["cache_read_input_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0
        usage_total["cache_creation_input_tokens"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        log.info(f"    [CLAUDE] turn {total_turns}/{MAX_TURNS} {u.input_tokens}in/{u.output_tokens}out stop={resp.stop_reason}")

        # Collect text from this turn
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
                result = execute_bash(cmd, cwd=_odir)
                log.info(f"    [BASH OUT] ({len(result)} chars)\n{result}")
            elif tu.name == "search":
                query = tu.input.get("query", "")
                log.info(f"    [SEARCH] {query}")
                result = _web_search(query)
                log.info(f"    [SEARCH OUT] ({len(result)} chars)")
            else:
                result = f"[Unknown tool: {tu.name}]"
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    cost = _cost_from_usage(usage_total)
    answer = "\n".join(all_text)
    log.info(f"    [CLAUDE] done: {total_turns} turns, {usage_total['input_tokens']+usage_total['output_tokens']} tokens, ${cost:.4f}")
    return _response_payload(answer, usage_total, total_turns, cost)


async def generate_async(task: str, references: dict, config: dict) -> dict:
    from eval.core import build_prompt
    _odir = config.get("_output_dir")
    prompt = build_prompt(
        task,
        references,
        output_file=_odir,
        extra_instructions=config.get("_prompt_repl_note"),
    )
    task_repl = config.get("_repl")
    tools = _build_tools(config)

    messages = [{"role": "user", "content": prompt}]
    all_text = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    total_turns = 0

    for turn in range(MAX_TURNS):
        total_turns = turn + 1
        kwargs = {"model": LLM_ID, "max_tokens": 16384, "system": SYSTEM_PROMPT, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = await _async_client.messages.create(**kwargs)

        # Track usage
        u = resp.usage
        usage_total["input_tokens"] += u.input_tokens
        usage_total["output_tokens"] += u.output_tokens
        usage_total["cache_read_input_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0
        usage_total["cache_creation_input_tokens"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        log.info(f"    [CLAUDE] turn {total_turns}/{MAX_TURNS} {u.input_tokens}in/{u.output_tokens}out stop={resp.stop_reason}")

        # Collect text from this turn
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
                result = await asyncio.to_thread(execute_bash, cmd, cwd=_odir)
                log.info(f"    [BASH OUT] ({len(result)} chars)\n{result}")
            elif tu.name == "search":
                query = tu.input.get("query", "")
                log.info(f"    [SEARCH] {query}")
                result = await _web_search_async(query)
                log.info(f"    [SEARCH OUT] ({len(result)} chars)")
            else:
                result = f"[Unknown tool: {tu.name}]"
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    cost = _cost_from_usage(usage_total)
    answer = "\n".join(all_text)
    log.info(f"    [CLAUDE] done: {total_turns} turns, {usage_total['input_tokens']+usage_total['output_tokens']} tokens, ${cost:.4f}")
    return _response_payload(answer, usage_total, total_turns, cost)
