"""Subject LLM: OpenAI-compatible chat completions (3rd-party models like Kimi, DeepSeek, Qwen, etc.)

Config injected by run.py via OAICHAT_BASE_URL, OAICHAT_API_KEY, OAICHAT_MODEL env vars.
Defaults live in eval/config.yaml, CLI args override.
"""

import json
import os
import asyncio
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from exa_py import Exa

from eval.tools import CODE_SCHEMA, BASH_SCHEMA, execute_code, execute_bash, MAX_TURNS, SYSTEM_PROMPT
from eval.log import log

load_dotenv()

_exa = Exa(os.environ.get("EXA_API_KEY"))

_SEARCH_SCHEMA = {
    "name": "search",
    "description": "Search the web for current information. Returns titles, URLs, and highlights.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
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


_base_url = os.environ.get("OAICHAT_BASE_URL")
_api_key = os.environ.get("OAICHAT_API_KEY")
_model = os.environ.get("OAICHAT_MODEL")
assert _base_url and _api_key and _model, "OAICHAT_BASE_URL, OAICHAT_API_KEY, OAICHAT_MODEL must be set (via config.yaml + run.py)"

_client = OpenAI(base_url=_base_url, api_key=_api_key)
_async_client = AsyncOpenAI(base_url=_base_url, api_key=_api_key)

LLM_ID = f"oaichat-{_model}"

_code_tool = {"type": "function", "function": CODE_SCHEMA}
_bash_tool = {"type": "function", "function": BASH_SCHEMA}
_search_tool = {"type": "function", "function": _SEARCH_SCHEMA}


def _extract_thinking(choice: dict) -> str | None:
    """Extract thinking/reasoning content from response if present."""
    msg = choice.get("message", {})
    for key in ("reasoning_content", "thinking", "reasoning"):
        if msg.get(key):
            return msg[key]
    return None


def _build_tools(config: dict) -> list[dict]:
    tools = []
    if config.get("enable_code"):
        tools.append(_code_tool)
    if config.get("enable_bash"):
        tools.append(_bash_tool)
    if config.get("enable_search"):
        tools.append(_search_tool)
    return tools


def _accumulate_usage(usage_total: dict[str, int | float], resp_usage: dict) -> None:
    for k in list(usage_total):
        usage_total[k] += resp_usage.get(k) or 0
    for k, v in resp_usage.items():
        if k not in usage_total and isinstance(v, (int, float)):
            usage_total.setdefault(k, 0)
            usage_total[k] += v


def _final_payload(text: str, model: str, usage_total: dict, turns: list[dict], turn_idx: int) -> dict:
    return {
        "text": text,
        "metadata": {
            "model": model,
            "usage": usage_total,
            "turns": turns,
            "total_turns": turn_idx + 1,
        },
    }


def generate(task: str, references: dict, config: dict) -> dict:
    from eval.core import build_prompt
    import os
    _odir = config.get("_output_dir")
    _tid = config.get("_task_id")
    _ofile = os.path.join(_odir, _tid) if _odir and _tid else _odir
    prompt = build_prompt(task, references, output_file=_ofile)
    task_repl = config.get("_repl")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

    tools = _build_tools(config)

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    turns = []

    for turn_idx in range(MAX_TURNS):
        log.info(f"    [oaichat] turn {turn_idx}")
        kwargs = {"model": _model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = _client.chat.completions.create(**kwargs)
        raw = resp.model_dump()
        choice = raw["choices"][0]
        msg = choice["message"]

        # Accumulate usage
        resp_usage = raw.get("usage") or {}
        _accumulate_usage(usage_total, resp_usage)

        # Capture turn data
        turn_data = {
            "turn": turn_idx,
            "finish_reason": choice.get("finish_reason"),
            "thinking": _extract_thinking(choice),
            "usage": resp_usage or None,
        }

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            log.info(f" -> done (finish={choice.get('finish_reason')})")
            turn_data["content"] = msg.get("content", "")
            turns.append(turn_data)

            return _final_payload(
                msg.get("content", ""),
                raw.get("model", _model),
                usage_total,
                turns,
                turn_idx,
            )

        # Append assistant message
        messages.append(msg)

        # Execute tools
        tool_results = []
        tool_names = [tc["function"]["name"] for tc in tool_calls]
        log.info(f" -> tools: {tool_names}")
        for tc in tool_calls:
            fn = tc["function"]
            args = json.loads(fn["arguments"])
            if fn["name"] == "execute_code":
                result = execute_code(args.get("code", ""), repl_instance=task_repl)
            elif fn["name"] == "bash":
                result = execute_bash(args.get("command", ""))
            elif fn["name"] == "search":
                result = _exa_search(args.get("query", ""))
            else:
                result = f"[Unknown tool: {fn['name']}]"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            tool_results.append({"tool": fn["name"], "args": fn["arguments"], "result_len": len(result)})

        turn_data["tool_calls"] = tool_results
        turns.append(turn_data)

    # Exhausted turns
    last_content = msg.get("content", "") if msg else ""
    return {
        "text": last_content,
        "metadata": {
            "model": raw.get("model", _model),
            "usage": usage_total,
            "turns": turns,
            "total_turns": MAX_TURNS,
            "exhausted": True,
        },
    }


async def generate_async(task: str, references: dict, config: dict) -> dict:
    from eval.core import build_prompt
    import os
    _odir = config.get("_output_dir")
    _tid = config.get("_task_id")
    _ofile = os.path.join(_odir, _tid) if _odir and _tid else _odir
    prompt = build_prompt(task, references, output_file=_ofile)
    task_repl = config.get("_repl")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    tools = _build_tools(config)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    turns = []

    for turn_idx in range(MAX_TURNS):
        log.info(f"    [oaichat] turn {turn_idx}")
        kwargs = {"model": _model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = await _async_client.chat.completions.create(**kwargs)
        raw = resp.model_dump()
        choice = raw["choices"][0]
        msg = choice["message"]

        resp_usage = raw.get("usage") or {}
        _accumulate_usage(usage_total, resp_usage)

        turn_data = {
            "turn": turn_idx,
            "finish_reason": choice.get("finish_reason"),
            "thinking": _extract_thinking(choice),
            "usage": resp_usage or None,
        }

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            log.info(f" -> done (finish={choice.get('finish_reason')})")
            turn_data["content"] = msg.get("content", "")
            turns.append(turn_data)
            return _final_payload(
                msg.get("content", ""),
                raw.get("model", _model),
                usage_total,
                turns,
                turn_idx,
            )

        messages.append(msg)

        tool_results = []
        tool_names = [tc["function"]["name"] for tc in tool_calls]
        log.info(f" -> tools: {tool_names}")
        for tc in tool_calls:
            fn = tc["function"]
            args = json.loads(fn["arguments"])
            if fn["name"] == "execute_code":
                result = await asyncio.to_thread(
                    execute_code, args.get("code", ""), repl_instance=task_repl
                )
            elif fn["name"] == "bash":
                result = await asyncio.to_thread(execute_bash, args.get("command", ""))
            elif fn["name"] == "search":
                result = await asyncio.to_thread(_exa_search, args.get("query", ""))
            else:
                result = f"[Unknown tool: {fn['name']}]"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            tool_results.append({"tool": fn["name"], "args": fn["arguments"], "result_len": len(result)})

        turn_data["tool_calls"] = tool_results
        turns.append(turn_data)

    last_content = msg.get("content", "") if msg else ""
    return {
        "text": last_content,
        "metadata": {
            "model": raw.get("model", _model),
            "usage": usage_total,
            "turns": turns,
            "total_turns": MAX_TURNS,
            "exhausted": True,
        },
    }


def generate_batch(prompts: list[str], configs: list[dict]) -> list[dict]:
    return [generate(p, {}, c) for p, c in zip(prompts, configs)]
