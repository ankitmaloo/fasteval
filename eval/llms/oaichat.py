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

_extra_body_raw = os.environ.get("OAICHAT_EXTRA_BODY")
_extra_body = json.loads(_extra_body_raw) if _extra_body_raw else None

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


_KNOWN_TOOLS = {"execute_code", "bash", "search"}

import re

_TOOL_CALL_TAG_RE = re.compile(r"</?tool_call>|</?arg_key>|</?arg_value>")


def _strip_tool_tags(text: str) -> str:
    """Strip spurious <tool_call>/<arg_key>/<arg_value> tags from content text."""
    return _TOOL_CALL_TAG_RE.sub("", text).strip()


def _parse_tool_call(fn: dict) -> tuple[str, dict, bool]:
    """Parse a tool call, handling models that mangle the function name/args.

    Returns (canonical_name, args_dict, is_malformed).
    """
    raw_name = fn["name"]
    raw_args = fn["arguments"]

    # Try normal parse first
    try:
        args = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        args = {}

    # Determine canonical tool name
    canonical = raw_name
    for name in _KNOWN_TOOLS:
        if raw_name == name or raw_name.startswith(name + "<") or raw_name.startswith(name + "</"):
            canonical = name
            break

    # Detect malformed: name contains XML-like tags, or args are empty when they shouldn't be
    malformed = canonical != raw_name or "<arg" in raw_name or "</arg" in raw_name
    if canonical in ("execute_code", "bash") and not args:
        malformed = True

    return canonical, args, malformed


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


def _append_conv(config: dict, entry: dict) -> None:
    store = config.get("_conv_store")
    tid = config.get("_task_id", "")
    if store and tid:
        store.append(tid, entry)


def _final_payload(text: str, model: str, usage_total: dict, turns: list[dict], turn_idx: int) -> dict:
    return {
        "text": _strip_tool_tags(text),
        "metadata": {
            "model": model,
            "usage": usage_total,
            "turns": turns,
            "total_turns": turn_idx + 1,
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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

    tools = _build_tools(config)


    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    turns = []

    _append_conv(config, {"turn": 0, "role": "system", "content": SYSTEM_PROMPT})
    _append_conv(config, {"turn": 0, "role": "user", "content": prompt})

    for turn_idx in range(MAX_TURNS):
        log.info(f"    [oaichat] turn {turn_idx}")
        kwargs = {"model": _model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if _extra_body:
            kwargs["extra_body"] = _extra_body
        resp = _client.chat.completions.create(**kwargs)
        raw = resp.model_dump()
        choice = raw["choices"][0]
        msg = choice["message"]

        resp_usage = raw.get("usage") or {}
        _accumulate_usage(usage_total, resp_usage)

        thinking = _extract_thinking(choice)
        turn_data = {
            "turn": turn_idx,
            "finish_reason": choice.get("finish_reason"),
            "thinking": thinking,
            "usage": resp_usage or None,
        }

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            log.info(f" -> done (finish={choice.get('finish_reason')})")
            turn_data["content"] = msg.get("content") or ""
            turns.append(turn_data)
            _append_conv(config, {"turn": turn_idx, "role": "assistant", "content": msg.get("content") or "", "thinking": thinking})

            return _final_payload(
                msg.get("content") or "",
                raw.get("model", _model),
                usage_total,
                turns,
                turn_idx,
            )

        # Append assistant message
        messages.append(msg)

        # Log assistant tool calls
        tc_log = [{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"], "id": tc["id"]} for tc in tool_calls]
        _append_conv(config, {"turn": turn_idx, "role": "assistant", "tool_calls": tc_log, "thinking": thinking})

        # Execute tools
        tool_results = []
        for tc in tool_calls:
            fn = tc["function"]
            name, args, malformed = _parse_tool_call(fn)
            log.info(f" -> tool: {name} (malformed={malformed})")
            if malformed:
                result = (
                    "[ERROR] Malformed tool call. Your function call was not properly formatted. "
                    "Please retry using the standard JSON arguments format. "
                    f"Expected tool '{name}' with proper JSON arguments."
                )
            elif name == "execute_code":
                result = execute_code(args.get("code", ""), repl_instance=task_repl)
            elif name == "bash":
                result = execute_bash(args.get("command", ""), cwd=_odir)
            elif name == "search":
                result = _exa_search(args.get("query", ""))
            else:
                result = f"[Unknown tool: {fn['name']}]"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            tool_results.append({"tool": name, "args": fn["arguments"], "result_len": len(result), "malformed": malformed})
            _append_conv(config, {"turn": turn_idx, "role": "tool", "name": name, "call_id": tc["id"], "result": result, "malformed": malformed})

        turn_data["tool_calls"] = tool_results
        turns.append(turn_data)

    # Exhausted turns
    last_content = (msg.get("content") or "") if msg else ""
    _append_conv(config, {"turn": MAX_TURNS, "role": "assistant", "content": last_content, "exhausted": True})
    result = _final_payload(last_content, raw.get("model", _model), usage_total, turns, MAX_TURNS - 1)
    result["metadata"]["exhausted"] = True
    return result


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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    tools = _build_tools(config)

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    turns = []

    _append_conv(config, {"turn": 0, "role": "system", "content": SYSTEM_PROMPT})
    _append_conv(config, {"turn": 0, "role": "user", "content": prompt})

    for turn_idx in range(MAX_TURNS):
        log.info(f"    [oaichat] turn {turn_idx}")
        kwargs = {"model": _model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if _extra_body:
            kwargs["extra_body"] = _extra_body
        resp = await _async_client.chat.completions.create(**kwargs)
        raw = resp.model_dump()
        choice = raw["choices"][0]
        msg = choice["message"]

        resp_usage = raw.get("usage") or {}
        _accumulate_usage(usage_total, resp_usage)

        thinking = _extract_thinking(choice)
        turn_data = {
            "turn": turn_idx,
            "finish_reason": choice.get("finish_reason"),
            "thinking": thinking,
            "usage": resp_usage or None,
        }

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            log.info(f" -> done (finish={choice.get('finish_reason')})")
            turn_data["content"] = msg.get("content") or ""
            turns.append(turn_data)
            _append_conv(config, {"turn": turn_idx, "role": "assistant", "content": msg.get("content") or "", "thinking": thinking})

            return _final_payload(
                msg.get("content") or "",
                raw.get("model", _model),
                usage_total,
                turns,
                turn_idx,
            )

        messages.append(msg)

        tc_log = [{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"], "id": tc["id"]} for tc in tool_calls]
        _append_conv(config, {"turn": turn_idx, "role": "assistant", "tool_calls": tc_log, "thinking": thinking})

        tool_results = []
        for tc in tool_calls:
            fn = tc["function"]
            name, args, malformed = _parse_tool_call(fn)
            log.info(f" -> tool: {name} (malformed={malformed})")
            if malformed:
                result = (
                    "[ERROR] Malformed tool call. Your function call was not properly formatted. "
                    "Please retry using the standard JSON arguments format. "
                    f"Expected tool '{name}' with proper JSON arguments."
                )
            elif name == "execute_code":
                result = await asyncio.to_thread(
                    execute_code, args.get("code", ""), repl_instance=task_repl
                )
            elif name == "bash":
                result = await asyncio.to_thread(execute_bash, args.get("command", ""), cwd=_odir)
            elif name == "search":
                result = await asyncio.to_thread(_exa_search, args.get("query", ""))
            else:
                result = f"[Unknown tool: {fn['name']}]"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            tool_results.append({"tool": name, "args": fn["arguments"], "result_len": len(result), "malformed": malformed})
            _append_conv(config, {"turn": turn_idx, "role": "tool", "name": name, "call_id": tc["id"], "result": result, "malformed": malformed})

        turn_data["tool_calls"] = tool_results
        turns.append(turn_data)

    last_content = (msg.get("content") or "") if msg else ""
    _append_conv(config, {"turn": MAX_TURNS, "role": "assistant", "content": last_content, "exhausted": True})
    result = _final_payload(last_content, raw.get("model", _model), usage_total, turns, MAX_TURNS - 1)
    result["metadata"]["exhausted"] = True
    return result


def generate_batch(prompts: list[str], configs: list[dict]) -> list[dict]:
    return [generate(p, {}, c) for p, c in zip(prompts, configs)]
