"""Subject LLM: GPT-5.2 with code execution + search via Responses API."""

import json
import os
import asyncio
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from eval.tools import CODE_SCHEMA, BASH_SCHEMA, execute_code, execute_bash, MAX_TURNS, SYSTEM_PROMPT
from eval.log import log

load_dotenv()

LLM_ID = "gpt-5.2"

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
_async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

_code_tool = {"type": "function", **CODE_SCHEMA}
_bash_tool = {"type": "function", **BASH_SCHEMA}


def _extract_reasoning(resp) -> str | None:
    try:
        raw = resp.model_dump()
    except Exception:
        return None
    items = raw.get("output")
    if not isinstance(items, list):
        return None

    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if isinstance(summary, list):
            for block in summary:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
        elif isinstance(summary, str) and summary.strip():
            parts.append(summary)
    joined = "\n".join(parts).strip()
    return joined or None


def _build_tools(config: dict) -> list[dict]:
    tools = []
    if config.get("enable_code"):
        tools.append(_code_tool)
    if config.get("enable_bash"):
        tools.append(_bash_tool)
    if config.get("enable_search"):
        tools.append({"type": "web_search"})
    return tools


def _payload(final_text: str, turns: list[dict], usage_total: dict[str, int]) -> dict:
    return {
        "text": final_text,
        "metadata": {
            "model": LLM_ID,
            "usage": usage_total,
            "total_turns": len(turns),
            "turns": turns,
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
    tools = _build_tools(config)

    input_items = [{"role": "user", "content": prompt}]
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    turns = []
    final_text = ""

    for turn in range(MAX_TURNS):
        kwargs = {"model": LLM_ID, "input": input_items, "instructions": SYSTEM_PROMPT, "reasoning": {"effort": "medium"}}
        if tools:
            kwargs["tools"] = tools
        resp = _client.responses.create(**kwargs)
        usage = resp.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
        usage_total["input_tokens"] += input_tokens
        usage_total["output_tokens"] += output_tokens
        usage_total["total_tokens"] += total_tokens
        turn_info = {
            "turn": turn + 1,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }
        reasoning = _extract_reasoning(resp)
        if reasoning:
            turn_info["thinking"] = reasoning
        log.info(f"    [OPENAI] turn {turn + 1}/{MAX_TURNS} {usage.input_tokens}in/{usage.output_tokens}out")

        fn_calls = [item for item in resp.output if item.type == "function_call"]
        if not fn_calls:
            final_text = resp.output_text or ""
            turn_info["content"] = final_text
            turns.append(turn_info)
            break

        turns.append(turn_info)

        input_items.extend(resp.output)

        for fc in fn_calls:
            args = json.loads(fc.arguments)
            if fc.name == "execute_code":
                code = args.get("code", "")
                log.info(f"    [CODE]\n{code}")
                result = execute_code(code, repl_instance=task_repl)
                log.info(f"    [CODE OUT] ({len(result)} chars)\n{result}")
            elif fc.name == "bash":
                cmd = args.get("command", "")
                log.info(f"    [BASH]\n{cmd}")
                result = execute_bash(cmd)
                log.info(f"    [BASH OUT] ({len(result)} chars)\n{result}")
            else:
                result = f"[Unknown tool: {fc.name}]"
            input_items.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": result,
            })

    if not final_text:
        final_text = resp.output_text or ""

    return _payload(final_text, turns, usage_total)


async def generate_async(task: str, references: dict, config: dict) -> dict:
    from eval.core import build_prompt
    import os
    _odir = config.get("_output_dir")
    _tid = config.get("_task_id")
    _ofile = os.path.join(_odir, _tid) if _odir and _tid else _odir
    prompt = build_prompt(task, references, output_file=_ofile)
    task_repl = config.get("_repl")
    tools = _build_tools(config)

    input_items = [{"role": "user", "content": prompt}]
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    turns = []
    final_text = ""

    for turn in range(MAX_TURNS):
        kwargs = {"model": LLM_ID, "input": input_items, "instructions": SYSTEM_PROMPT, "reasoning": {"effort": "medium"}}
        if tools:
            kwargs["tools"] = tools
        resp = await _async_client.responses.create(**kwargs)
        usage = resp.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
        usage_total["input_tokens"] += input_tokens
        usage_total["output_tokens"] += output_tokens
        usage_total["total_tokens"] += total_tokens
        turn_info = {
            "turn": turn + 1,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }
        reasoning = _extract_reasoning(resp)
        if reasoning:
            turn_info["thinking"] = reasoning
        log.info(f"    [OPENAI] turn {turn + 1}/{MAX_TURNS} {usage.input_tokens}in/{usage.output_tokens}out")

        fn_calls = [item for item in resp.output if item.type == "function_call"]
        if not fn_calls:
            final_text = resp.output_text or ""
            turn_info["content"] = final_text
            turns.append(turn_info)
            break

        turns.append(turn_info)
        input_items.extend(resp.output)

        for fc in fn_calls:
            args = json.loads(fc.arguments)
            if fc.name == "execute_code":
                code = args.get("code", "")
                log.info(f"    [CODE]\n{code}")
                result = await asyncio.to_thread(execute_code, code, repl_instance=task_repl)
                log.info(f"    [CODE OUT] ({len(result)} chars)\n{result}")
            elif fc.name == "bash":
                cmd = args.get("command", "")
                log.info(f"    [BASH]\n{cmd}")
                result = await asyncio.to_thread(execute_bash, cmd)
                log.info(f"    [BASH OUT] ({len(result)} chars)\n{result}")
            else:
                result = f"[Unknown tool: {fc.name}]"
            input_items.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": result,
            })

    if not final_text:
        final_text = resp.output_text or ""
    return _payload(final_text, turns, usage_total)
