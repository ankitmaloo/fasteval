"""Subject LLM: Gemini 3 Flash Preview with code execution + search."""

import asyncio
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from eval.tools import CODE_SCHEMA, BASH_SCHEMA, execute_code, execute_bash, MAX_TURNS, SYSTEM_PROMPT
from eval.log import log

load_dotenv()

LLM_ID = "gemini-3-flash-preview"

_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

_code_decl = types.FunctionDeclaration(
    name=CODE_SCHEMA["name"],
    description=CODE_SCHEMA["description"],
    parameters=types.Schema(
        type="OBJECT",
        properties={"code": types.Schema(type="STRING", description="Python code to execute")},
        required=["code"],
    ),
)

_bash_decl = types.FunctionDeclaration(
    name=BASH_SCHEMA["name"],
    description=BASH_SCHEMA["description"],
    parameters=types.Schema(
        type="OBJECT",
        properties={"command": types.Schema(type="STRING", description="The bash command to execute")},
        required=["command"],
    ),
)

_search_decl = types.FunctionDeclaration(
    name="search",
    description="Search the web for information. Returns relevant search results.",
    parameters=types.Schema(
        type="OBJECT",
        properties={"query": types.Schema(type="STRING", description="The search query")},
        required=["query"],
    ),
)


def _usage_counts(usage) -> dict[str, int]:
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
    }


def _build_tools(config: dict) -> list[types.Tool]:
    decls = []
    if config.get("enable_code"):
        decls.append(_code_decl)
    if config.get("enable_bash"):
        decls.append(_bash_decl)
    if config.get("enable_search"):
        decls.append(_search_decl)
    return [types.Tool(function_declarations=decls)] if decls else []


def _response_payload(text: str, usage_total: dict[str, int], turns: list[dict]) -> dict:
    return {
        "text": text,
        "metadata": {
            "model": LLM_ID,
            "usage": usage_total,
            "turns": turns,
            "total_turns": len(turns),
        },
    }


def _search_subagent(query: str) -> str:
    """Run a separate Gemini call with Google Search grounding."""
    resp = _client.models.generate_content(
        model=LLM_ID,
        contents=f"Search and summarize: {query}",
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    return resp.text or ""


async def _search_subagent_async(query: str) -> str:
    """Run a separate Gemini call with Google Search grounding (async)."""
    resp = await _client.aio.models.generate_content(
        model=LLM_ID,
        contents=f"Search and summarize: {query}",
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    return resp.text or ""


def generate(task: str, references: dict, config: dict) -> dict:
    from eval.core import build_prompt
    import os
    _odir = config.get("_output_dir")
    _tid = config.get("_task_id")
    _ofile = os.path.join(_odir, _tid) if _odir and _tid else _odir
    prompt = build_prompt(task, references, output_file=_ofile)
    task_repl = config.get("_repl")
    tools = _build_tools(config)

    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    turns: list[dict] = []

    for turn in range(MAX_TURNS):
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
        )
        if tools:
            cfg.tools = tools
        resp = _client.models.generate_content(model=LLM_ID, contents=contents, config=cfg)
        candidate = resp.candidates[0]
        usage = resp.usage_metadata
        usage_counts = _usage_counts(usage)
        usage_total["input_tokens"] += usage_counts["input_tokens"]
        usage_total["output_tokens"] += usage_counts["output_tokens"]
        usage_total["total_tokens"] += usage_counts["total_tokens"]
        turns.append({"turn": turn + 1, "usage": usage_counts})
        log.info(f"    [GEMINI] turn {turn + 1}/{MAX_TURNS} {usage.prompt_token_count}in/{usage.candidates_token_count}out")

        fn_calls = [p for p in candidate.content.parts if p.function_call]
        if not fn_calls:
            return _response_payload(resp.text or "", usage_total, turns)

        contents.append(candidate.content)

        fn_parts = []
        for part in fn_calls:
            fc = part.function_call
            if fc.name == "execute_code":
                code = fc.args.get("code", "")
                log.info(f"    [CODE]\n{code}")
                result = execute_code(code, repl_instance=task_repl)
                log.info(f"    [CODE OUT] ({len(result)} chars)\n{result}")
            elif fc.name == "bash":
                cmd = fc.args.get("command", "")
                log.info(f"    [BASH]\n{cmd}")
                result = execute_bash(cmd)
                log.info(f"    [BASH OUT] ({len(result)} chars)\n{result}")
            elif fc.name == "search":
                query = fc.args.get("query", "")
                log.info(f"    [SEARCH] {query}")
                result = _search_subagent(query)
                log.info(f"    [SEARCH OUT] ({len(result)} chars)")
            else:
                result = f"[Unknown tool: {fc.name}]"
            fn_parts.append(types.Part(function_response=types.FunctionResponse(
                name=fc.name, response={"output": result},
            )))
        contents.append(types.Content(role="user", parts=fn_parts))

    return _response_payload(resp.text or "", usage_total, turns)


async def generate_async(task: str, references: dict, config: dict) -> dict:
    from eval.core import build_prompt
    import os
    _odir = config.get("_output_dir")
    _tid = config.get("_task_id")
    _ofile = os.path.join(_odir, _tid) if _odir and _tid else _odir
    prompt = build_prompt(task, references, output_file=_ofile)
    task_repl = config.get("_repl")
    tools = _build_tools(config)

    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    turns: list[dict] = []

    for turn in range(MAX_TURNS):
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
        )
        if tools:
            cfg.tools = tools
        resp = await _client.aio.models.generate_content(
            model=LLM_ID,
            contents=contents,
            config=cfg,
        )
        candidate = resp.candidates[0]
        usage = resp.usage_metadata
        usage_counts = _usage_counts(usage)
        usage_total["input_tokens"] += usage_counts["input_tokens"]
        usage_total["output_tokens"] += usage_counts["output_tokens"]
        usage_total["total_tokens"] += usage_counts["total_tokens"]
        turns.append({"turn": turn + 1, "usage": usage_counts})
        log.info(f"    [GEMINI] turn {turn + 1}/{MAX_TURNS} {usage.prompt_token_count}in/{usage.candidates_token_count}out")

        fn_calls = [p for p in candidate.content.parts if p.function_call]
        if not fn_calls:
            return _response_payload(resp.text or "", usage_total, turns)

        contents.append(candidate.content)

        fn_parts = []
        for part in fn_calls:
            fc = part.function_call
            if fc.name == "execute_code":
                code = fc.args.get("code", "")
                log.info(f"    [CODE]\n{code}")
                result = await asyncio.to_thread(execute_code, code, repl_instance=task_repl)
                log.info(f"    [CODE OUT] ({len(result)} chars)\n{result}")
            elif fc.name == "bash":
                cmd = fc.args.get("command", "")
                log.info(f"    [BASH]\n{cmd}")
                result = await asyncio.to_thread(execute_bash, cmd)
                log.info(f"    [BASH OUT] ({len(result)} chars)\n{result}")
            elif fc.name == "search":
                query = fc.args.get("query", "")
                log.info(f"    [SEARCH] {query}")
                result = await _search_subagent_async(query)
                log.info(f"    [SEARCH OUT] ({len(result)} chars)")
            else:
                result = f"[Unknown tool: {fc.name}]"
            fn_parts.append(types.Part(function_response=types.FunctionResponse(
                name=fc.name, response={"output": result},
            )))
        contents.append(types.Content(role="user", parts=fn_parts))

    return _response_payload(resp.text or "", usage_total, turns)
