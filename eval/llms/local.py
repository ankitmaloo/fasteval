"""Subject LLM: Local model via OpenAI-compatible endpoint with bash tool use.

Config via env vars:
  LOCAL_MODEL_URL  - endpoint URL (default: http://localhost:8000/v1/chat/completions)
  LOCAL_MODEL_NAME - model name for the request (default: default)
"""

import json
import os
import requests
from dotenv import load_dotenv

from eval.tools import BASH_SCHEMA, execute_bash, MAX_TURNS, SYSTEM_PROMPT, max_turns_for_config

load_dotenv()

_url = os.environ.get("LOCAL_MODEL_URL", "http://localhost:8000/v1/chat/completions")
_model = os.environ.get("LOCAL_MODEL_NAME", "default")

LLM_ID = f"local-{_model}"

_tools = [{"type": "function", "function": BASH_SCHEMA}]


def generate(task: str, references: dict[str, str], config: dict) -> str:
    parts = [task]
    if references:
        parts.append("\n\n--- REFERENCE FILES ---")
        for name, content in references.items():
            parts.append(f"\n### {name}\n{content}")
    prompt = "\n".join(parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    max_turns = max_turns_for_config(config)

    for _ in range(max_turns):
        resp = requests.post(_url, json={
            "model": _model,
            "messages": messages,
            "tools": _tools,
            "tool_choice": "auto",
        })
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg.get("content", "")

        # Append assistant message
        messages.append(msg)

        # Execute tools
        for tc in tool_calls:
            fn = tc["function"]
            if fn["name"] == "bash":
                args = json.loads(fn["arguments"])
                result = execute_bash(args.get("command", ""))
            else:
                result = f"[Unknown tool: {fn['name']}]"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    return msg.get("content", "")


def generate_batch(prompts: list[str], configs: list[dict]) -> list[str]:
    """Batch mode — runs each prompt through the agentic loop."""
    return [generate(p, {}, c) for p, c in zip(prompts, configs)]
