"""
LLM Client with unified interface for OpenAI, Ollama, and OpenAI-compatible APIs.
Supports streaming, tool/function calling, and structured JSON output.
"""

import json
import os
from typing import Any

import yaml
from openai import OpenAI


class LLMClient:
    """Unified LLM client supporting multiple backends via OpenAI-compatible API."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        llm_cfg = cfg["llm"]
        api_key = os.environ.get("OPENAI_API_KEY", llm_cfg.get("api_key", ""))
        base_url = llm_cfg.get("base_url", "https://api.openai.com/v1")

        self.model = llm_cfg.get("model", "gpt-4o")
        self.temperature = llm_cfg.get("temperature", 0.7)
        self.max_tokens = llm_cfg.get("max_tokens", 4096)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Send a chat completion request and return the parsed response."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        result = {
            "role": "assistant",
            "content": choice.message.content,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

        if choice.message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in choice.message.tool_calls
            ]

        return result

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_handlers: dict[str, callable],
        max_rounds: int = 5,
    ) -> dict[str, Any]:
        """Chat loop that automatically handles tool calling back-and-forth."""
        history = list(messages)
        tool_call_log = []

        for _ in range(max_rounds):
            response = self.chat(messages=history, tools=tools)

            if response["finish_reason"] == "stop":
                return {"answer": response["content"], "tool_calls": tool_call_log, "history": history}

            if "tool_calls" not in response:
                return {"answer": response["content"], "tool_calls": tool_call_log, "history": history}

            history.append({"role": "assistant", "content": response.get("content"),
                            "tool_calls": [{"id": tc["id"], "type": "function",
                                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}}
                                           for tc in response["tool_calls"]]})

            for tc in response["tool_calls"]:
                handler = tool_handlers.get(tc["name"])
                if handler:
                    result = handler(**tc["arguments"])
                    tool_call_log.append({"tool": tc["name"], "args": tc["arguments"], "result": str(result)[:500]})
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })

        return {"answer": response.get("content"), "tool_calls": tool_call_log, "history": history}
