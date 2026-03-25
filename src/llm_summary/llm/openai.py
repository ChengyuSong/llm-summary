"""OpenAI-compatible backend (GPT, vLLM, etc.)."""

import os
import sys
from typing import Any

from .base import LLMBackend, LLMResponse


class OpenAIBackend(LLMBackend):
    """OpenAI API backend (also works with vLLM and other compatible servers)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 16384,
        base_url: str | None = None,
        enable_thinking: bool = True,
    ):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.enable_thinking = enable_thinking
        self._client: Any = None

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    @property
    def client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                ) from None

            if not self.api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY or pass api_key."
                )

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> str:
        """Generate a completion using OpenAI."""
        response = self.complete_with_metadata(
            prompt, system, response_format=response_format,
        )
        return response.content

    def complete_with_metadata(
        self, prompt: str, system: str | None = None, cache_system: bool = False,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Generate a completion with metadata."""
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if not self.enable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""

        if choice.finish_reason == "length":
            print(
                f"WARNING: OpenAI response may be incomplete "
                f"(finish_reason=length). "
                f"Consider increasing max_tokens (current: {self.max_tokens}).",
                file=sys.stderr,
            )

        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=False,
        )

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> Any:
        """Generate a completion with tool use support.

        Accepts Anthropic-format tool definitions and messages, converts to
        OpenAI format, and returns an adapter that mimics the Anthropic
        response structure.
        """
        oai_messages: list[dict[str, Any]] = []

        if system:
            oai_messages.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                oai_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    # May contain text and tool_use blocks
                    text_parts = []
                    tool_calls = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "tool_use":
                                import json
                                tool_calls.append({
                                    "id": item["id"],
                                    "type": "function",
                                    "function": {
                                        "name": item["name"],
                                        "arguments": json.dumps(item.get("input", {})),
                                    },
                                })
                    oai_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else None,
                    }
                    if tool_calls:
                        oai_msg["tool_calls"] = tool_calls
                    oai_messages.append(oai_msg)
                elif role == "user":
                    # May contain tool_result blocks
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            oai_messages.append({
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item.get("content", ""),
                            })
                        elif isinstance(item, dict) and item.get("type") == "text":
                            oai_messages.append({"role": "user", "content": item["text"]})
                        elif isinstance(item, str):
                            oai_messages.append({"role": "user", "content": item})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "max_tokens": self.max_tokens,
        }

        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in tools
            ]
        if not self.enable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        response = self.client.chat.completions.create(**kwargs)

        if os.environ.get("OPENAI_DEBUG"):
            print("[OPENAI DEBUG] Tool use response:")
            print(response.model_dump_json(indent=2))

        return _OpenAIToolResponse(response)


class _OpenAIToolResponse:
    """Adapter to make OpenAI responses look like Anthropic responses."""

    def __init__(self, response: Any):
        self._response = response
        self.content: list[Any] = []
        self.stop_reason = "end_turn"

        if not response.choices:
            return

        choice = response.choices[0]
        message = choice.message

        if message.content:
            self.content.append(_TextBlock(text=message.content))

        if message.tool_calls:
            import json
            self.stop_reason = "tool_use"
            for tc in message.tool_calls:
                args = tc.function.arguments
                self.content.append(
                    _ToolUseBlock(
                        id=tc.id,
                        name=tc.function.name,
                        input=json.loads(args) if args else {},
                    )
                )


class _TextBlock:
    """Mimics Anthropic TextBlock."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    """Mimics Anthropic ToolUseBlock."""

    def __init__(self, id: str, name: str, input: dict):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input
