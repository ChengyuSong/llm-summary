"""GCP Vertex AI backend for Claude models."""

import os
from typing import Any

from .base import LLMBackend, LLMResponse


class VertexAIBackend(LLMBackend):
    """GCP Vertex AI backend supporting Claude models."""

    def __init__(
        self,
        model: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        max_tokens: int = 4096,
    ):
        super().__init__(model)
        # Check multiple environment variables in order of preference
        self.project_id = (
            project_id
            or os.environ.get("VERTEX_AI_PROJECT")
            or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("CLOUD_ML_PROJECT_ID")
        )
        self.location = location or os.environ.get("VERTEX_AI_LOCATION", "global")
        self.max_tokens = max_tokens
        self._client = None

    @property
    def default_model(self) -> str:
        return "claude-haiku-4-5@20251001"

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import AnthropicVertex
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic[vertex]"
                )

            if not self.project_id:
                raise ValueError(
                    "GCP project ID required. Set VERTEX_AI_PROJECT or pass project_id."
                )

            self._client = AnthropicVertex(
                project_id=self.project_id,
                region=self.location,
            )
        return self._client

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion using Claude via Vertex AI."""
        response = self.complete_with_metadata(prompt, system)
        return response.content

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> Any:
        """
        Generate a completion with tool use support.

        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool definitions
            system: Optional system message

        Returns:
            Raw API response object with tool_use blocks if applicable
        """
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        if os.environ.get("VERTEX_DEBUG"):
            print("[VERTEX DEBUG] Tool use response:")
            print(response.model_dump_json(indent=2))

        return response

    def complete_with_metadata(
        self, prompt: str, system: str | None = None
    ) -> LLMResponse:
        """Generate a completion with metadata."""
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        # Debug: dump full response if VERTEX_DEBUG env var is set
        if os.environ.get("VERTEX_DEBUG"):
            print("[VERTEX DEBUG] Full response:")
            print(response.model_dump_json(indent=2))

        # Extract content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        # Extract token usage
        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)

        # Check for cache hit
        cached = False
        if hasattr(response.usage, "cache_read_input_tokens"):
            cached = response.usage.cache_read_input_tokens > 0

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=cached,
        )
