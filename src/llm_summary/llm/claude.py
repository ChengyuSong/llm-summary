"""Anthropic Claude backend."""

import os

from .base import LLMBackend, LLMResponse


class ClaudeBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
    ):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self._client = None

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-20250514"

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )

            if not self.api_key:
                raise ValueError(
                    "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key."
                )

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion using Claude."""
        response = self.complete_with_metadata(prompt, system)
        return response.content

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
