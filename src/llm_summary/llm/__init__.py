"""LLM backend implementations."""

from .base import LLMBackend
from .claude import ClaudeBackend
from .gemini import GeminiBackend
from .llamacpp import LlamaCppBackend
from .ollama import OllamaBackend
from .openai import OpenAIBackend

__all__ = [
    "LLMBackend",
    "ClaudeBackend",
    "GeminiBackend",
    "OpenAIBackend",
    "OllamaBackend",
    "LlamaCppBackend",
]


def create_backend(
    backend_type: str,
    model: str | None = None,
    **kwargs,
) -> LLMBackend:
    """
    Create an LLM backend instance.

    Args:
        backend_type: One of "claude", "openai", "ollama", "llamacpp", "gemini"
        model: Model name (uses default if not specified)
        **kwargs: Additional backend-specific arguments

    Returns:
        An LLMBackend instance
    """
    if backend_type == "claude":
        return ClaudeBackend(model=model, **kwargs)
    elif backend_type == "gemini":
        return GeminiBackend(model=model, **kwargs)
    elif backend_type == "openai":
        return OpenAIBackend(model=model, **kwargs)
    elif backend_type == "ollama":
        return OllamaBackend(model=model, **kwargs)
    elif backend_type == "llamacpp":
        return LlamaCppBackend(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
