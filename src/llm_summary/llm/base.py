"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this backend."""
        pass

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt
            system: Optional system message

        Returns:
            The completion text
        """
        pass

    @abstractmethod
    def complete_with_metadata(
        self, prompt: str, system: str | None = None
    ) -> LLMResponse:
        """
        Generate a completion with metadata.

        Args:
            prompt: The user prompt
            system: Optional system message

        Returns:
            LLMResponse with content and metadata
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
