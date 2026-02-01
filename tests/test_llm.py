"""Tests for LLM backends."""

import pytest
from unittest.mock import Mock, patch

from llm_summary.llm.base import LLMBackend, LLMResponse
from llm_summary.llm import create_backend


class MockBackend(LLMBackend):
    """Mock LLM backend for testing."""

    def __init__(self, model=None, response="Mock response"):
        super().__init__(model)
        self._response = response

    @property
    def default_model(self) -> str:
        return "mock-model"

    def complete(self, prompt: str, system: str | None = None) -> str:
        return self._response

    def complete_with_metadata(self, prompt: str, system: str | None = None) -> LLMResponse:
        return LLMResponse(
            content=self._response,
            model=self.model,
            input_tokens=len(prompt),
            output_tokens=len(self._response),
        )


class TestLLMBackend:
    """Tests for LLMBackend abstract class."""

    def test_mock_backend(self):
        """Test mock backend works."""
        backend = MockBackend()

        assert backend.model == "mock-model"
        assert backend.complete("test") == "Mock response"

    def test_custom_response(self):
        """Test mock with custom response."""
        backend = MockBackend(response="Custom response")
        assert backend.complete("test") == "Custom response"

    def test_with_metadata(self):
        """Test complete_with_metadata."""
        backend = MockBackend()
        response = backend.complete_with_metadata("test prompt")

        assert response.content == "Mock response"
        assert response.model == "mock-model"
        assert response.input_tokens == len("test prompt")


class TestCreateBackend:
    """Tests for create_backend factory function."""

    def test_invalid_backend(self):
        """Test creating invalid backend type."""
        with pytest.raises(ValueError):
            create_backend("invalid_backend")


class TestClaudeBackend:
    """Tests for Claude backend (mocked)."""

    def test_default_model(self):
        """Test default model name."""
        from llm_summary.llm.claude import ClaudeBackend

        backend = ClaudeBackend(api_key="test-key")
        assert "claude" in backend.default_model.lower()

    def test_missing_api_key(self):
        """Test error when API key is missing."""
        from llm_summary.llm.claude import ClaudeBackend

        # Clear environment variable temporarily
        with patch.dict("os.environ", {}, clear=True):
            backend = ClaudeBackend()
            with pytest.raises(ValueError):
                _ = backend.client


class TestOpenAIBackend:
    """Tests for OpenAI backend (mocked)."""

    def test_default_model(self):
        """Test default model name."""
        from llm_summary.llm.openai import OpenAIBackend

        backend = OpenAIBackend(api_key="test-key")
        assert "gpt" in backend.default_model.lower()

    def test_custom_base_url(self):
        """Test setting custom base URL."""
        from llm_summary.llm.openai import OpenAIBackend

        backend = OpenAIBackend(api_key="test", base_url="http://localhost:8080")
        assert backend.base_url == "http://localhost:8080"


class TestOllamaBackend:
    """Tests for Ollama backend."""

    def test_default_model(self):
        """Test default model name."""
        from llm_summary.llm.ollama import OllamaBackend

        backend = OllamaBackend()
        assert backend.default_model is not None

    def test_custom_base_url(self):
        """Test setting custom base URL."""
        from llm_summary.llm.ollama import OllamaBackend

        backend = OllamaBackend(base_url="http://localhost:12345")
        assert backend.base_url == "http://localhost:12345"

    def test_is_available_when_offline(self):
        """Test is_available when Ollama is not running."""
        from llm_summary.llm.ollama import OllamaBackend

        backend = OllamaBackend(base_url="http://localhost:99999")
        assert backend.is_available() is False
