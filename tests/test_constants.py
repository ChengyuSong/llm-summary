"""Tests for builder constants module."""

from llm_summary.builder.constants import (
    CHARS_PER_TOKEN,
    DOCKER_WORKSPACE_BUILD,
    DOCKER_WORKSPACE_SRC,
    LARGE_OUTPUT_TOKEN_THRESHOLD,
    MAX_CONTEXT_TOKENS,
    MAX_TURNS_AUTOTOOLS,
    MAX_TURNS_CMAKE,
    MAX_TURNS_ERROR_ANALYSIS,
    TIMEOUT_BUILD,
    TIMEOUT_CONFIGURE,
    TIMEOUT_LONG_BUILD,
)


class TestDockerConstants:
    """Tests for Docker workspace constants."""

    def test_workspace_src_value(self):
        """Test DOCKER_WORKSPACE_SRC has correct value."""
        assert DOCKER_WORKSPACE_SRC == "/workspace/src"

    def test_workspace_build_value(self):
        """Test DOCKER_WORKSPACE_BUILD has correct value."""
        assert DOCKER_WORKSPACE_BUILD == "/workspace/build"


class TestTimeoutConstants:
    """Tests for timeout constants."""

    def test_configure_timeout(self):
        """Test TIMEOUT_CONFIGURE value."""
        assert TIMEOUT_CONFIGURE == 300
        assert isinstance(TIMEOUT_CONFIGURE, int)

    def test_build_timeout(self):
        """Test TIMEOUT_BUILD value."""
        assert TIMEOUT_BUILD == 600
        assert isinstance(TIMEOUT_BUILD, int)

    def test_long_build_timeout(self):
        """Test TIMEOUT_LONG_BUILD value."""
        assert TIMEOUT_LONG_BUILD == 1200
        assert isinstance(TIMEOUT_LONG_BUILD, int)

    def test_timeout_ordering(self):
        """Test that timeout values are in ascending order."""
        assert TIMEOUT_CONFIGURE < TIMEOUT_BUILD < TIMEOUT_LONG_BUILD


class TestReActConstants:
    """Tests for ReAct loop parameters."""

    def test_max_turns_cmake(self):
        """Test MAX_TURNS_CMAKE value."""
        assert MAX_TURNS_CMAKE == 15
        assert isinstance(MAX_TURNS_CMAKE, int)

    def test_max_turns_autotools(self):
        """Test MAX_TURNS_AUTOTOOLS value."""
        assert MAX_TURNS_AUTOTOOLS == 20
        assert isinstance(MAX_TURNS_AUTOTOOLS, int)

    def test_max_turns_error_analysis(self):
        """Test MAX_TURNS_ERROR_ANALYSIS value."""
        assert MAX_TURNS_ERROR_ANALYSIS == 10
        assert isinstance(MAX_TURNS_ERROR_ANALYSIS, int)

    def test_max_turns_ordering(self):
        """Test that max turns for error analysis is less than build loops."""
        assert MAX_TURNS_ERROR_ANALYSIS < MAX_TURNS_CMAKE
        assert MAX_TURNS_ERROR_ANALYSIS < MAX_TURNS_AUTOTOOLS


class TestTokenConstants:
    """Tests for token limit constants."""

    def test_max_context_tokens(self):
        """Test MAX_CONTEXT_TOKENS value."""
        assert MAX_CONTEXT_TOKENS == 100000
        assert isinstance(MAX_CONTEXT_TOKENS, int)

    def test_large_output_threshold(self):
        """Test LARGE_OUTPUT_TOKEN_THRESHOLD value."""
        assert LARGE_OUTPUT_TOKEN_THRESHOLD == 1000
        assert isinstance(LARGE_OUTPUT_TOKEN_THRESHOLD, int)

    def test_chars_per_token(self):
        """Test CHARS_PER_TOKEN value."""
        assert CHARS_PER_TOKEN == 4
        assert isinstance(CHARS_PER_TOKEN, int)

    def test_threshold_less_than_max(self):
        """Test that large output threshold is less than max context."""
        assert LARGE_OUTPUT_TOKEN_THRESHOLD < MAX_CONTEXT_TOKENS
