"""Tests for JSON parsing utilities."""

import pytest

from llm_summary.builder.json_utils import parse_llm_json, strip_markdown_json


class TestStripMarkdownJson:
    """Tests for strip_markdown_json function."""

    def test_strip_json_code_block(self):
        """Test stripping ```json``` code blocks."""
        text = '```json\n{"key": "value"}\n```'
        result = strip_markdown_json(text)
        assert result == '{"key": "value"}'

    def test_strip_generic_code_block(self):
        """Test stripping ``` code blocks without language."""
        text = '```\n{"key": "value"}\n```'
        result = strip_markdown_json(text)
        assert result == '{"key": "value"}'

    def test_plain_json(self):
        """Test that plain JSON is unchanged."""
        text = '{"key": "value"}'
        result = strip_markdown_json(text)
        assert result == '{"key": "value"}'

    def test_whitespace_handling(self):
        """Test that leading/trailing whitespace is stripped."""
        text = '  \n  {"key": "value"}  \n  '
        result = strip_markdown_json(text)
        assert result == '{"key": "value"}'

    def test_json_block_with_extra_text(self):
        """Test JSON block with extra text before/after."""
        text = 'Here is the JSON:\n```json\n{"key": "value"}\n```\nDone'
        result = strip_markdown_json(text)
        # Should remove opening markers but only trailing ```
        assert '{"key": "value"}' in result
        assert not result.startswith('```')

    def test_empty_string(self):
        """Test empty string handling."""
        result = strip_markdown_json('')
        assert result == ''


class TestParseLLMJson:
    """Tests for parse_llm_json function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        text = '{"key": "value", "number": 42}'
        result = parse_llm_json(text)
        assert result == {"key": "value", "number": 42}

    def test_parse_json_in_code_block(self):
        """Test parsing JSON wrapped in code blocks."""
        text = '```json\n{"key": "value"}\n```'
        result = parse_llm_json(text)
        assert result == {"key": "value"}

    def test_parse_invalid_json_with_default(self):
        """Test that invalid JSON returns default."""
        text = 'not valid json {'
        default = {"error": "parse_failed"}
        result = parse_llm_json(text, default_response=default, verbose=False)
        assert result == default

    def test_parse_invalid_json_without_default(self):
        """Test that invalid JSON returns empty dict by default."""
        text = 'not valid json {'
        result = parse_llm_json(text, verbose=False)
        assert result == {}

    def test_parse_array(self):
        """Test parsing JSON array."""
        text = '[1, 2, 3]'
        result = parse_llm_json(text)
        assert result == [1, 2, 3]

    def test_parse_nested_json(self):
        """Test parsing nested JSON structures."""
        text = '''```json
        {
            "flags": ["-DFLAG=1", "-DFLAG=2"],
            "config": {
                "enabled": true,
                "timeout": 300
            }
        }
        ```'''
        result = parse_llm_json(text)
        assert result["flags"] == ["-DFLAG=1", "-DFLAG=2"]
        assert result["config"]["enabled"] is True
        assert result["config"]["timeout"] == 300

    def test_verbose_mode_prints_error(self, capsys):
        """Test that verbose mode prints error message."""
        text = 'invalid json'
        result = parse_llm_json(text, verbose=True)
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out
        assert "Failed to parse" in captured.out

    def test_non_verbose_mode_silent(self, capsys):
        """Test that non-verbose mode doesn't print errors."""
        text = 'invalid json'
        result = parse_llm_json(text, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_parse_with_trailing_comma(self):
        """Test that trailing commas cause parse error (JSON standard)."""
        text = '{"key": "value",}'  # Trailing comma - invalid JSON
        result = parse_llm_json(text, verbose=False)
        assert result == {}  # Should return default

    def test_parse_complex_llm_response(self):
        """Test parsing a realistic LLM response with explanation."""
        # strip_markdown_json only strips from start/end, not middle text
        text = """```json
{
    "diagnosis": "Missing dependency",
    "suggested_flags": ["-DUSE_SYSTEM_LIB=ON"],
    "confidence": "high"
}
```"""
        result = parse_llm_json(text)
        assert result["diagnosis"] == "Missing dependency"
        assert "-DUSE_SYSTEM_LIB=ON" in result["suggested_flags"]
        assert result["confidence"] == "high"
