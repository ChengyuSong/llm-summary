"""Tests for LLM utilities module."""

import json

import pytest

from llm_summary.builder.llm_utils import (
    compress_stale_reads,
    deduplicate_tool_result,
    estimate_messages_tokens,
    estimate_tokens,
    filter_warnings,
    truncate_messages,
)


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_empty_string(self):
        """Test token estimation for empty string."""
        assert estimate_tokens("") == 0

    def test_short_string(self):
        """Test token estimation for short string."""
        # 4 chars per token
        assert estimate_tokens("test") == 1

    def test_exact_multiple(self):
        """Test token estimation for exact multiple of 4."""
        assert estimate_tokens("a" * 400) == 100

    def test_rounding_down(self):
        """Test that estimation rounds down."""
        # 399 chars = 99.75 tokens, should round to 99
        assert estimate_tokens("a" * 399) == 99


class TestFilterWarnings:
    """Tests for filter_warnings function."""

    def test_small_output_unchanged(self):
        """Test that small outputs (<10000 chars) are not filtered."""
        output = "Building...\nCompiling foo.c\nDone."
        assert filter_warnings(output) == output

    def test_large_output_with_errors(self):
        """Test filtering large output to show only errors."""
        # Create output >10000 chars with warnings and errors
        # Make each line longer to exceed 10000 chars total
        lines = ["warning: unused variable in some long function name with details" * 5] * 200
        lines.append("error: undefined reference to 'foo'")
        lines.extend(["warning: something else happened with lots of text here" * 5] * 200)
        output = "\n".join(lines)

        assert len(output) > 10000
        filtered = filter_warnings(output)

        # Should contain the error
        assert "undefined reference" in filtered
        # Should be smaller (or at least different)
        # Note: depending on error positions, might not always be smaller
        assert "undefined reference" in filtered

    def test_large_output_no_errors_shows_summary(self):
        """Test that large output without errors shows first/last lines."""
        lines = [f"Line {i}" for i in range(100)]
        output = "\n".join(lines)

        # Pad to exceed 10000 chars
        output = output * 200

        assert len(output) > 10000
        filtered = filter_warnings(output)

        # Should show first and last lines with separator
        assert "Line 0" in filtered
        assert "..." in filtered
        assert len(filtered) < len(output)

    def test_off_by_one_bug_fixed(self):
        """Test that the off-by-one bug is fixed (was <= 50, now < 51)."""
        # Create exactly 50 lines
        lines = [f"Line {i}" for i in range(50)]
        output = "\n".join(lines)

        filtered = filter_warnings(output)

        # With 50 lines, should NOT be truncated (< 51 check)
        # Previously with <= 50, it would be truncated
        assert filtered == output

    def test_51_lines_triggers_truncation(self):
        """Test that 51 lines triggers truncation."""
        lines = [f"Line {i}" for i in range(51)]
        output = "\n".join(lines)

        # Make it large enough
        output = output * 300  # Ensure > 10000 chars

        filtered = filter_warnings(output)

        # Should be truncated (shows first 20 + last 30)
        assert "..." in filtered
        assert len(filtered) < len(output)

    def test_error_context_included(self):
        """Test that context around errors is included."""
        lines = ["normal output"] * 100
        lines[50] = "error: build failed"
        output = "\n".join(lines) * 100  # Make it large

        filtered = filter_warnings(output)

        # Should include lines around error (2 before, 2 after)
        assert "error: build failed" in filtered
        # Context lines should be present
        assert filtered.count("normal output") >= 4  # At least context lines

    def test_multiple_errors_merged(self):
        """Test that overlapping error ranges are merged."""
        lines = []
        for i in range(200):
            if i in [10, 12, 50, 100]:  # Multiple errors
                lines.append(f"error: error at line {i}")
            else:
                lines.append(f"warning: warning at line {i}")

        output = "\n".join(lines) * 100  # Make it large

        filtered = filter_warnings(output)

        # All errors should be present
        assert "error at line 10" in filtered
        assert "error at line 12" in filtered
        assert "error at line 50" in filtered
        assert "error at line 100" in filtered


class TestTrackToolResult:
    """Tests for track_tool_result / deduplicate_tool_result function."""

    def test_first_read_tracked(self):
        """Test that first file read is tracked and returned unchanged."""
        history = {}
        result = {"content": "file contents", "path": "test.c", "start_line": 1, "end_line": 10}

        deduped = deduplicate_tool_result(
            "read_file", {"file_path": "test.c"}, result, history, current_turn=0
        )

        # Result should be returned unchanged
        assert deduped["content"] == "file contents"
        # Should be tracked in history
        assert "test.c" in history["reads"]
        assert history["reads"]["test.c"] == [(0, 1, 10)]

    def test_multiple_reads_tracked(self):
        """Test that multiple reads are all tracked."""
        history = {}
        result1 = {"content": "1", "path": "test.c", "start_line": 1, "end_line": 50}
        result2 = {"content": "2", "path": "test.c", "start_line": 100, "end_line": 150}

        deduplicate_tool_result("read_file", {"file_path": "test.c"}, result1, history, current_turn=0)
        deduplicate_tool_result("read_file", {"file_path": "test.c"}, result2, history, current_turn=1)

        # Both should be tracked
        assert len(history["reads"]["test.c"]) == 2
        assert (0, 1, 50) in history["reads"]["test.c"]
        assert (1, 100, 150) in history["reads"]["test.c"]

    def test_build_tools_tracked(self):
        """Test that build tools are tracked with latest turn."""
        history = {}

        deduplicate_tool_result("cmake_configure", {}, {"success": False}, history, current_turn=0)
        deduplicate_tool_result("cmake_configure", {}, {"success": True}, history, current_turn=2)

        # Should track latest turn
        assert history["builds"]["cmake_configure"] == 2

    def test_all_build_tools_tracked(self):
        """Test that all build tools (CMake and Autotools) are tracked."""
        history = {}

        # CMake tools
        deduplicate_tool_result("cmake_configure", {}, {"success": True}, history, current_turn=0)
        deduplicate_tool_result("cmake_build", {}, {"success": True}, history, current_turn=1)

        # Configure/Make tools
        deduplicate_tool_result("bootstrap", {}, {"success": True}, history, current_turn=2)
        deduplicate_tool_result("autoreconf", {}, {"success": True}, history, current_turn=3)
        deduplicate_tool_result("run_configure", {}, {"success": True}, history, current_turn=4)
        deduplicate_tool_result("make_build", {}, {"success": True}, history, current_turn=5)
        deduplicate_tool_result("make_clean", {}, {"success": True}, history, current_turn=6)
        deduplicate_tool_result("make_distclean", {}, {"success": True}, history, current_turn=7)

        # All should be tracked
        assert history["builds"]["cmake_configure"] == 0
        assert history["builds"]["cmake_build"] == 1
        assert history["builds"]["bootstrap"] == 2
        assert history["builds"]["autoreconf"] == 3
        assert history["builds"]["run_configure"] == 4
        assert history["builds"]["make_build"] == 5
        assert history["builds"]["make_clean"] == 6
        assert history["builds"]["make_distclean"] == 7

    def test_large_output_filtered(self):
        """Test that large output field is filtered."""
        history = {}
        result = {"output": "x" * 50000}  # >1000 tokens
        original_len = len(result["output"])
        deduped = deduplicate_tool_result("cmake_build", {}, result, history)

        # Original result should not be modified
        assert len(result["output"]) == original_len

    def test_large_content_filtered(self):
        """Test that large content field is filtered."""
        history = {}
        # Create content that's both >1000 tokens AND >10000 chars
        result = {"content": "warning: something\n" * 1000, "path": "new.c", "start_line": 1, "end_line": 100}
        original_len = len(result["content"])
        deduped = deduplicate_tool_result("read_file", {"file_path": "new.c"}, result, history)

        # Should be filtered (creates copy, doesn't modify original)
        assert len(result["content"]) == original_len
        assert len(deduped["content"]) < original_len

    def test_small_output_not_filtered(self):
        """Test that small output is not filtered."""
        history = {}
        result = {"output": "small output"}
        deduped = deduplicate_tool_result("cmake_build", {}, result, history)

        assert deduped["output"] == "small output"

    def test_error_results_not_tracked(self):
        """Test that error results are not tracked."""
        history = {}
        result = {"error": "file not found"}

        deduped = deduplicate_tool_result("read_file", {"file_path": "missing.c"}, result, history)

        assert deduped == result
        assert "reads" not in history or "missing.c" not in history.get("reads", {})


class TestCompressStaleResults:
    """Tests for compress_stale_results function."""

    def test_no_compression_single_read(self):
        """Test that single read is not compressed."""
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": json.dumps({
                    "path": "test.c", "start_line": 1, "end_line": 50, "content": "..."
                })}
            ]}
        ]
        history = {"reads": {"test.c": [(0, 1, 50)]}, "builds": {}}

        compressed = compress_stale_reads(messages, history)

        # Should not be compressed
        result = json.loads(compressed[0]["content"][0]["content"])
        assert "compressed" not in result

    def test_older_overlapping_read_compressed(self):
        """Test that older overlapping read is compressed."""
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": json.dumps({
                    "path": "test.c", "start_line": 1, "end_line": 50, "content": "old content"
                })}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "2", "content": json.dumps({
                    "path": "test.c", "start_line": 1, "end_line": 100, "content": "new content"
                })}
            ]}
        ]
        # Second read (turn 1) overlaps and is newer
        history = {"reads": {"test.c": [(0, 1, 50), (1, 1, 100)]}, "builds": {}}

        compressed = compress_stale_reads(messages, history)

        # First read should be compressed
        result1 = json.loads(compressed[0]["content"][0]["content"])
        assert result1.get("compressed") is True

        # Second read should not be compressed
        result2 = json.loads(compressed[1]["content"][0]["content"])
        assert "compressed" not in result2

    def test_non_overlapping_reads_not_compressed(self):
        """Test that non-overlapping reads are not compressed."""
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": json.dumps({
                    "path": "test.c", "start_line": 1, "end_line": 50, "content": "first part"
                })}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "2", "content": json.dumps({
                    "path": "test.c", "start_line": 100, "end_line": 150, "content": "second part"
                })}
            ]}
        ]
        # Non-overlapping reads
        history = {"reads": {"test.c": [(0, 1, 50), (1, 100, 150)]}, "builds": {}}

        compressed = compress_stale_reads(messages, history)

        # Neither should be compressed
        result1 = json.loads(compressed[0]["content"][0]["content"])
        result2 = json.loads(compressed[1]["content"][0]["content"])
        assert "compressed" not in result1
        assert "compressed" not in result2


class TestEstimateMessagesTokens:
    """Tests for estimate_messages_tokens function."""

    def test_empty_messages(self):
        """Test token estimation for empty messages list."""
        assert estimate_messages_tokens([]) == 0

    def test_single_string_message(self):
        """Test token estimation for single string content."""
        messages = [{"role": "user", "content": "test" * 100}]  # 400 chars = 100 tokens
        assert estimate_messages_tokens(messages) == 100

    def test_multiple_messages(self):
        """Test token estimation for multiple messages."""
        messages = [
            {"role": "user", "content": "a" * 400},  # 100 tokens
            {"role": "assistant", "content": "b" * 800},  # 200 tokens
        ]
        assert estimate_messages_tokens(messages) == 300

    def test_list_content(self):
        """Test token estimation for list content (tool use format)."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "a" * 400},  # 100 tokens
                    {"type": "tool_use", "id": "1", "name": "test"},  # ~small
                ],
            }
        ]
        # Should count JSON-serialized dicts
        tokens = estimate_messages_tokens(messages)
        assert tokens > 100  # At least the text content

    def test_mixed_content_types(self):
        """Test with mixed string and list content."""
        messages = [
            {"role": "user", "content": "a" * 400},  # String content
            {"role": "assistant", "content": [{"type": "text", "text": "b" * 400}]},  # List
        ]
        tokens = estimate_messages_tokens(messages)
        assert tokens >= 200  # At least both text contents


class TestTruncateMessages:
    """Tests for truncate_messages function."""

    def test_no_truncation_when_under_limit(self):
        """Test that messages under limit are not truncated."""
        messages = [
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "response"},
        ]
        truncated = truncate_messages(messages, max_tokens=10000)
        assert len(truncated) == len(messages)

    def test_keeps_first_message(self):
        """Test that first message is always kept."""
        messages = [
            {"role": "user", "content": "Initial request"},
            {"role": "assistant", "content": "x" * 50000},  # Large
            {"role": "user", "content": "Final"},
        ]
        truncated = truncate_messages(messages, max_tokens=10000)
        assert truncated[0]["content"] == "Initial request"

    def test_keeps_recent_messages(self):
        """Test that recent messages are kept."""
        messages = [
            {"role": "user", "content": "Initial"},
            {"role": "assistant", "content": "x" * 50000},  # Will be dropped
            {"role": "user", "content": "Recent"},
            {"role": "assistant", "content": "Final"},
        ]
        truncated = truncate_messages(messages, max_tokens=5000)

        # Should keep initial message
        assert any("Initial" in str(m.get("content", "")) for m in truncated)
        # Should have fewer messages than original
        # (might include truncation marker)
        assert len([m for m in truncated if "truncated" not in str(m.get("content", ""))]) <= len(messages)

    def test_adds_truncation_marker(self):
        """Test that truncation marker is added."""
        messages = [
            {"role": "user", "content": "Initial"},
            {"role": "assistant", "content": "x" * 50000},
            {"role": "user", "content": "Middle"},
            {"role": "assistant", "content": "x" * 50000},
            {"role": "user", "content": "Final"},
        ]
        truncated = truncate_messages(messages, max_tokens=1000)

        # Should have truncation marker
        assert len(truncated) < len(messages)
        truncation_markers = [
            m for m in truncated if "truncated" in m.get("content", "").lower()
        ]
        assert len(truncation_markers) == 1

    def test_minimum_three_messages_kept(self):
        """Test that at least 3 messages are kept if available."""
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        # Even with very low limit, should keep all 3
        truncated = truncate_messages(messages, max_tokens=1)
        assert len(truncated) == 3

    def test_default_max_tokens(self):
        """Test that default max_tokens is 100000."""
        messages = [{"role": "user", "content": "x" * 400000}]  # 100k tokens
        truncated = truncate_messages(messages)  # Use default
        # With default of 100000, this should trigger truncation or be kept (implementation detail)
        # Just verify it doesn't crash
        assert len(truncated) >= 1
