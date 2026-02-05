"""LLM message and token utilities for the builder module."""

import json


def estimate_tokens(text: str) -> int:
    """
    Rough token count estimate (4 chars per token).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def filter_warnings(output: str) -> str:
    """
    Filter out warnings, keep only errors when output is large.

    Args:
        output: Build output to filter

    Returns:
        Filtered output with errors and context
    """
    lines = output.split('\n')

    # If output is small, keep everything
    if len(output) < 10000:
        return output

    # Otherwise, keep only error lines and some context
    error_keywords = ['error:', 'Error:', 'ERROR:', 'failed', 'Failed', 'FAILED']
    error_ranges = []

    for i, line in enumerate(lines):
        # Keep errors
        if any(kw in line for kw in error_keywords):
            # Include 2 lines before and after for context
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            error_ranges.append((start, end))

    if not error_ranges:
        # No errors found, keep first 20 and last 30 lines
        if len(lines) < 51:  # Fixed off-by-one bug: was <= 50
            return output
        return '\n'.join(lines[:20] + ['...'] + lines[-30:])

    # Merge overlapping ranges
    merged_ranges = []
    error_ranges.sort()
    current_start, current_end = error_ranges[0]

    for start, end in error_ranges[1:]:
        if start <= current_end:
            # Overlapping, merge
            current_end = max(current_end, end)
        else:
            # Non-overlapping, save current and start new
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end

    merged_ranges.append((current_start, current_end))

    # Extract lines from merged ranges
    filtered = []
    for start, end in merged_ranges:
        filtered.extend(lines[start:end])
        filtered.append('---')

    return '\n'.join(filtered)


def deduplicate_tool_result(
    tool_name: str,
    tool_input: dict,
    result: dict,
    seen_tools: set,
) -> dict:
    """
    Deduplicate tool results to avoid sending same content repeatedly.

    Args:
        tool_name: Name of the tool
        tool_input: Tool input parameters
        result: Tool execution result
        seen_tools: Set tracking seen tool calls (modified in-place)

    Returns:
        Potentially deduplicated result
    """
    # Create key for deduplication
    if tool_name == "read_file":
        key = f"read:{tool_input.get('file_path')}:{tool_input.get('start_line', 1)}"
        if key in seen_tools:
            # Already read this file, return short reference
            return {
                "cached": True,
                "message": f"File {tool_input.get('file_path')} already read (see earlier output)"
            }
        seen_tools.add(key)

    # Filter large outputs
    if "output" in result and isinstance(result["output"], str):
        if estimate_tokens(result["output"]) > 1000:
            result = result.copy()
            result["output"] = filter_warnings(result["output"])

    if "content" in result and isinstance(result["content"], str):
        if estimate_tokens(result["content"]) > 1000:
            result = result.copy()
            result["content"] = filter_warnings(result["content"])

    return result


def estimate_messages_tokens(messages: list) -> int:
    """
    Estimate total tokens in messages array.

    Args:
        messages: List of message dicts

    Returns:
        Estimated total token count
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total += estimate_tokens(json.dumps(item))
                elif isinstance(item, str):
                    total += estimate_tokens(item)
    return total


def truncate_messages(messages: list, max_tokens: int = 100000) -> list:
    """
    Truncate old messages to stay under token limit.

    Args:
        messages: List of message dicts
        max_tokens: Maximum token limit

    Returns:
        Truncated list of messages
    """
    current_tokens = estimate_messages_tokens(messages)

    if current_tokens <= max_tokens:
        return messages

    # Keep first message (initial request) and recent messages
    if len(messages) <= 3:
        return messages

    # Keep first message and last N messages
    truncated = [messages[0]]

    # Add recent messages from the end
    for msg in reversed(messages[1:]):
        msg_tokens = estimate_tokens(json.dumps(msg.get("content", "")))
        if current_tokens - msg_tokens < max_tokens:
            truncated.insert(1, msg)
        else:
            current_tokens -= msg_tokens

    if len(truncated) < len(messages):
        # Add a marker that we truncated
        truncated.insert(1, {
            "role": "user",
            "content": f"[{len(messages) - len(truncated)} earlier messages truncated to save context]"
        })

    return truncated
