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


# Build/configure tool names that should only keep latest attempt
# These are iterative trial-and-error tools where only the latest attempt matters
BUILD_TOOLS = {
    # CMake (from actions.py)
    "cmake_configure",
    "cmake_build",
    # Configure/Make (from actions.py)
    "bootstrap",
    "autoreconf",
    "run_configure",
    "make_build",
    "make_clean",
    "make_distclean",
    # Package installation
    "install_packages",
}


def track_tool_result(
    tool_name: str,
    tool_input: dict,
    result: dict,
    tool_history: dict,
    current_turn: int = 0,
) -> dict:
    """
    Track tool results for context management.

    - File reads: Track for overlapping range compression
    - Build/configure tools: Track for keeping only latest attempt

    Args:
        tool_name: Name of the tool
        tool_input: Tool input parameters
        result: Tool execution result
        tool_history: Dict tracking tool calls (modified in-place)
                      - 'reads': file_path -> list of (turn, start_line, end_line)
                      - 'builds': tool_name -> latest_turn
        current_turn: Current turn number in the ReAct loop

    Returns:
        The result (always returns full content)
    """
    # Initialize history sections if needed
    if 'reads' not in tool_history:
        tool_history['reads'] = {}
    if 'builds' not in tool_history:
        tool_history['builds'] = {}

    if tool_name == "read_file":
        # Use the actual result to determine what was read
        file_path = result.get('path') or tool_input.get('file_path')

        # Skip tracking if this was an error
        if 'error' in result:
            return result

        # Get the actual range that was read from the result
        actual_start = result.get('start_line')
        actual_end = result.get('end_line')

        if actual_start is None or actual_end is None:
            return result

        # Track this read
        if file_path not in tool_history['reads']:
            tool_history['reads'][file_path] = []
        tool_history['reads'][file_path].append((current_turn, actual_start, actual_end))

    elif tool_name in BUILD_TOOLS:
        # Track build/configure attempts - only latest matters
        tool_history['builds'][tool_name] = current_turn

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


# Keep old name for backwards compatibility
def deduplicate_tool_result(
    tool_name: str,
    tool_input: dict,
    result: dict,
    tool_history: dict,
    current_turn: int = 0,
) -> dict:
    """Backwards-compatible alias for track_tool_result."""
    return track_tool_result(tool_name, tool_input, result, tool_history, current_turn)


def _ranges_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check if two line ranges overlap."""
    return start1 <= end2 and start2 <= end1


def compress_stale_results(messages: list, tool_history: dict) -> list:
    """
    Compress stale tool results to save context tokens.

    - File reads: Compress older reads if a newer read overlaps the same range
    - Build/configure: Keep only the latest attempt, compress older attempts

    Args:
        messages: List of message dicts
        tool_history: Dict with 'reads' and 'builds' tracking info

    Returns:
        Messages with stale results compressed
    """
    seen_reads = tool_history.get('reads', {})
    latest_builds = tool_history.get('builds', {})

    # Find the latest read for each file (for overlapping range detection)
    latest_reads = {}  # file_path -> (latest_turn, start, end)
    for file_path, reads in seen_reads.items():
        if not reads:
            continue
        sorted_reads = sorted(reads, key=lambda x: x[0], reverse=True)
        latest_reads[file_path] = sorted_reads[0]

    # Process messages and compress stale results
    compressed = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            compressed.append(msg)
            continue

        # Process tool_result blocks
        new_content = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                new_content.append(block)
                continue

            # Parse the tool result content
            try:
                result_str = block.get("content", "")
                result = json.loads(result_str) if isinstance(result_str, str) else result_str
            except (json.JSONDecodeError, TypeError):
                new_content.append(block)
                continue

            # Check for build/configure tools - keep only latest attempt
            # We need to find the tool_use block to know which tool this result is for
            tool_use_id = block.get("tool_use_id")
            if not tool_use_id:
                new_content.append(block)
                continue

            tool_name = _find_tool_name_for_result(messages, tool_use_id)

            if tool_name and tool_name in BUILD_TOOLS:
                # Check if there's a later attempt of this same tool
                latest_turn = latest_builds.get(tool_name)
                if latest_turn is not None:
                    # This is an older attempt - compress it
                    # We detect "older" by checking if there are results from this tool
                    # after this message in the conversation
                    is_latest = _is_latest_build_attempt(messages, msg, tool_name)
                    if not is_latest:
                        success = result.get("success", False)
                        new_content.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps({
                                "compressed": True,
                                "message": f"Earlier {tool_name} attempt ({'succeeded' if success else 'failed'}), see later attempt"
                            })
                        })
                        continue

            # Check for file reads - compress if overlapping with later read
            file_path = result.get("path")
            if file_path and file_path in latest_reads:
                result_start = result.get("start_line")
                result_end = result.get("end_line")

                if result_start is not None and result_end is not None:
                    # Check if this read overlaps with a LATER read (higher turn number)
                    # We need to find which turn this read is from
                    this_read_turn = None
                    for turn, start, end in seen_reads[file_path]:
                        if start == result_start and end == result_end:
                            this_read_turn = turn
                            break

                    is_stale = False
                    if this_read_turn is not None:
                        for turn, start, end in seen_reads[file_path]:
                            # Only consider reads from LATER turns
                            if turn > this_read_turn and _ranges_overlap(result_start, result_end, start, end):
                                is_stale = True
                                break

                    if is_stale:
                        new_content.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps({
                                "compressed": True,
                                "message": f"File {file_path} lines {result_start}-{result_end} re-read later"
                            })
                        })
                        continue

            # Not stale, keep original
            new_content.append(block)

        compressed.append({"role": msg["role"], "content": new_content})

    return compressed


def _find_tool_name_for_result(messages: list, tool_use_id: str) -> str | None:
    """Find the tool name that corresponds to a tool_use_id."""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("id") == tool_use_id:
                    return block.get("name")
    return None


def _is_latest_build_attempt(messages: list, current_msg: dict, tool_name: str) -> bool:
    """Check if the current message contains the latest build attempt for a tool."""
    found_current = False
    for msg in messages:
        if msg is current_msg:
            found_current = True
            continue
        if not found_current:
            continue
        # Look for later uses of the same tool
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("name") == tool_name:
                    return False  # Found a later use
    return True


# Backwards compatible alias
def compress_stale_reads(messages: list, tool_history: dict) -> list:
    """Backwards-compatible alias for compress_stale_results."""
    return compress_stale_results(messages, tool_history)


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
