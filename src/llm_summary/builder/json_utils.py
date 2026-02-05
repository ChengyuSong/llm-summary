"""JSON parsing utilities for LLM responses."""

import json


def strip_markdown_json(text: str) -> str:
    """
    Strip ```json``` code blocks from LLM response.

    Args:
        text: LLM response that may contain markdown code blocks

    Returns:
        Cleaned JSON string
    """
    json_str = text.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]  # Remove ```json
    if json_str.startswith("```"):
        json_str = json_str[3:]  # Remove ```
    if json_str.endswith("```"):
        json_str = json_str[:-3:]  # Remove trailing ```
    return json_str.strip()


def parse_llm_json(
    text: str,
    default_response: dict | None = None,
    verbose: bool = False,
) -> dict:
    """
    Parse JSON from LLM response with error handling.

    Args:
        text: LLM response containing JSON
        default_response: Fallback dict to return on parse error
        verbose: If True, print error messages

    Returns:
        Parsed JSON dict or default_response on error
    """
    if default_response is None:
        default_response = {}

    try:
        json_str = strip_markdown_json(text)
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        if verbose:
            print(f"[ERROR] Failed to parse LLM response as JSON: {e}")
            print(f"[ERROR] Response: {text[:500]}...")
        return default_response
