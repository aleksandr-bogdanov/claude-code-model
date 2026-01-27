"""Pydantic AI Model implementation."""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(content: str) -> dict[str, Any] | None:
    """
    Extract JSON object from response content.

    Tries in order:
    1. Direct JSON parse
    2. Markdown code block
    3. Embedded {...} in text

    Args:
        content: Raw response string

    Returns:
        Parsed dict or None if no valid JSON object found
    """
    content = content.strip()
    if not content:
        return None

    # Try 1: Direct parse
    result = _try_parse_json(content)
    if result is not None:
        return result

    # Try 2: Markdown code block
    block = _extract_code_block(content)
    if block:
        result = _try_parse_json(block)
        if result is not None:
            return result

    # Try 3: Find embedded JSON object
    obj_str = _find_json_object(content)
    if obj_str:
        result = _try_parse_json(obj_str)
        if result is not None:
            return result

    return None


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Try to parse text as JSON, return dict or None."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _extract_code_block(content: str) -> str | None:
    """Extract content from ```json ... ``` or ``` ... ``` block."""
    pattern = r"```(?:json)?\s*\n(.*?)\n```"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _find_json_object(content: str) -> str | None:
    """Find balanced {...} in content."""
    start = content.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(content[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : i + 1]

    return None
