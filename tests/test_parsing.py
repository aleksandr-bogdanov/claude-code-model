"""Tests for response parsing utilities."""
from __future__ import annotations

import pytest

from claude_code_model.model import extract_json


class TestExtractJson:
    """Test JSON extraction from various formats."""

    # Pure JSON
    def test_pure_json_object(self) -> None:
        assert extract_json('{"key": "value"}') == {"key": "value"}

    def test_pure_json_with_leading_whitespace(self) -> None:
        assert extract_json('  {"key": "value"}') == {"key": "value"}

    def test_pure_json_with_trailing_whitespace(self) -> None:
        assert extract_json('{"key": "value"}  \n') == {"key": "value"}

    # Markdown code blocks
    def test_markdown_json_block(self) -> None:
        content = '```json\n{"key": "value"}\n```'
        assert extract_json(content) == {"key": "value"}

    def test_markdown_block_no_language(self) -> None:
        content = '```\n{"key": "value"}\n```'
        assert extract_json(content) == {"key": "value"}

    def test_markdown_block_with_surrounding_text(self) -> None:
        content = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        assert extract_json(content) == {"key": "value"}

    # Embedded JSON
    def test_embedded_json_in_text(self) -> None:
        content = 'The answer is {"key": "value"} as requested.'
        assert extract_json(content) == {"key": "value"}

    def test_embedded_json_at_end(self) -> None:
        content = 'Result: {"status": "ok"}'
        assert extract_json(content) == {"status": "ok"}

    # Nested structures
    def test_nested_objects(self) -> None:
        content = '{"outer": {"inner": "value"}}'
        assert extract_json(content) == {"outer": {"inner": "value"}}

    def test_deeply_nested(self) -> None:
        content = '{"a": {"b": {"c": {"d": 1}}}}'
        assert extract_json(content) == {"a": {"b": {"c": {"d": 1}}}}

    def test_json_with_array(self) -> None:
        content = '{"items": [1, 2, 3]}'
        assert extract_json(content) == {"items": [1, 2, 3]}

    def test_json_with_nested_array_of_objects(self) -> None:
        content = '{"users": [{"name": "a"}, {"name": "b"}]}'
        assert extract_json(content) == {"users": [{"name": "a"}, {"name": "b"}]}

    # Edge cases
    def test_empty_string_returns_none(self) -> None:
        assert extract_json("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert extract_json("   \n\t  ") is None

    def test_no_json_returns_none(self) -> None:
        assert extract_json("Just plain text without JSON") is None

    def test_invalid_json_returns_none(self) -> None:
        assert extract_json("{invalid json}") is None

    def test_unclosed_brace_returns_none(self) -> None:
        assert extract_json('{"key": "value"') is None

    def test_json_array_at_root_returns_none(self) -> None:
        # We only extract objects, not arrays at root level
        assert extract_json("[1, 2, 3]") is None

    def test_json_with_special_chars(self) -> None:
        content = '{"msg": "hello\\nworld"}'
        assert extract_json(content) == {"msg": "hello\nworld"}

    def test_json_with_unicode(self) -> None:
        content = '{"emoji": "🎉"}'
        assert extract_json(content) == {"emoji": "🎉"}
