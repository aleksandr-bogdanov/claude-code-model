"""Tests for response parsing utilities."""

from __future__ import annotations


from claude_code_model.model import extract_json, ParsedToolCall, extract_tool_calls


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


class TestExtractToolCalls:
    """Test tool call extraction."""

    # TOOL_CALL format
    def test_tool_call_format_basic(self) -> None:
        content = 'TOOL_CALL: search({"query": "python"})'
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].args == {"query": "python"}

    def test_tool_call_with_surrounding_text(self) -> None:
        content = 'I will search.\nTOOL_CALL: search({"q": "test"})\nDone.'
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].args == {"q": "test"}

    def test_tool_call_complex_args(self) -> None:
        content = 'TOOL_CALL: create({"name": "test", "count": 5, "active": true})'
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].args == {"name": "test", "count": 5, "active": True}

    # XML format
    def test_xml_format_basic(self) -> None:
        content = '<tool_call name="search">{"query": "python"}</tool_call>'
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].args == {"query": "python"}

    def test_xml_format_with_whitespace(self) -> None:
        content = """<tool_call name="search">
            {"query": "test"}
        </tool_call>"""
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].args == {"query": "test"}

    # Multiple calls
    def test_multiple_tool_calls_same_format(self) -> None:
        content = """TOOL_CALL: first({"a": 1})
        TOOL_CALL: second({"b": 2})"""
        calls = extract_tool_calls(content)
        assert len(calls) == 2
        assert calls[0].name == "first"
        assert calls[1].name == "second"

    def test_multiple_tool_calls_mixed_formats(self) -> None:
        content = """TOOL_CALL: func1({"x": 1})
        <tool_call name="func2">{"y": 2}</tool_call>"""
        calls = extract_tool_calls(content)
        assert len(calls) == 2

    # Edge cases
    def test_no_tool_calls_returns_empty(self) -> None:
        assert extract_tool_calls("Just regular text") == []

    def test_empty_string_returns_empty(self) -> None:
        assert extract_tool_calls("") == []

    def test_invalid_json_skipped(self) -> None:
        content = "TOOL_CALL: bad({not valid json})"
        assert extract_tool_calls(content) == []

    def test_partial_match_ignored(self) -> None:
        content = "TOOL_CALL: incomplete("
        assert extract_tool_calls(content) == []

    def test_tool_call_with_nested_json(self) -> None:
        content = 'TOOL_CALL: update({"data": {"nested": "value"}})'
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].args == {"data": {"nested": "value"}}


class TestParsedToolCall:
    """Test ParsedToolCall dataclass."""

    def test_attributes(self) -> None:
        tc = ParsedToolCall(name="test", args={"key": "value"})
        assert tc.name == "test"
        assert tc.args == {"key": "value"}
