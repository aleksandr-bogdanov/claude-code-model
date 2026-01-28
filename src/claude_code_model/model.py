"""Pydantic AI Model implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from claude_code_model.cli import ClaudeCodeCLI


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


@dataclass
class ParsedToolCall:
    """Parsed tool call from model response."""

    name: str
    args: dict[str, Any]


def extract_tool_calls(content: str) -> list[ParsedToolCall]:
    """
    Extract tool calls from response content.

    Detects formats:
    1. TOOL_CALL: name({"arg": "val"})
    2. <tool_call name="name">{"arg": "val"}</tool_call>

    Args:
        content: Raw response string

    Returns:
        List of parsed tool calls (empty if none found)
    """
    calls: list[ParsedToolCall] = []

    # Pattern 1: TOOL_CALL: name({...})
    pattern1 = r'TOOL_CALL:\s*(\w+)\((\{.*?\})\)'
    for match in re.finditer(pattern1, content, re.DOTALL):
        name = match.group(1)
        args_str = match.group(2)
        args = _try_parse_json(args_str)
        if args is not None:
            calls.append(ParsedToolCall(name=name, args=args))

    # Pattern 2: <tool_call name="name">{...}</tool_call>
    pattern2 = r'<tool_call\s+name="(\w+)">(.*?)</tool_call>'
    for match in re.finditer(pattern2, content, re.DOTALL):
        name = match.group(1)
        args_str = match.group(2).strip()
        args = _try_parse_json(args_str)
        if args is not None:
            calls.append(ParsedToolCall(name=name, args=args))

    return calls


@dataclass(init=False)
class ClaudeCodeModel(Model):
    """Pydantic AI Model adapter for Claude Code CLI."""

    model: Literal["sonnet", "opus", "haiku"]
    timeout: int
    cwd: Path | None
    _cli: ClaudeCodeCLI = field(repr=False)
    _model_name: str = field(repr=False)

    def __init__(
        self,
        *,
        model: Literal["sonnet", "opus", "haiku"] = "sonnet",
        timeout: int = 300,
        cwd: Path | None = None,
        settings: ModelSettings | None = None,
    ) -> None:
        """Initialize Claude Code Model."""
        self.model = model
        self.timeout = timeout
        self.cwd = cwd
        self._cli = ClaudeCodeCLI(
            model=model,
            timeout=timeout,
            cwd=cwd,
        )
        self._model_name = f"claude-code:{model}"
        super().__init__(settings=settings)

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name

    @property
    def system(self) -> str:
        """Return provider name."""
        return "claude-code"

    @classmethod
    def supported_builtin_tools(cls) -> frozenset[type[AbstractBuiltinTool]]:
        """Claude Code supports no builtin tools."""
        return frozenset()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """
        Make request to CLI and parse response.

        Args:
            messages: Conversation messages
            model_settings: Optional model settings (unused)
            model_request_parameters: Request parameters with tools and output config

        Returns:
            ModelResponse with parts
        """
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )

        prompt = self._build_prompt(messages, model_request_parameters)
        result = await self._cli.run(prompt)

        parts: list[ModelResponsePart] = []

        # Check for tool calls first
        tool_calls = extract_tool_calls(result.stdout)
        if tool_calls and model_request_parameters.function_tools:
            tool_names = {t.name for t in model_request_parameters.function_tools}
            for tc in tool_calls:
                if tc.name in tool_names:
                    parts.append(ToolCallPart(tool_name=tc.name, args=tc.args))

        # If no valid tool calls, return text
        if not parts:
            parts.append(TextPart(content=result.stdout))

        return ModelResponse(
            parts=parts,
            usage=RequestUsage(input_tokens=0, output_tokens=0),
        )

    def _build_prompt(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> str:
        """Convert Pydantic AI messages to prompt string."""
        from pydantic_ai.messages import (
            ModelRequest,
            RetryPromptPart,
            SystemPromptPart,
            ToolReturnPart,
            UserPromptPart,
        )

        sections: list[str] = []
        system_parts: list[str] = []
        conversation_parts: list[str] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                for req_part in msg.parts:
                    if isinstance(req_part, SystemPromptPart):
                        system_parts.append(req_part.content)
                    elif isinstance(req_part, UserPromptPart):
                        conversation_parts.append(f"User: {req_part.content}")
                    elif isinstance(req_part, ToolReturnPart):
                        conversation_parts.append(
                            f"Tool Result ({req_part.tool_name}): {req_part.content}"
                        )
                    elif isinstance(req_part, RetryPromptPart):
                        conversation_parts.append(f"Retry: {req_part.content}")
            elif isinstance(msg, ModelResponse):
                for resp_part in msg.parts:
                    if isinstance(resp_part, TextPart):
                        conversation_parts.append(f"Assistant: {resp_part.content}")
                    elif isinstance(resp_part, ToolCallPart):
                        args_json = json.dumps(resp_part.args)
                        conversation_parts.append(
                            f"Assistant called: {resp_part.tool_name}({args_json})"
                        )

        # Build sections
        if system_parts:
            sections.append("## Instructions\n" + "\n".join(system_parts))

        if model_request_parameters.function_tools:
            tools_text = self._format_tools(model_request_parameters.function_tools)
            sections.append("## Available Tools\n" + tools_text)

        if model_request_parameters.output_tools:
            schema_text = self._format_result_schema(
                model_request_parameters.output_tools
            )
            sections.append("## Required Output Format\n" + schema_text)

        if conversation_parts:
            sections.append("## Conversation\n" + "\n".join(conversation_parts))

        return "\n\n".join(sections)

    def _format_tools(self, tools: list[ToolDefinition]) -> str:
        """Format tool definitions for prompt."""
        lines = []
        for tool in tools:
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}")
            lines.append(f"Parameters: {json.dumps(tool.parameters_json_schema)}")
            lines.append(f"To call: TOOL_CALL: {tool.name}({{...}})")
            lines.append("")
        return "\n".join(lines)

    def _format_result_schema(self, tools: list[ToolDefinition]) -> str:
        """Format result schema for prompt."""
        if not tools:
            return ""
        tool = tools[0]  # Use first output tool
        return f"Return JSON matching this schema:\n{json.dumps(tool.parameters_json_schema, indent=2)}"


# Keep ClaudeCodeAgentModel for backwards compatibility if needed
@dataclass
class ClaudeCodeAgentModel:
    """Deprecated: Use ClaudeCodeModel directly."""

    cli: ClaudeCodeCLI
    _function_tools: list[ToolDefinition] = field(default_factory=list)
    _allow_text_result: bool = True
    _result_tools: list[ToolDefinition] = field(default_factory=list)
