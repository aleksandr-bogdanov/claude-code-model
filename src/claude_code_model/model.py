"""Pydantic AI Model implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
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

logger = logging.getLogger(__name__)


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
    pattern1 = r"TOOL_CALL:\s*(\w+)\((\{.*?\})\)"
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
    verbose: bool
    _cli: ClaudeCodeCLI = field(repr=False)
    _model_name: str = field(repr=False)
    _request_count: int = field(repr=False, default=0)
    _total_time: float = field(repr=False, default=0.0)

    def __init__(
        self,
        *,
        model: Literal["sonnet", "opus", "haiku"] = "sonnet",
        timeout: int = 30,  # Default 30 seconds per request
        cwd: Path | None = None,
        verbose: bool = False,
        settings: ModelSettings | None = None,
    ) -> None:
        """
        Initialize Claude Code Model.

        Args:
            model: Which Claude model to use (sonnet, opus, haiku)
            timeout: Maximum time to wait for CLI response in seconds (default 30)
            cwd: Working directory for CLI execution
            verbose: If True, enable debug logging of prompts and responses
            settings: Optional Pydantic AI model settings
        """
        self.model = model
        self.timeout = timeout
        self.cwd = cwd
        self.verbose = verbose
        self._cli = ClaudeCodeCLI(
            model=model,
            timeout=timeout,
            cwd=cwd,
            verbose=verbose,
        )
        self._model_name = f"claude-code:{model}"
        self._request_count = 0
        self._total_time = 0.0
        if verbose:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
                )
                logger.addHandler(handler)
        super().__init__(settings=settings)

    @property
    def request_count(self) -> int:
        """Number of CLI requests made."""
        return self._request_count

    @property
    def total_time(self) -> float:
        """Total time spent in CLI requests (seconds)."""
        return self._total_time

    def reset_stats(self) -> None:
        """Reset request count and timing stats."""
        self._request_count = 0
        self._total_time = 0.0

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

        self._request_count += 1
        request_num = self._request_count

        if self.verbose:
            logger.debug("=" * 60)
            logger.debug(
                "REQUEST #%d (total time so far: %.1fs)",
                request_num,
                self._total_time,
            )
            logger.debug("=" * 60)
            logger.debug("Building prompt from %d messages", len(messages))
            if model_request_parameters.function_tools:
                available_tools = [
                    t.name for t in model_request_parameters.function_tools
                ]
                logger.debug("Available tools: %s", available_tools)
            if model_request_parameters.output_tools:
                logger.debug(
                    "Output schema: %s",
                    model_request_parameters.output_tools[0].name
                    if model_request_parameters.output_tools
                    else "None",
                )

        prompt = self._build_prompt(messages, model_request_parameters)
        result = await self._cli.run(prompt)
        self._total_time += result.elapsed_seconds

        parts: list[ModelResponsePart] = []

        # Check for tool calls first
        tool_calls = extract_tool_calls(result.stdout)

        if self.verbose:
            logger.debug("Parsing response for tool calls...")
            logger.debug("Found %d potential tool call(s)", len(tool_calls))
            for tc in tool_calls:
                logger.debug("  - Tool: %s, Args: %s", tc.name, tc.args)

        if tool_calls and model_request_parameters.function_tools:
            tool_names = {t.name for t in model_request_parameters.function_tools}
            for tc in tool_calls:
                if tc.name in tool_names:
                    if self.verbose:
                        logger.debug("Valid tool call: %s", tc.name)
                    parts.append(ToolCallPart(tool_name=tc.name, args=tc.args))
                elif self.verbose:
                    logger.debug(
                        "Ignoring unknown tool: %s (available: %s)", tc.name, tool_names
                    )

        # If no valid tool calls, return text
        if not parts:
            content = result.stdout
            # If structured output is expected, try to extract just the JSON
            if model_request_parameters.output_tools:
                extracted = extract_json(content)
                if extracted is not None:
                    content = json.dumps(extracted)
                    if self.verbose:
                        logger.debug("Extracted JSON for structured output")
                elif self.verbose:
                    logger.debug(
                        "Could not extract JSON from response, returning raw text"
                    )
            elif self.verbose:
                logger.debug("No valid tool calls found, returning text response")
            parts.append(TextPart(content=content))

        if self.verbose:
            logger.debug("Response parts: %d", len(parts))
            for i, part in enumerate(parts):
                if isinstance(part, TextPart):
                    logger.debug("  Part %d: TextPart (%d chars)", i, len(part.content))
                elif isinstance(part, ToolCallPart):
                    logger.debug("  Part %d: ToolCallPart(%s)", i, part.tool_name)
            logger.debug(
                "Request #%d completed in %.1fs (total: %d requests, %.1fs)",
                request_num,
                result.elapsed_seconds,
                self._request_count,
                self._total_time,
            )

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
        lines = [
            "You have access to the following tools. When you need to use a tool, "
            "you MUST output the tool call using EXACTLY this format:",
            "",
            'TOOL_CALL: tool_name({"param": "value"})',
            "",
            "IMPORTANT RULES:",
            "1. The tool call must be on its own line",
            "2. You can include brief text BEFORE the tool call to explain your reasoning",
            "3. After outputting TOOL_CALL, you MUST STOP IMMEDIATELY",
            "4. Do NOT simulate or imagine what the tool response might be",
            "5. Do NOT continue the conversation after TOOL_CALL",
            "6. Just output the TOOL_CALL line and nothing after it",
            "",
            "Available tools:",
            "",
        ]
        for tool in tools:
            lines.append(f"### {tool.name}")
            lines.append(f"Description: {tool.description}")
            params = tool.parameters_json_schema
            if params.get("properties"):
                lines.append("Parameters:")
                for pname, pinfo in params["properties"].items():
                    required = pname in params.get("required", [])
                    req_str = " (required)" if required else " (optional)"
                    ptype = pinfo.get("type", "any")
                    pdesc = pinfo.get("description", "")
                    lines.append(f"  - {pname}: {ptype}{req_str} - {pdesc}")
            else:
                lines.append("Parameters: none")
            # Add example
            example_args: dict[str, Any] = {}
            for pname, pinfo in params.get("properties", {}).items():
                if pinfo.get("type") == "string":
                    example_args[pname] = f"<{pname}>"
                elif pinfo.get("type") == "number":
                    example_args[pname] = 0
                elif pinfo.get("type") == "boolean":
                    example_args[pname] = True
            lines.append(f"Example: TOOL_CALL: {tool.name}({json.dumps(example_args)})")
            lines.append("")
        return "\n".join(lines)

    def _format_result_schema(self, tools: list[ToolDefinition]) -> str:
        """Format result schema for prompt."""
        if not tools:
            return ""
        tool = tools[0]  # Use first output tool
        lines = [
            "You MUST respond with a valid JSON object matching this schema.",
            "Output ONLY the JSON - no markdown code blocks, no explanation before or after.",
            "If you need to use a tool first, do that instead of returning JSON.",
            "",
            "Schema:",
            json.dumps(tool.parameters_json_schema, indent=2),
        ]
        return "\n".join(lines)


# Keep ClaudeCodeAgentModel for backwards compatibility if needed
@dataclass
class ClaudeCodeAgentModel:
    """Agent model for handling message conversion and CLI requests."""

    cli: ClaudeCodeCLI
    function_tools: list[ToolDefinition] = field(default_factory=list)
    allow_text_output: bool = True
    output_tools: list[ToolDefinition] = field(default_factory=list)

    def _build_prompt(self, messages: list[ModelMessage]) -> str:
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

        if self.function_tools:
            tools_text = self._format_tools(self.function_tools)
            sections.append("## Available Tools\n" + tools_text)

        if self.output_tools:
            schema_text = self._format_output_schema(self.output_tools)
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

    def _format_output_schema(self, tools: list[ToolDefinition]) -> str:
        """Format output schema for prompt."""
        if not tools:
            return ""
        tool = tools[0]  # Use first output tool
        return f"Return JSON matching this schema:\n{json.dumps(tool.parameters_json_schema, indent=2)}"
