"""Tests for Pydantic AI Model adapter."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from claude_code_model.cli import ClaudeCodeCLI
from claude_code_model.model import ClaudeCodeModel


class TestClaudeCodeModel:
    """Test ClaudeCodeModel class."""

    @pytest.fixture
    def model(self) -> ClaudeCodeModel:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            return ClaudeCodeModel()

    def test_model_name_default(self, model: ClaudeCodeModel) -> None:
        assert model.model_name == "claude-code:sonnet"

    def test_model_name_with_opus(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            model = ClaudeCodeModel(model="opus")
        assert model.model_name == "claude-code:opus"

    def test_model_name_with_haiku(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            model = ClaudeCodeModel(model="haiku")
        assert model.model_name == "claude-code:haiku"

    def test_system_property(self, model: ClaudeCodeModel) -> None:
        assert model.system == "claude-code"

    def test_default_timeout(self, model: ClaudeCodeModel) -> None:
        assert model.timeout == 300

    def test_custom_timeout(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            model = ClaudeCodeModel(timeout=60)
        assert model.timeout == 60

    def test_cli_created(self, model: ClaudeCodeModel) -> None:
        assert isinstance(model._cli, ClaudeCodeCLI)

    def test_cli_inherits_config(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            model = ClaudeCodeModel(model="opus", timeout=120, cwd=Path("/tmp"))
        assert model._cli.model == "opus"
        assert model._cli.timeout == 120
        assert model._cli.cwd == Path("/tmp")

    @pytest.mark.asyncio
    async def test_request_returns_text_response(self, model: ClaudeCodeModel) -> None:
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        from pydantic_ai.models import ModelRequestParameters

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"Hello, world!", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]
            params = ModelRequestParameters(
                function_tools=[],
                output_tools=None,
                allow_text_output=True,
                output_mode="text",
            )
            response = await model.request(messages, None, params)

        assert len(response.parts) == 1
        from pydantic_ai.messages import TextPart

        assert isinstance(response.parts[0], TextPart)
        assert response.parts[0].content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_supported_builtin_tools(self) -> None:
        assert ClaudeCodeModel.supported_builtin_tools() == frozenset()

    @pytest.mark.asyncio
    async def test_request_returns_usage_with_zero_tokens(
        self, model: ClaudeCodeModel
    ) -> None:
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        from pydantic_ai.models import ModelRequestParameters

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"response", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]
            params = ModelRequestParameters(
                function_tools=[],
                output_tools=None,
                allow_text_output=True,
                output_mode="text",
            )
            response = await model.request(messages, None, params)

        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_detects_tool_call_in_response(self) -> None:
        from pydantic_ai.messages import ModelRequest, ToolCallPart, UserPromptPart
        from pydantic_ai.models import ModelRequestParameters
        from pydantic_ai.tools import ToolDefinition

        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            model = ClaudeCodeModel()

        tool = ToolDefinition(
            name="search",
            description="Search",
            parameters_json_schema={"type": "object"},
        )

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b'TOOL_CALL: search({"query": "test"})',
            b"",
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Search")])]
            params = ModelRequestParameters(
                function_tools=[tool],
                output_tools=None,
                allow_text_output=True,
                output_mode="text",
            )
            response = await model.request(messages, None, params)

        assert len(response.parts) == 1
        assert isinstance(response.parts[0], ToolCallPart)
        assert response.parts[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_ignores_unknown_tool_call(self, model: ClaudeCodeModel) -> None:
        from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart
        from pydantic_ai.models import ModelRequestParameters

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'TOOL_CALL: unknown_tool({"a": 1})', b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Test")])]
            params = ModelRequestParameters(
                function_tools=[],
                output_tools=None,
                allow_text_output=True,
                output_mode="text",
            )
            response = await model.request(messages, None, params)

        # Should fall back to text since tool not in function_tools
        assert len(response.parts) == 1
        assert isinstance(response.parts[0], TextPart)

    @pytest.mark.asyncio
    async def test_handles_multiple_tool_calls(self) -> None:
        from pydantic_ai.messages import ModelRequest, ToolCallPart, UserPromptPart
        from pydantic_ai.models import ModelRequestParameters
        from pydantic_ai.tools import ToolDefinition

        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            model = ClaudeCodeModel()

        tools = [
            ToolDefinition(name="tool1", description="T1", parameters_json_schema={}),
            ToolDefinition(name="tool2", description="T2", parameters_json_schema={}),
        ]

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b'TOOL_CALL: tool1({"a": 1})\nTOOL_CALL: tool2({"b": 2})',
            b"",
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Test")])]
            params = ModelRequestParameters(
                function_tools=tools,
                output_tools=None,
                allow_text_output=True,
                output_mode="text",
            )
            response = await model.request(messages, None, params)

        assert len(response.parts) == 2
        assert all(isinstance(p, ToolCallPart) for p in response.parts)

    @pytest.mark.asyncio
    async def test_returns_model_response_type(self, model: ClaudeCodeModel) -> None:
        from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
        from pydantic_ai.models import ModelRequestParameters

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"test", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]
            params = ModelRequestParameters(
                function_tools=[],
                output_tools=None,
                allow_text_output=True,
                output_mode="text",
            )
            response = await model.request(messages, None, params)

        assert isinstance(response, ModelResponse)
