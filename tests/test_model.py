"""Tests for Pydantic AI Model adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from claude_code_model.cli import ClaudeCodeCLI
from claude_code_model.model import ClaudeCodeAgentModel, ClaudeCodeModel


class TestClaudeCodeModel:
    """Test ClaudeCodeModel class."""

    @pytest.fixture
    def model(self) -> ClaudeCodeModel:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            return ClaudeCodeModel()

    def test_model_name_default(self, model: ClaudeCodeModel) -> None:
        assert model.model_name == "claude-code:sonnet"

    def test_model_name_with_opus(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            model = ClaudeCodeModel(model="opus")
        assert model.model_name == "claude-code:opus"

    def test_model_name_with_haiku(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            model = ClaudeCodeModel(model="haiku")
        assert model.model_name == "claude-code:haiku"

    def test_system_property(self, model: ClaudeCodeModel) -> None:
        assert model.system == "claude-code"

    def test_default_timeout(self, model: ClaudeCodeModel) -> None:
        assert model.timeout == 300

    def test_custom_timeout(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            model = ClaudeCodeModel(timeout=60)
        assert model.timeout == 60

    def test_cli_created(self, model: ClaudeCodeModel) -> None:
        assert isinstance(model._cli, ClaudeCodeCLI)

    def test_cli_inherits_config(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
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

        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
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

        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
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


class TestBuildPrompt:
    """Test _build_prompt method."""

    @pytest.fixture
    def agent_model(self) -> ClaudeCodeAgentModel:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI()
        return ClaudeCodeAgentModel(
            cli=cli,
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        )

    def test_system_prompt_in_instructions(
        self, agent_model: ClaudeCodeAgentModel
    ) -> None:
        from pydantic_ai.messages import ModelRequest, SystemPromptPart

        messages = [ModelRequest(parts=[SystemPromptPart(content="Be helpful.")])]
        prompt = agent_model._build_prompt(messages)
        assert "## Instructions" in prompt
        assert "Be helpful." in prompt

    def test_multiple_system_prompts_combined(
        self, agent_model: ClaudeCodeAgentModel
    ) -> None:
        from pydantic_ai.messages import ModelRequest, SystemPromptPart

        messages = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="Rule 1"),
                    SystemPromptPart(content="Rule 2"),
                ]
            )
        ]
        prompt = agent_model._build_prompt(messages)
        assert "Rule 1" in prompt
        assert "Rule 2" in prompt

    def test_user_prompt_formatted(self, agent_model: ClaudeCodeAgentModel) -> None:
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        prompt = agent_model._build_prompt(messages)
        assert "User: Hello" in prompt

    def test_assistant_text_formatted(self, agent_model: ClaudeCodeAgentModel) -> None:
        from pydantic_ai.messages import ModelResponse, TextPart

        messages = [ModelResponse(parts=[TextPart(content="Hi there")])]
        prompt = agent_model._build_prompt(messages)
        assert "Assistant: Hi there" in prompt

    def test_tool_call_formatted(self, agent_model: ClaudeCodeAgentModel) -> None:
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        messages = [
            ModelResponse(parts=[ToolCallPart(tool_name="search", args={"q": "test"})])
        ]
        prompt = agent_model._build_prompt(messages)
        assert "Assistant called: search" in prompt
        assert '"q"' in prompt

    def test_tool_return_formatted(self, agent_model: ClaudeCodeAgentModel) -> None:
        from pydantic_ai.messages import ModelRequest, ToolReturnPart

        messages = [
            ModelRequest(
                parts=[ToolReturnPart(tool_name="search", content="found 3 results")]
            )
        ]
        prompt = agent_model._build_prompt(messages)
        assert "Tool Result (search):" in prompt
        assert "found 3 results" in prompt

    def test_tools_section_when_tools_provided(self) -> None:
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        from pydantic_ai.tools import ToolDefinition

        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI()
        tool = ToolDefinition(
            name="search",
            description="Search for files",
            parameters_json_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        agent_model = ClaudeCodeAgentModel(
            cli=cli,
            function_tools=[tool],
            allow_text_output=True,
            output_tools=[],
        )
        messages = [ModelRequest(parts=[UserPromptPart(content="Find files")])]
        prompt = agent_model._build_prompt(messages)
        assert "## Available Tools" in prompt
        assert "search" in prompt
        assert "Search for files" in prompt

    def test_result_schema_section_when_result_tools(self) -> None:
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        from pydantic_ai.tools import ToolDefinition

        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI()
        output_tool = ToolDefinition(
            name="final_result",
            description="Return structured output",
            parameters_json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        )
        agent_model = ClaudeCodeAgentModel(
            cli=cli,
            function_tools=[],
            allow_text_output=False,
            output_tools=[output_tool],
        )
        messages = [ModelRequest(parts=[UserPromptPart(content="Analyze")])]
        prompt = agent_model._build_prompt(messages)
        assert "## Required Output Format" in prompt

    def test_conversation_section_present(
        self, agent_model: ClaudeCodeAgentModel
    ) -> None:
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        prompt = agent_model._build_prompt(messages)
        assert "## Conversation" in prompt

    def test_full_conversation_flow(self, agent_model: ClaudeCodeAgentModel) -> None:
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            SystemPromptPart,
            TextPart,
            UserPromptPart,
        )

        messages = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="You are helpful."),
                    UserPromptPart(content="What is 2+2?"),
                ]
            ),
            ModelResponse(parts=[TextPart(content="4")]),
            ModelRequest(parts=[UserPromptPart(content="Thanks!")]),
        ]
        prompt = agent_model._build_prompt(messages)
        assert "You are helpful." in prompt
        assert "User: What is 2+2?" in prompt
        assert "Assistant: 4" in prompt
        assert "User: Thanks!" in prompt
