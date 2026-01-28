"""Tests for CLI wrapper."""

from __future__ import annotations

import asyncio
import pytest

from pathlib import Path
from unittest.mock import AsyncMock, patch

from claude_code_model.cli import (
    CLIResult,
    ClaudeCodeCLI,
    ClaudeCodeError,
    ClaudeCodeExecutionError,
    ClaudeCodeNotFoundError,
    ClaudeCodeTimeoutError,
)


class TestExceptions:
    """Test exception hierarchy."""

    def test_base_error_is_exception(self) -> None:
        assert issubclass(ClaudeCodeError, Exception)

    def test_not_found_inherits_from_base(self) -> None:
        assert issubclass(ClaudeCodeNotFoundError, ClaudeCodeError)

    def test_timeout_inherits_from_base(self) -> None:
        assert issubclass(ClaudeCodeTimeoutError, ClaudeCodeError)

    def test_execution_inherits_from_base(self) -> None:
        assert issubclass(ClaudeCodeExecutionError, ClaudeCodeError)

    def test_exception_message_preserved(self) -> None:
        msg = "CLI not found in PATH"
        err = ClaudeCodeNotFoundError(msg)
        assert msg in str(err)

    def test_exception_can_be_raised_and_caught(self) -> None:
        with pytest.raises(ClaudeCodeError):
            raise ClaudeCodeNotFoundError("test")


class TestCLIResult:
    """Test CLIResult dataclass."""

    def test_success_when_exit_code_zero(self) -> None:
        result = CLIResult(stdout="output", stderr="", exit_code=0)
        assert result.success is True

    def test_failure_when_exit_code_nonzero(self) -> None:
        result = CLIResult(stdout="", stderr="error", exit_code=1)
        assert result.success is False

    def test_failure_when_exit_code_negative(self) -> None:
        result = CLIResult(stdout="", stderr="killed", exit_code=-9)
        assert result.success is False

    def test_attributes_accessible(self) -> None:
        result = CLIResult(stdout="out", stderr="err", exit_code=0)
        assert result.stdout == "out"
        assert result.stderr == "err"
        assert result.exit_code == 0


class TestClaudeCodeCLIInit:
    """Test ClaudeCodeCLI initialization."""

    def test_finds_claude_executable(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI()
            assert cli._executable == Path("/usr/bin/claude")

    def test_raises_when_not_found(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value=None):
            with pytest.raises(ClaudeCodeNotFoundError) as exc_info:
                ClaudeCodeCLI()
            assert "not found" in str(exc_info.value).lower()

    def test_default_model_is_sonnet(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI()
            assert cli.model == "sonnet"

    def test_custom_model(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI(model="opus")
            assert cli.model == "opus"

    def test_default_timeout_300(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI()
            assert cli.timeout == 300

    def test_custom_timeout(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI(timeout=60)
            assert cli.timeout == 60

    def test_default_cwd_is_none(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI()
            assert cli.cwd is None

    def test_custom_cwd(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI(cwd=Path("/tmp/project"))
            assert cli.cwd == Path("/tmp/project")


class TestClaudeCodeCLIRun:
    """Test ClaudeCodeCLI.run() async method."""

    @pytest.fixture
    def cli(self) -> ClaudeCodeCLI:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            return ClaudeCodeCLI()

    @pytest.mark.asyncio
    async def test_successful_run_returns_result(self, cli: ClaudeCodeCLI) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'{"status": "ok"}', b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await cli.run("test prompt")

        assert result.stdout == '{"status": "ok"}'
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.success is True

    @pytest.mark.asyncio
    async def test_command_includes_prompt(self, cli: ClaudeCodeCLI) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"out", b"")
        mock_proc.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            await cli.run("my prompt")

        args = mock_exec.call_args[0]
        assert "-p" in args
        assert "my prompt" in args

    @pytest.mark.asyncio
    async def test_command_includes_model_flag(self, cli: ClaudeCodeCLI) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"out", b"")
        mock_proc.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            await cli.run("test")

        args = mock_exec.call_args[0]
        assert "--model" in args
        assert "sonnet" in args

    @pytest.mark.asyncio
    async def test_custom_model_in_command(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI(model="opus")

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"out", b"")
        mock_proc.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            await cli.run("test")

        args = mock_exec.call_args[0]
        assert "opus" in args

    @pytest.mark.asyncio
    async def test_timeout_raises_exception(self, cli: ClaudeCodeCLI) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(ClaudeCodeTimeoutError) as exc_info:
                await cli.run("test")
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_exception(self, cli: ClaudeCodeCLI) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"error occurred")
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(ClaudeCodeExecutionError) as exc_info:
                await cli.run("test")
        assert "error occurred" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cwd_passed_to_subprocess(self) -> None:
        with patch(
            "claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"
        ):
            cli = ClaudeCodeCLI(cwd=Path("/tmp/project"))

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"out", b"")
        mock_proc.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            await cli.run("test")

        kwargs = mock_exec.call_args[1]
        assert kwargs["cwd"] == Path("/tmp/project")

    @pytest.mark.asyncio
    async def test_uses_pipe_for_stdout_stderr(self, cli: ClaudeCodeCLI) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"out", b"")
        mock_proc.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_proc
        ) as mock_exec:
            await cli.run("test")

        kwargs = mock_exec.call_args[1]
        import asyncio.subprocess

        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
