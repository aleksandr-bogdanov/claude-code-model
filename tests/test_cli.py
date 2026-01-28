"""Tests for CLI wrapper."""
from __future__ import annotations

import pytest

from pathlib import Path
from unittest.mock import patch

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
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI()
            assert cli._executable == Path("/usr/bin/claude")

    def test_raises_when_not_found(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value=None):
            with pytest.raises(ClaudeCodeNotFoundError) as exc_info:
                ClaudeCodeCLI()
            assert "not found" in str(exc_info.value).lower()

    def test_default_model_is_sonnet(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI()
            assert cli.model == "sonnet"

    def test_custom_model(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI(model="opus")
            assert cli.model == "opus"

    def test_default_timeout_300(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI()
            assert cli.timeout == 300

    def test_custom_timeout(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI(timeout=60)
            assert cli.timeout == 60

    def test_default_cwd_is_none(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI()
            assert cli.cwd is None

    def test_custom_cwd(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI(cwd=Path("/tmp/project"))
            assert cli.cwd == Path("/tmp/project")
