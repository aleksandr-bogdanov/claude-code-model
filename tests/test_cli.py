"""Tests for CLI wrapper."""
from __future__ import annotations

import pytest

from claude_code_model.cli import (
    CLIResult,
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
