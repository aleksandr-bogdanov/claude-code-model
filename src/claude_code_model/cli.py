"""CLI wrapper for Claude Code."""
from __future__ import annotations

from dataclasses import dataclass


class ClaudeCodeError(Exception):
    """Base exception for claude-code-model."""

    pass


class ClaudeCodeNotFoundError(ClaudeCodeError):
    """Raised when claude CLI executable is not found."""

    pass


class ClaudeCodeTimeoutError(ClaudeCodeError):
    """Raised when CLI command times out."""

    pass


class ClaudeCodeExecutionError(ClaudeCodeError):
    """Raised when CLI returns non-zero exit code."""

    pass


@dataclass
class CLIResult:
    """Result from CLI execution."""

    stdout: str
    stderr: str
    exit_code: int

    @property
    def success(self) -> bool:
        """Return True if command exited with code 0."""
        return self.exit_code == 0
