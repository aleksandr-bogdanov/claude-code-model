"""CLI wrapper for Claude Code."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


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


@dataclass
class ClaudeCodeCLI:
    """Wrapper for Claude Code CLI."""

    model: str = "sonnet"
    timeout: int = 300
    cwd: Path | None = None
    _executable: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Find and validate claude executable."""
        claude_path = shutil.which("claude")
        if claude_path is None:
            raise ClaudeCodeNotFoundError(
                "claude CLI not found. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )
        self._executable = Path(claude_path)
