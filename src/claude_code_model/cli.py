"""CLI wrapper for Claude Code."""
from __future__ import annotations

import asyncio
from asyncio import subprocess as async_subprocess
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

    async def run(self, prompt: str) -> CLIResult:
        """
        Execute claude CLI with prompt and return result.

        Args:
            prompt: The prompt to send to Claude

        Returns:
            CLIResult with stdout, stderr, exit_code

        Raises:
            ClaudeCodeTimeoutError: If command times out
            ClaudeCodeExecutionError: If command exits with non-zero code
        """
        cmd = [
            str(self._executable),
            "-p",
            prompt,
            "--model",
            self.model,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=async_subprocess.PIPE,
                stderr=async_subprocess.PIPE,
                cwd=self.cwd,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as e:
            raise ClaudeCodeTimeoutError(
                f"CLI timeout after {self.timeout} seconds"
            ) from e

        stdout = stdout_bytes.decode() if stdout_bytes else ""
        stderr = stderr_bytes.decode() if stderr_bytes else ""

        if proc.returncode != 0:
            raise ClaudeCodeExecutionError(
                f"CLI exited with code {proc.returncode}: {stderr}"
            )

        return CLIResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode or 0,
        )
