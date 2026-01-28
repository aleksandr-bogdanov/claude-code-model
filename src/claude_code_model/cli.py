"""CLI wrapper for Claude Code."""

from __future__ import annotations

import asyncio
from asyncio import subprocess as async_subprocess
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


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
    elapsed_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """Return True if command exited with code 0."""
        return self.exit_code == 0


@dataclass
class ClaudeCodeCLI:
    """Wrapper for Claude Code CLI."""

    model: str = "sonnet"
    timeout: int = 30  # Default 30 seconds per request
    cwd: Path | None = None
    verbose: bool = False
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
        if self.verbose:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
                )
                logger.addHandler(handler)

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
            "--no-session-persistence",  # Prevent conversation bleed-through
        ]

        if self.verbose:
            logger.debug("=" * 60)
            logger.debug("CLAUDE CLI REQUEST")
            logger.debug("=" * 60)
            logger.debug("Command: claude -p <prompt> --model %s", self.model)
            logger.debug("Timeout: %d seconds", self.timeout)
            logger.debug("Prompt (%d chars):\n%s", len(prompt), prompt[:2000])
            if len(prompt) > 2000:
                logger.debug("... (truncated, total %d chars)", len(prompt))
            logger.debug("-" * 60)

        start_time = time.perf_counter()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=async_subprocess.DEVNULL,  # Prevent hang when no TTY
                stdout=async_subprocess.PIPE,
                stderr=async_subprocess.PIPE,
                cwd=self.cwd,
            )

            if self.verbose:
                logger.debug(
                    "Process started (PID: %s), waiting for response...", proc.pid
                )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as e:
            elapsed = time.perf_counter() - start_time
            logger.error("CLI timeout after %.1fs (limit: %ds)", elapsed, self.timeout)
            raise ClaudeCodeTimeoutError(
                f"CLI timeout after {self.timeout} seconds"
            ) from e
        except asyncio.CancelledError:
            elapsed = time.perf_counter() - start_time
            logger.warning("CLI request was cancelled after %.1fs", elapsed)
            raise

        elapsed = time.perf_counter() - start_time
        stdout = stdout_bytes.decode() if stdout_bytes else ""
        stderr = stderr_bytes.decode() if stderr_bytes else ""

        if self.verbose:
            logger.debug("=" * 60)
            logger.debug("CLAUDE CLI RESPONSE (took %.1fs)", elapsed)
            logger.debug("=" * 60)
            logger.debug("Exit code: %s", proc.returncode)
            logger.debug("Stdout (%d chars):\n%s", len(stdout), stdout[:2000])
            if len(stdout) > 2000:
                logger.debug("... (truncated, total %d chars)", len(stdout))
            if stderr:
                logger.debug("Stderr: %s", stderr)
            logger.debug("-" * 60)

        if proc.returncode != 0:
            logger.error("CLI exited with code %s: %s", proc.returncode, stderr)
            raise ClaudeCodeExecutionError(
                f"CLI exited with code {proc.returncode}: {stderr}"
            )

        return CLIResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode or 0,
            elapsed_seconds=elapsed,
        )
