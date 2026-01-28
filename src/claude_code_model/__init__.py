"""Claude Code Model - Pydantic AI adapter for Claude Code CLI."""

from claude_code_model.cli import (
    CLIResult,
    ClaudeCodeCLI,
    ClaudeCodeError,
    ClaudeCodeExecutionError,
    ClaudeCodeNotFoundError,
    ClaudeCodeTimeoutError,
)
from claude_code_model.model import ClaudeCodeModel

__all__ = [
    # Main class
    "ClaudeCodeModel",
    # CLI
    "ClaudeCodeCLI",
    "CLIResult",
    # Exceptions
    "ClaudeCodeError",
    "ClaudeCodeNotFoundError",
    "ClaudeCodeTimeoutError",
    "ClaudeCodeExecutionError",
]
__version__ = "0.1.0"
