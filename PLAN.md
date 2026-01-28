# PLAN.md — Parallel TDD Execution

## Overview

This plan enables **parallel execution** using multiple Claude Code instances.
Each wave contains independent tasks that can run simultaneously.

**Model**: Sonnet for implementation, Haiku for reviews (cost-efficient)

---

## Dependency Graph

```
Wave 1 (Foundation - parallel)
├── Task A: Project Setup
├── Task B: Exceptions + CLIResult (tests + impl)
└── Task C: JSON Extraction (tests + impl)

Wave 2 (depends on Wave 1 - parallel)
├── Task D: CLI Init (tests + impl) [needs B]
├── Task E: Tool Call Parsing (tests + impl) [needs C]
└── Task F: (merge A+B+C)

Wave 3 (depends on Wave 2 - parallel)
├── Task G: CLI Run (tests + impl) [needs D]
├── Task H: ClaudeCodeModel class (tests + impl) [needs D]
└── Task I: (merge D+E)

Wave 4 (depends on Wave 3 - parallel)
├── Task J: Message Conversion (tests + impl) [needs H]
├── Task K: Request Method (tests + impl) [needs G+H]
└── Task L: (merge G+H)

Wave 5 (depends on Wave 4 - parallel)
├── Task M: Package Exports [needs all]
├── Task N: Simple Example [needs M]
├── Task O: Tools Example [needs M]
└── Task P: Multi-Agent Example [needs M]

Wave 6 (Final)
└── Task Q: README + Polish
```

---

## Wave 1: Foundation (3 parallel agents)

### Agent 1A: Project Setup
**Branch**: `wave1/setup`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task A for claude-code-model project.

## Task: Project Setup

Create the project skeleton with these files:

1. `pyproject.toml`:
```toml
[project]
name = "claude-code-model"
version = "0.1.0"
description = "Pydantic AI model adapter for Claude Code CLI"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
dependencies = [
    "pydantic-ai>=0.1.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.mypy]
strict = true
python_version = "3.11"

[tool.ruff]
target-version = "py311"
line-length = 88
```

2. `src/claude_code_model/__init__.py`:
```python
"""Claude Code Model - Pydantic AI adapter for Claude Code CLI."""
__version__ = "0.1.0"
```

3. `src/claude_code_model/cli.py`:
```python
"""CLI wrapper for Claude Code."""
```

4. `src/claude_code_model/model.py`:
```python
"""Pydantic AI Model implementation."""
```

5. `src/claude_code_model/py.typed` (empty file)

6. `tests/__init__.py` (empty file)

7. `tests/conftest.py`:
```python
"""Pytest configuration."""
import pytest
```

## Verification
Run: `uv sync && uv run pytest --collect-only`
Should show 0 tests collected, no errors.

## Commit
Message: "chore: initialize project structure"
```

---

### Agent 1B: Exceptions + CLIResult
**Branch**: `wave1/cli-exceptions`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task B for claude-code-model project.

## Task: Exceptions and CLIResult with TDD

### Step 1: Write Tests First

Create `tests/test_cli.py`:

```python
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
```

### Step 2: Verify Tests Fail
Run: `uv run pytest tests/test_cli.py -v`
Should fail with ImportError (expected - Red phase).

### Step 3: Implement to Pass Tests

Update `src/claude_code_model/cli.py`:

```python
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
```

### Step 4: Verify Tests Pass
Run: `uv run pytest tests/test_cli.py -v`
All tests should pass (Green phase).

Run: `uv run mypy src/claude_code_model/cli.py`
Should pass with no errors.

## Commit
Message: "feat: add CLI exceptions and CLIResult dataclass"
```

---

### Agent 1C: JSON Extraction
**Branch**: `wave1/json-parsing`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task C for claude-code-model project.

## Task: JSON Extraction with TDD

### Step 1: Write Tests First

Create `tests/test_parsing.py`:

```python
"""Tests for response parsing utilities."""
from __future__ import annotations

import pytest

from claude_code_model.model import extract_json


class TestExtractJson:
    """Test JSON extraction from various formats."""

    # Pure JSON
    def test_pure_json_object(self) -> None:
        assert extract_json('{"key": "value"}') == {"key": "value"}

    def test_pure_json_with_leading_whitespace(self) -> None:
        assert extract_json('  {"key": "value"}') == {"key": "value"}

    def test_pure_json_with_trailing_whitespace(self) -> None:
        assert extract_json('{"key": "value"}  \n') == {"key": "value"}

    # Markdown code blocks
    def test_markdown_json_block(self) -> None:
        content = '```json\n{"key": "value"}\n```'
        assert extract_json(content) == {"key": "value"}

    def test_markdown_block_no_language(self) -> None:
        content = '```\n{"key": "value"}\n```'
        assert extract_json(content) == {"key": "value"}

    def test_markdown_block_with_surrounding_text(self) -> None:
        content = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        assert extract_json(content) == {"key": "value"}

    # Embedded JSON
    def test_embedded_json_in_text(self) -> None:
        content = 'The answer is {"key": "value"} as requested.'
        assert extract_json(content) == {"key": "value"}

    def test_embedded_json_at_end(self) -> None:
        content = 'Result: {"status": "ok"}'
        assert extract_json(content) == {"status": "ok"}

    # Nested structures
    def test_nested_objects(self) -> None:
        content = '{"outer": {"inner": "value"}}'
        assert extract_json(content) == {"outer": {"inner": "value"}}

    def test_deeply_nested(self) -> None:
        content = '{"a": {"b": {"c": {"d": 1}}}}'
        assert extract_json(content) == {"a": {"b": {"c": {"d": 1}}}}

    def test_json_with_array(self) -> None:
        content = '{"items": [1, 2, 3]}'
        assert extract_json(content) == {"items": [1, 2, 3]}

    def test_json_with_nested_array_of_objects(self) -> None:
        content = '{"users": [{"name": "a"}, {"name": "b"}]}'
        assert extract_json(content) == {"users": [{"name": "a"}, {"name": "b"}]}

    # Edge cases
    def test_empty_string_returns_none(self) -> None:
        assert extract_json("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert extract_json("   \n\t  ") is None

    def test_no_json_returns_none(self) -> None:
        assert extract_json("Just plain text without JSON") is None

    def test_invalid_json_returns_none(self) -> None:
        assert extract_json("{invalid json}") is None

    def test_unclosed_brace_returns_none(self) -> None:
        assert extract_json('{"key": "value"') is None

    def test_json_array_at_root_returns_none(self) -> None:
        # We only extract objects, not arrays at root level
        assert extract_json("[1, 2, 3]") is None

    def test_json_with_special_chars(self) -> None:
        content = '{"msg": "hello\\nworld"}'
        assert extract_json(content) == {"msg": "hello\nworld"}

    def test_json_with_unicode(self) -> None:
        content = '{"emoji": "🎉"}'
        assert extract_json(content) == {"emoji": "🎉"}
```

### Step 2: Verify Tests Fail
Run: `uv run pytest tests/test_parsing.py -v`
Should fail with ImportError (expected - Red phase).

### Step 3: Implement to Pass Tests

Update `src/claude_code_model/model.py`:

```python
"""Pydantic AI Model implementation."""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(content: str) -> dict[str, Any] | None:
    """
    Extract JSON object from response content.

    Tries in order:
    1. Direct JSON parse
    2. Markdown code block
    3. Embedded {...} in text

    Args:
        content: Raw response string

    Returns:
        Parsed dict or None if no valid JSON object found
    """
    content = content.strip()
    if not content:
        return None

    # Try 1: Direct parse
    result = _try_parse_json(content)
    if result is not None:
        return result

    # Try 2: Markdown code block
    block = _extract_code_block(content)
    if block:
        result = _try_parse_json(block)
        if result is not None:
            return result

    # Try 3: Find embedded JSON object
    obj_str = _find_json_object(content)
    if obj_str:
        result = _try_parse_json(obj_str)
        if result is not None:
            return result

    return None


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Try to parse text as JSON, return dict or None."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _extract_code_block(content: str) -> str | None:
    """Extract content from ```json ... ``` or ``` ... ``` block."""
    pattern = r"```(?:json)?\s*\n(.*?)\n```"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _find_json_object(content: str) -> str | None:
    """Find balanced {...} in content."""
    start = content.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(content[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : i + 1]

    return None
```

### Step 4: Verify Tests Pass
Run: `uv run pytest tests/test_parsing.py -v`
All tests should pass (Green phase).

Run: `uv run mypy src/claude_code_model/model.py`
Should pass with no errors.

## Commit
Message: "feat: add JSON extraction utilities"
```

---

## Wave 2: CLI Init + Tool Parsing (2 parallel agents)

**Prerequisites**: Merge Wave 1 branches first, or rebase on them.

### Agent 2D: CLI Init
**Branch**: `wave2/cli-init`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task D for claude-code-model project.

## Prerequisites
Ensure wave1/cli-exceptions is merged or rebase on it.

## Task: ClaudeCodeCLI Initialization with TDD

### Step 1: Write Tests First

Append to `tests/test_cli.py`:

```python
from pathlib import Path
from unittest.mock import patch

from claude_code_model.cli import ClaudeCodeCLI


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
```

### Step 2: Verify New Tests Fail
Run: `uv run pytest tests/test_cli.py::TestClaudeCodeCLIInit -v`
Should fail (Red phase).

### Step 3: Implement to Pass Tests

Update `src/claude_code_model/cli.py`, add after CLIResult:

```python
import shutil
from dataclasses import dataclass, field
from pathlib import Path


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
```

Update imports at top of file:
```python
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
```

### Step 4: Verify Tests Pass
Run: `uv run pytest tests/test_cli.py -v`
All tests should pass.

Run: `uv run mypy src/claude_code_model/cli.py`
Should pass.

## Commit
Message: "feat: add ClaudeCodeCLI initialization"
```

---

### Agent 2E: Tool Call Parsing
**Branch**: `wave2/tool-parsing`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task E for claude-code-model project.

## Prerequisites
Ensure wave1/json-parsing is merged or rebase on it.

## Task: Tool Call Detection with TDD

### Step 1: Write Tests First

Append to `tests/test_parsing.py`:

```python
from claude_code_model.model import ParsedToolCall, extract_tool_calls


class TestExtractToolCalls:
    """Test tool call extraction."""

    # TOOL_CALL format
    def test_tool_call_format_basic(self) -> None:
        content = 'TOOL_CALL: search({"query": "python"})'
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].args == {"query": "python"}

    def test_tool_call_with_surrounding_text(self) -> None:
        content = 'I will search.\nTOOL_CALL: search({"q": "test"})\nDone.'
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].args == {"q": "test"}

    def test_tool_call_complex_args(self) -> None:
        content = 'TOOL_CALL: create({"name": "test", "count": 5, "active": true})'
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].args == {"name": "test", "count": 5, "active": True}

    # XML format
    def test_xml_format_basic(self) -> None:
        content = '<tool_call name="search">{"query": "python"}</tool_call>'
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].args == {"query": "python"}

    def test_xml_format_with_whitespace(self) -> None:
        content = '''<tool_call name="search">
            {"query": "test"}
        </tool_call>'''
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].args == {"query": "test"}

    # Multiple calls
    def test_multiple_tool_calls_same_format(self) -> None:
        content = '''TOOL_CALL: first({"a": 1})
        TOOL_CALL: second({"b": 2})'''
        calls = extract_tool_calls(content)
        assert len(calls) == 2
        assert calls[0].name == "first"
        assert calls[1].name == "second"

    def test_multiple_tool_calls_mixed_formats(self) -> None:
        content = '''TOOL_CALL: func1({"x": 1})
        <tool_call name="func2">{"y": 2}</tool_call>'''
        calls = extract_tool_calls(content)
        assert len(calls) == 2

    # Edge cases
    def test_no_tool_calls_returns_empty(self) -> None:
        assert extract_tool_calls("Just regular text") == []

    def test_empty_string_returns_empty(self) -> None:
        assert extract_tool_calls("") == []

    def test_invalid_json_skipped(self) -> None:
        content = 'TOOL_CALL: bad({not valid json})'
        assert extract_tool_calls(content) == []

    def test_partial_match_ignored(self) -> None:
        content = 'TOOL_CALL: incomplete('
        assert extract_tool_calls(content) == []

    def test_tool_call_with_nested_json(self) -> None:
        content = 'TOOL_CALL: update({"data": {"nested": "value"}})'
        calls = extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].args == {"data": {"nested": "value"}}


class TestParsedToolCall:
    """Test ParsedToolCall dataclass."""

    def test_attributes(self) -> None:
        tc = ParsedToolCall(name="test", args={"key": "value"})
        assert tc.name == "test"
        assert tc.args == {"key": "value"}
```

### Step 2: Verify New Tests Fail
Run: `uv run pytest tests/test_parsing.py::TestExtractToolCalls -v`
Should fail (Red phase).

### Step 3: Implement to Pass Tests

Append to `src/claude_code_model/model.py`:

```python
from dataclasses import dataclass


@dataclass
class ParsedToolCall:
    """Parsed tool call from model response."""

    name: str
    args: dict[str, Any]


def extract_tool_calls(content: str) -> list[ParsedToolCall]:
    """
    Extract tool calls from response content.

    Detects formats:
    1. TOOL_CALL: name({"arg": "val"})
    2. <tool_call name="name">{"arg": "val"}</tool_call>

    Args:
        content: Raw response string

    Returns:
        List of parsed tool calls (empty if none found)
    """
    calls: list[ParsedToolCall] = []

    # Pattern 1: TOOL_CALL: name({...})
    pattern1 = r'TOOL_CALL:\s*(\w+)\((\{.*?\})\)'
    for match in re.finditer(pattern1, content, re.DOTALL):
        name = match.group(1)
        args_str = match.group(2)
        args = _try_parse_json(args_str)
        if args is not None:
            calls.append(ParsedToolCall(name=name, args=args))

    # Pattern 2: <tool_call name="name">{...}</tool_call>
    pattern2 = r'<tool_call\s+name="(\w+)">(.*?)</tool_call>'
    for match in re.finditer(pattern2, content, re.DOTALL):
        name = match.group(1)
        args_str = match.group(2).strip()
        args = _try_parse_json(args_str)
        if args is not None:
            calls.append(ParsedToolCall(name=name, args=args))

    return calls
```

Update imports at top:
```python
from dataclasses import dataclass
```

### Step 4: Verify Tests Pass
Run: `uv run pytest tests/test_parsing.py -v`
All tests should pass.

## Commit
Message: "feat: add tool call extraction"
```

---

## Wave 3: CLI Run + Model Class (2 parallel agents)

### Agent 3G: CLI Run Method
**Branch**: `wave3/cli-run`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task G for claude-code-model project.

## Prerequisites
Ensure wave2/cli-init is merged or rebase on it.

## Task: ClaudeCodeCLI.run() Method with TDD

### Step 1: Write Tests First

Append to `tests/test_cli.py`:

```python
import asyncio
from unittest.mock import AsyncMock


class TestClaudeCodeCLIRun:
    """Test ClaudeCodeCLI.run() async method."""

    @pytest.fixture
    def cli(self) -> ClaudeCodeCLI:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            return ClaudeCodeCLI()

    @pytest.mark.asyncio
    async def test_successful_run_returns_result(self, cli: ClaudeCodeCLI) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'{"status": "ok"}', b'')
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
        mock_proc.communicate.return_value = (b'out', b'')
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await cli.run("my prompt")

        args = mock_exec.call_args[0]
        assert "-p" in args
        assert "my prompt" in args

    @pytest.mark.asyncio
    async def test_command_includes_model_flag(self, cli: ClaudeCodeCLI) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'out', b'')
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await cli.run("test")

        args = mock_exec.call_args[0]
        assert "--model" in args
        assert "sonnet" in args

    @pytest.mark.asyncio
    async def test_custom_model_in_command(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI(model="opus")

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'out', b'')
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
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
        mock_proc.communicate.return_value = (b'', b'error occurred')
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(ClaudeCodeExecutionError) as exc_info:
                await cli.run("test")
        assert "error occurred" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cwd_passed_to_subprocess(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI(cwd=Path("/tmp/project"))

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'out', b'')
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await cli.run("test")

        kwargs = mock_exec.call_args[1]
        assert kwargs["cwd"] == Path("/tmp/project")

    @pytest.mark.asyncio
    async def test_uses_pipe_for_stdout_stderr(self, cli: ClaudeCodeCLI) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'out', b'')
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await cli.run("test")

        kwargs = mock_exec.call_args[1]
        import asyncio.subprocess
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
```

### Step 2: Verify New Tests Fail
Run: `uv run pytest tests/test_cli.py::TestClaudeCodeCLIRun -v`

### Step 3: Implement to Pass Tests

Add to `ClaudeCodeCLI` class in `src/claude_code_model/cli.py`:

```python
import asyncio
from asyncio import subprocess as async_subprocess


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
            "-p", prompt,
            "--model", self.model,
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
                f"CLI timed out after {self.timeout} seconds"
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
```

Update imports:
```python
import asyncio
from asyncio import subprocess as async_subprocess
```

### Step 4: Verify Tests Pass
Run: `uv run pytest tests/test_cli.py -v`

## Commit
Message: "feat: add ClaudeCodeCLI.run() async method"
```

---

### Agent 3H: ClaudeCodeModel Class
**Branch**: `wave3/model-class`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task H for claude-code-model project.

## Prerequisites
Ensure wave2/cli-init and wave2/tool-parsing are merged.

## Task: ClaudeCodeModel with TDD

### Step 1: Write Tests First

Create `tests/test_model.py`:

```python
"""Tests for Pydantic AI Model adapter."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from claude_code_model.cli import ClaudeCodeCLI
from claude_code_model.model import ClaudeCodeAgentModel, ClaudeCodeModel


class TestClaudeCodeModel:
    """Test ClaudeCodeModel class."""

    @pytest.fixture
    def model(self) -> ClaudeCodeModel:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            return ClaudeCodeModel()

    def test_name_default(self, model: ClaudeCodeModel) -> None:
        assert model.name() == "claude-code:sonnet"

    def test_name_with_opus(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            model = ClaudeCodeModel(model="opus")
        assert model.name() == "claude-code:opus"

    def test_name_with_haiku(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            model = ClaudeCodeModel(model="haiku")
        assert model.name() == "claude-code:haiku"

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
    async def test_agent_model_returns_instance(self, model: ClaudeCodeModel) -> None:
        agent_model = await model.agent_model(
            function_tools=[],
            allow_text_result=True,
            result_tools=[],
        )
        assert isinstance(agent_model, ClaudeCodeAgentModel)

    @pytest.mark.asyncio
    async def test_agent_model_receives_function_tools(self, model: ClaudeCodeModel) -> None:
        from pydantic_ai.tools import ToolDefinition

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters_json_schema={"type": "object", "properties": {}},
        )
        agent_model = await model.agent_model(
            function_tools=[tool],
            allow_text_result=True,
            result_tools=[],
        )
        assert len(agent_model._function_tools) == 1
        assert agent_model._function_tools[0].name == "test_tool"

    @pytest.mark.asyncio
    async def test_agent_model_receives_result_tools(self, model: ClaudeCodeModel) -> None:
        from pydantic_ai.tools import ToolDefinition

        tool = ToolDefinition(
            name="result_tool",
            description="Result schema",
            parameters_json_schema={"type": "object"},
        )
        agent_model = await model.agent_model(
            function_tools=[],
            allow_text_result=False,
            result_tools=[tool],
        )
        assert len(agent_model._result_tools) == 1
```

### Step 2: Verify Tests Fail
Run: `uv run pytest tests/test_model.py -v`

### Step 3: Implement to Pass Tests

Append to `src/claude_code_model/model.py`:

```python
from pathlib import Path
from typing import Literal

from pydantic_ai.models import AgentModel, Model
from pydantic_ai.tools import ToolDefinition

from claude_code_model.cli import ClaudeCodeCLI


@dataclass
class ClaudeCodeModel(Model):
    """Pydantic AI Model adapter for Claude Code CLI."""

    model: Literal["sonnet", "opus", "haiku"] = "sonnet"
    timeout: int = 300
    cwd: Path | None = None
    _cli: ClaudeCodeCLI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize CLI wrapper."""
        self._cli = ClaudeCodeCLI(
            model=self.model,
            timeout=self.timeout,
            cwd=self.cwd,
        )

    def name(self) -> str:
        """Return model identifier."""
        return f"claude-code:{self.model}"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create AgentModel instance for this configuration."""
        return ClaudeCodeAgentModel(
            cli=self._cli,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


@dataclass
class ClaudeCodeAgentModel(AgentModel):
    """AgentModel implementation using Claude Code CLI."""

    cli: ClaudeCodeCLI
    _function_tools: list[ToolDefinition] = field(default_factory=list)
    _allow_text_result: bool = True
    _result_tools: list[ToolDefinition] = field(default_factory=list)

    def __init__(
        self,
        cli: ClaudeCodeCLI,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> None:
        self.cli = cli
        self._function_tools = function_tools
        self._allow_text_result = allow_text_result
        self._result_tools = result_tools
```

Update imports:
```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
```

### Step 4: Verify Tests Pass
Run: `uv run pytest tests/test_model.py -v`

## Commit
Message: "feat: add ClaudeCodeModel and ClaudeCodeAgentModel classes"
```

---

## Wave 4: Message Conversion + Request Method (2 parallel agents)

### Agent 4J: Message Conversion
**Branch**: `wave4/message-conversion`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task J for claude-code-model project.

## Prerequisites
Ensure wave3/model-class is merged.

## Task: Message Conversion (_build_prompt) with TDD

### Step 1: Write Tests First

Append to `tests/test_model.py`:

```python
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition


class TestBuildPrompt:
    """Test _build_prompt method."""

    @pytest.fixture
    def agent_model(self) -> ClaudeCodeAgentModel:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI()
        return ClaudeCodeAgentModel(
            cli=cli,
            function_tools=[],
            allow_text_result=True,
            result_tools=[],
        )

    def test_system_prompt_in_instructions(self, agent_model: ClaudeCodeAgentModel) -> None:
        messages = [ModelRequest(parts=[SystemPromptPart(content="Be helpful.")])]
        prompt = agent_model._build_prompt(messages)
        assert "## Instructions" in prompt
        assert "Be helpful." in prompt

    def test_multiple_system_prompts_combined(self, agent_model: ClaudeCodeAgentModel) -> None:
        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="Rule 1"),
                SystemPromptPart(content="Rule 2"),
            ])
        ]
        prompt = agent_model._build_prompt(messages)
        assert "Rule 1" in prompt
        assert "Rule 2" in prompt

    def test_user_prompt_formatted(self, agent_model: ClaudeCodeAgentModel) -> None:
        messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        prompt = agent_model._build_prompt(messages)
        assert "User: Hello" in prompt

    def test_assistant_text_formatted(self, agent_model: ClaudeCodeAgentModel) -> None:
        messages = [ModelResponse(parts=[TextPart(content="Hi there")])]
        prompt = agent_model._build_prompt(messages)
        assert "Assistant: Hi there" in prompt

    def test_tool_call_formatted(self, agent_model: ClaudeCodeAgentModel) -> None:
        messages = [
            ModelResponse(parts=[
                ToolCallPart.from_raw_args(tool_name="search", args={"q": "test"})
            ])
        ]
        prompt = agent_model._build_prompt(messages)
        assert "Assistant called: search" in prompt
        assert '"q"' in prompt

    def test_tool_return_formatted(self, agent_model: ClaudeCodeAgentModel) -> None:
        messages = [
            ModelRequest(parts=[
                ToolReturnPart(tool_name="search", content="found 3 results")
            ])
        ]
        prompt = agent_model._build_prompt(messages)
        assert "Tool Result (search):" in prompt
        assert "found 3 results" in prompt

    def test_tools_section_when_tools_provided(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
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
            allow_text_result=True,
            result_tools=[],
        )
        messages = [ModelRequest(parts=[UserPromptPart(content="Find files")])]
        prompt = agent_model._build_prompt(messages)
        assert "## Available Tools" in prompt
        assert "search" in prompt
        assert "Search for files" in prompt

    def test_result_schema_section_when_result_tools(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI()
        result_tool = ToolDefinition(
            name="final_result",
            description="Return structured result",
            parameters_json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
        )
        agent_model = ClaudeCodeAgentModel(
            cli=cli,
            function_tools=[],
            allow_text_result=False,
            result_tools=[result_tool],
        )
        messages = [ModelRequest(parts=[UserPromptPart(content="Analyze")])]
        prompt = agent_model._build_prompt(messages)
        assert "## Required Output Format" in prompt

    def test_conversation_section_present(self, agent_model: ClaudeCodeAgentModel) -> None:
        messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        prompt = agent_model._build_prompt(messages)
        assert "## Conversation" in prompt

    def test_full_conversation_flow(self, agent_model: ClaudeCodeAgentModel) -> None:
        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="You are helpful."),
                UserPromptPart(content="What is 2+2?"),
            ]),
            ModelResponse(parts=[TextPart(content="4")]),
            ModelRequest(parts=[UserPromptPart(content="Thanks!")]),
        ]
        prompt = agent_model._build_prompt(messages)
        assert "You are helpful." in prompt
        assert "User: What is 2+2?" in prompt
        assert "Assistant: 4" in prompt
        assert "User: Thanks!" in prompt
```

### Step 2: Verify Tests Fail
Run: `uv run pytest tests/test_model.py::TestBuildPrompt -v`

### Step 3: Implement to Pass Tests

Add method to `ClaudeCodeAgentModel` in `src/claude_code_model/model.py`:

```python
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


    def _build_prompt(self, messages: list[ModelMessage]) -> str:
        """Convert Pydantic AI messages to prompt string."""
        sections: list[str] = []
        system_parts: list[str] = []
        conversation_parts: list[str] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        system_parts.append(part.content)
                    elif isinstance(part, UserPromptPart):
                        conversation_parts.append(f"User: {part.content}")
                    elif isinstance(part, ToolReturnPart):
                        conversation_parts.append(
                            f"Tool Result ({part.tool_name}): {part.content}"
                        )
            elif isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        conversation_parts.append(f"Assistant: {part.content}")
                    elif isinstance(part, ToolCallPart):
                        args_json = json.dumps(part.args)
                        conversation_parts.append(
                            f"Assistant called: {part.tool_name}({args_json})"
                        )

        # Build sections
        if system_parts:
            sections.append("## Instructions\n" + "\n".join(system_parts))

        if self._function_tools:
            tools_text = self._format_tools(self._function_tools)
            sections.append("## Available Tools\n" + tools_text)

        if self._result_tools:
            schema_text = self._format_result_schema(self._result_tools)
            sections.append("## Required Output Format\n" + schema_text)

        if conversation_parts:
            sections.append("## Conversation\n" + "\n".join(conversation_parts))

        return "\n\n".join(sections)

    def _format_tools(self, tools: list[ToolDefinition]) -> str:
        """Format tool definitions for prompt."""
        lines = []
        for tool in tools:
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}")
            lines.append(f"Parameters: {json.dumps(tool.parameters_json_schema)}")
            lines.append(f"To call: TOOL_CALL: {tool.name}({{...}})")
            lines.append("")
        return "\n".join(lines)

    def _format_result_schema(self, tools: list[ToolDefinition]) -> str:
        """Format result schema for prompt."""
        if not tools:
            return ""
        tool = tools[0]  # Use first result tool
        return f"Return JSON matching this schema:\n{json.dumps(tool.parameters_json_schema, indent=2)}"
```

### Step 4: Verify Tests Pass
Run: `uv run pytest tests/test_model.py -v`

## Commit
Message: "feat: add message conversion (_build_prompt)"
```

---

### Agent 4K: Request Method
**Branch**: `wave4/request-method`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task K for claude-code-model project.

## Prerequisites
Ensure wave3/cli-run and wave4/message-conversion are merged.

## Task: AgentModel.request() Method with TDD

### Step 1: Write Tests First

Append to `tests/test_model.py`:

```python
from unittest.mock import AsyncMock

from pydantic_ai.usage import Usage


class TestAgentModelRequest:
    """Test ClaudeCodeAgentModel.request() method."""

    @pytest.fixture
    def agent_model(self) -> ClaudeCodeAgentModel:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI()
        return ClaudeCodeAgentModel(
            cli=cli,
            function_tools=[],
            allow_text_result=True,
            result_tools=[],
        )

    @pytest.mark.asyncio
    async def test_returns_text_response(self, agent_model: ClaudeCodeAgentModel) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'Hello, world!', b'')
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]
            response, usage = await agent_model.request(messages, None)

        assert len(response.parts) == 1
        assert isinstance(response.parts[0], TextPart)
        assert response.parts[0].content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_returns_usage_with_zero_tokens(self, agent_model: ClaudeCodeAgentModel) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'response', b'')
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]
            _, usage = await agent_model.request(messages, None)

        assert isinstance(usage, Usage)
        assert usage.request_tokens == 0
        assert usage.response_tokens == 0

    @pytest.mark.asyncio
    async def test_detects_tool_call_in_response(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI()
        tool = ToolDefinition(
            name="search",
            description="Search",
            parameters_json_schema={"type": "object"},
        )
        agent_model = ClaudeCodeAgentModel(
            cli=cli,
            function_tools=[tool],
            allow_text_result=True,
            result_tools=[],
        )

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b'TOOL_CALL: search({"query": "test"})',
            b''
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Search")])]
            response, _ = await agent_model.request(messages, None)

        assert len(response.parts) == 1
        assert isinstance(response.parts[0], ToolCallPart)
        assert response.parts[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_ignores_unknown_tool_call(self, agent_model: ClaudeCodeAgentModel) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b'TOOL_CALL: unknown_tool({"a": 1})',
            b''
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Test")])]
            response, _ = await agent_model.request(messages, None)

        # Should fall back to text since tool not in function_tools
        assert len(response.parts) == 1
        assert isinstance(response.parts[0], TextPart)

    @pytest.mark.asyncio
    async def test_handles_multiple_tool_calls(self) -> None:
        with patch("claude_code_model.cli.shutil.which", return_value="/usr/bin/claude"):
            cli = ClaudeCodeCLI()
        tools = [
            ToolDefinition(name="tool1", description="T1", parameters_json_schema={}),
            ToolDefinition(name="tool2", description="T2", parameters_json_schema={}),
        ]
        agent_model = ClaudeCodeAgentModel(
            cli=cli,
            function_tools=tools,
            allow_text_result=True,
            result_tools=[],
        )

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b'TOOL_CALL: tool1({"a": 1})\nTOOL_CALL: tool2({"b": 2})',
            b''
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Test")])]
            response, _ = await agent_model.request(messages, None)

        assert len(response.parts) == 2
        assert all(isinstance(p, ToolCallPart) for p in response.parts)

    @pytest.mark.asyncio
    async def test_returns_model_response_type(self, agent_model: ClaudeCodeAgentModel) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'test', b'')
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]
            response, _ = await agent_model.request(messages, None)

        assert isinstance(response, ModelResponse)
```

### Step 2: Verify Tests Fail
Run: `uv run pytest tests/test_model.py::TestAgentModelRequest -v`

### Step 3: Implement to Pass Tests

Add method to `ClaudeCodeAgentModel`:

```python
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import Usage


    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> tuple[ModelResponse, Usage]:
        """
        Make request to CLI and parse response.

        Args:
            messages: Conversation messages
            model_settings: Optional model settings (unused)

        Returns:
            Tuple of (ModelResponse, Usage)
        """
        prompt = self._build_prompt(messages)
        result = await self.cli.run(prompt)

        parts: list[TextPart | ToolCallPart] = []

        # Check for tool calls first
        tool_calls = extract_tool_calls(result.stdout)
        if tool_calls and self._function_tools:
            tool_names = {t.name for t in self._function_tools}
            for tc in tool_calls:
                if tc.name in tool_names:
                    parts.append(
                        ToolCallPart.from_raw_args(
                            tool_name=tc.name,
                            args=tc.args,
                        )
                    )

        # If no valid tool calls, return text
        if not parts:
            parts.append(TextPart(content=result.stdout))

        return ModelResponse(parts=parts), Usage(
            request_tokens=0,
            response_tokens=0,
        )
```

### Step 4: Verify Tests Pass
Run: `uv run pytest tests/test_model.py -v`

## Commit
Message: "feat: add AgentModel.request() method"
```

---

## Wave 5: Package Exports + Examples (4 parallel agents)

### Agent 5M: Package Exports
**Branch**: `wave5/exports`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task M for claude-code-model project.

## Prerequisites
Ensure all Wave 4 branches are merged.

## Task: Package Exports

Update `src/claude_code_model/__init__.py`:

```python
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
```

## Verification
Run:
```bash
uv run python -c "from claude_code_model import ClaudeCodeModel, ClaudeCodeError; print('OK')"
uv run pytest
uv run mypy src/
```

## Commit
Message: "feat: configure package exports"
```

---

### Agent 5N: Simple Example
**Branch**: `wave5/example-simple`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task N for claude-code-model project.

## Task: Simple Structured Output Example

Create `examples/simple.py`:

```python
"""
Simple structured output example.

Demonstrates using ClaudeCodeModel with Pydantic AI for type-safe responses.
"""
from __future__ import annotations

from pydantic import BaseModel

from pydantic_ai import Agent

from claude_code_model import ClaudeCodeModel


class ReviewResult(BaseModel):
    """Code review result schema."""

    verdict: str  # APPROVE, REQUEST_CHANGES, COMMENT
    issues: list[str]
    suggestions: list[str]


def main() -> None:
    """Run the example."""
    # Create agent with structured output
    agent: Agent[None, ReviewResult] = Agent(
        ClaudeCodeModel(),
        result_type=ReviewResult,
        system_prompt=(
            "You are a code reviewer. Analyze code and return your review "
            "as JSON matching the required schema. Be concise but thorough."
        ),
    )

    # Sample code to review
    code = '''
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item["price"] * item["quantity"]
    return total
'''

    # Run the agent
    result = agent.run_sync(f"Review this Python code:\n```python\n{code}\n```")

    # Result is typed!
    print(f"Verdict: {result.data.verdict}")
    print(f"Issues found: {len(result.data.issues)}")
    for issue in result.data.issues:
        print(f"  - {issue}")
    print(f"Suggestions: {len(result.data.suggestions)}")
    for suggestion in result.data.suggestions:
        print(f"  - {suggestion}")


if __name__ == "__main__":
    main()
```

## Note
This example requires claude CLI to be installed and authenticated.

## Commit
Message: "docs: add simple structured output example"
```

---

### Agent 5O: Tools Example
**Branch**: `wave5/example-tools`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task O for claude-code-model project.

## Task: Tools Example

Create `examples/with_tools.py`:

```python
"""
Example with tool usage.

Demonstrates how to give the agent tools it can call.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent

from claude_code_model import ClaudeCodeModel


# Create agent
agent: Agent[None, str] = Agent(
    ClaudeCodeModel(),
    system_prompt=(
        "You are a helpful assistant with access to file system tools. "
        "Use the tools to answer questions about files and directories."
    ),
)


@agent.tool_plain
def read_file(path: str) -> str:
    """
    Read content from a file.

    Args:
        path: Path to the file to read
    """
    try:
        return Path(path).read_text()[:2000]  # Limit to 2000 chars
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@agent.tool_plain
def list_directory(path: str = ".") -> str:
    """
    List files and directories.

    Args:
        path: Directory path (defaults to current directory)
    """
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: Path not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"

        items = []
        for item in sorted(p.iterdir())[:30]:  # Limit to 30 items
            prefix = "📁 " if item.is_dir() else "📄 "
            items.append(f"{prefix}{item.name}")
        return "\n".join(items) if items else "(empty directory)"
    except PermissionError:
        return f"Error: Permission denied: {path}"


@agent.tool_plain
def file_info(path: str) -> str:
    """
    Get information about a file.

    Args:
        path: Path to the file
    """
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: Path not found: {path}"

        stat = p.stat()
        return (
            f"Name: {p.name}\n"
            f"Type: {'directory' if p.is_dir() else 'file'}\n"
            f"Size: {stat.st_size} bytes\n"
        )
    except Exception as e:
        return f"Error: {e}"


def main() -> None:
    """Run the example."""
    print("File Assistant (type 'quit' to exit)\n")

    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue

            result = agent.run_sync(query)
            print(f"\nAssistant: {result.data}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
```

## Commit
Message: "docs: add tools usage example"
```

---

### Agent 5P: Multi-Agent Example
**Branch**: `wave5/example-multi-agent`

```
PROMPT FOR CLAUDE CODE (Sonnet):

You are implementing Task P for claude-code-model project.

## Task: Multi-Agent Example

Create `examples/multi_agent.py`:

```python
"""
Multi-agent delegation example.

Demonstrates using different models for different tasks
and having agents delegate to each other.
"""
from __future__ import annotations

import asyncio

from pydantic import BaseModel

from pydantic_ai import Agent

from claude_code_model import ClaudeCodeModel


# --- Schemas ---


class ResearchResult(BaseModel):
    """Research findings schema."""

    topic: str
    key_points: list[str]
    summary: str


class AnalysisResult(BaseModel):
    """Final analysis schema."""

    conclusion: str
    recommendations: list[str]
    confidence: str  # high, medium, low


# --- Agents ---

# Fast agent for quick research tasks (uses haiku)
researcher: Agent[None, ResearchResult] = Agent(
    ClaudeCodeModel(model="haiku"),
    result_type=ResearchResult,
    system_prompt=(
        "You are a research assistant. When given a topic, provide "
        "key points and a brief summary. Be factual and concise."
    ),
)

# Powerful agent for complex analysis (uses sonnet)
analyst: Agent[None, AnalysisResult] = Agent(
    ClaudeCodeModel(model="sonnet"),
    result_type=AnalysisResult,
    system_prompt=(
        "You are a senior analyst. Analyze research findings and provide "
        "actionable recommendations. Consider multiple perspectives."
    ),
)


# --- Tool for delegation ---


@analyst.tool_plain
async def research_topic(topic: str) -> str:
    """
    Delegate research to the research specialist.

    Args:
        topic: The topic to research
    """
    result = await researcher.run(f"Research this topic: {topic}")
    return (
        f"Research on '{result.data.topic}':\n"
        f"Key Points:\n" +
        "\n".join(f"- {p}" for p in result.data.key_points) +
        f"\n\nSummary: {result.data.summary}"
    )


# --- Main ---


async def main() -> None:
    """Run the multi-agent example."""
    print("Multi-Agent Analysis System")
    print("=" * 40)
    print()

    # The analyst will use the researcher tool to gather info
    query = (
        "Analyze the pros and cons of using type hints in Python. "
        "First research the topic, then provide your analysis."
    )

    print(f"Query: {query}\n")
    print("Running analysis (this may use multiple agents)...\n")

    result = await analyst.run(query)

    print("=" * 40)
    print("ANALYSIS RESULT")
    print("=" * 40)
    print(f"\nConclusion: {result.data.conclusion}")
    print(f"\nConfidence: {result.data.confidence}")
    print("\nRecommendations:")
    for i, rec in enumerate(result.data.recommendations, 1):
        print(f"  {i}. {rec}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Commit
Message: "docs: add multi-agent delegation example"
```

---

## PR Review Prompts (Haiku - cost efficient)

After each wave, run review agents on the PRs:

```
PROMPT FOR PR REVIEW (Haiku):

Review this PR for the claude-code-model project.

## Review Checklist
1. **Type Safety**: All functions have type hints? mypy passes?
2. **Tests**: Coverage adequate? Edge cases tested?
3. **Code Style**: Follows project conventions? No print() debugging?
4. **Errors**: Specific exceptions? No bare except?
5. **Documentation**: Docstrings present? Clear?

## Output Format
```markdown
## Review: [PR Title]

### ✅ Approved / ⚠️ Changes Requested

### Issues Found
- [ ] Issue 1: description (file:line)
- [ ] Issue 2: description (file:line)

### Suggestions
- Suggestion 1
- Suggestion 2

### Tests Missing
- Test case 1
- Test case 2
```

Review the code changes and provide feedback.
```

---

## Fix Agent Prompts (Sonnet)

```
PROMPT FOR FIXES (Sonnet):

You are fixing issues from code review.

## PR: [PR Title]
## Branch: [branch-name]

## Issues to Fix:
1. [Issue from review]
2. [Issue from review]

## Instructions:
1. Read the current code
2. Fix each issue
3. Run tests to verify: `uv run pytest`
4. Run type check: `uv run mypy src/`
5. Commit with message: "fix: address review feedback"

Fix all issues and verify tests pass.
```

---

## Execution Orchestrator Script

```bash
#!/bin/bash
# orchestrate.sh - Run parallel waves

set -e

REPO_DIR="$(pwd)"

run_wave() {
    local wave=$1
    shift
    local branches=("$@")

    echo "=== Wave $wave: Starting ${#branches[@]} parallel agents ==="

    # Start agents in parallel
    local pids=()
    for branch in "${branches[@]}"; do
        (
            cd "$REPO_DIR"
            git checkout -b "$branch" main 2>/dev/null || git checkout "$branch"
            # Run claude with the prompt from PLAN.md
            claude -p "$(cat prompts/${branch}.md)" --model sonnet
        ) &
        pids+=($!)
    done

    # Wait for all to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done

    echo "=== Wave $wave: Complete ==="
}

# Wave 1: Foundation
run_wave 1 "wave1/setup" "wave1/cli-exceptions" "wave1/json-parsing"

# Merge wave 1
for branch in wave1/setup wave1/cli-exceptions wave1/json-parsing; do
    git checkout main && git merge "$branch" --no-edit
done

# Wave 2: CLI Init + Tool Parsing
run_wave 2 "wave2/cli-init" "wave2/tool-parsing"

# ... continue for all waves
```

---

## Summary

| Wave | Tasks | Parallel Agents | Dependencies |
|------|-------|-----------------|--------------|
| 1 | A, B, C | 3 | None |
| 2 | D, E | 2 | Wave 1 |
| 3 | G, H | 2 | Wave 2 |
| 4 | J, K | 2 | Wave 3 |
| 5 | M, N, O, P | 4 | Wave 4 |
| 6 | Q | 1 | Wave 5 |

**Total**: 14 implementation agents + review agents + fix agents

Each wave's prompts are complete and self-contained for Sonnet execution.
