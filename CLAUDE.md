# CLAUDE.md — Development Guide

## Project Summary

`claude-code-model` is a Pydantic AI model adapter that uses Claude Code CLI instead of API calls. It lets Max subscribers use Pydantic AI's type-safe agent framework without paying per-token API costs.

## Repository Structure

```
claude-code-model/
├── src/claude_code_model/
│   ├── __init__.py      # Exports: ClaudeCodeModel
│   ├── cli.py           # CLI wrapper
│   ├── model.py         # Pydantic AI Model implementation
│   └── py.typed         # Type hint marker
├── tests/
│   ├── test_cli.py
│   ├── test_model.py
│   └── test_parsing.py
├── examples/
│   ├── simple.py
│   ├── with_tools.py
│   └── multi_agent.py
├── pyproject.toml
├── README.md
└── CLAUDE.md            # This file
```

## Commands

```bash
# Setup
uv sync

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=claude_code_model

# Type check
uv run mypy src/

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Run example
uv run python examples/simple.py

# Build package
uv build

# Install locally for testing
uv pip install -e .
```

## Code Style

### Python
- Python 3.11+
- Type hints on ALL functions (enforced by mypy strict)
- Pydantic for data classes
- `ruff` for linting and formatting
- Google-style docstrings (brief)

### Naming
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `ClaudeCodeModel` |
| Functions | snake_case | `extract_json()` |
| Constants | UPPER_SNAKE | `DEFAULT_TIMEOUT` |
| Private | _prefix | `_parse_response()` |

### Imports
```python
# Standard library
from __future__ import annotations
import asyncio
import json

# Third party
from pydantic import BaseModel
from pydantic_ai import Agent

# Local
from claude_code_model.cli import ClaudeCodeCLI
```

## Key Implementation Details

### Pydantic AI Model Interface

We implement `pydantic_ai.models.Model`:

```python
from pydantic_ai.models import Model, AgentModel

class ClaudeCodeModel(Model):
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
        """Create AgentModel instance."""
        return ClaudeCodeAgentModel(...)
```

And `pydantic_ai.models.AgentModel`:

```python
class ClaudeCodeAgentModel(AgentModel):
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> tuple[ModelResponse, Usage]:
        """Make request and return response."""
        ...
```

### Message Types

From `pydantic_ai.messages`:

```python
# Request parts (user → model)
SystemPromptPart   # System instructions
UserPromptPart     # User message
ToolReturnPart     # Result of tool call
RetryPromptPart    # Retry instruction

# Response parts (model → user)
TextPart           # Text response
ToolCallPart       # Model wants to call a tool
```

### Building the Prompt

Convert messages to a single prompt string:

```python
def _build_prompt(self, messages: list[ModelMessage]) -> str:
    parts = []
    
    # 1. System prompts
    parts.append("## Instructions\n" + system_content)
    
    # 2. Tool definitions (if any)
    parts.append("## Available Tools\n" + tool_docs)
    
    # 3. Output schema (if structured output)
    parts.append("## Required Output Format\n" + schema_json)
    
    # 4. Conversation history
    parts.append("## Conversation\n" + conversation)
    
    return "\n\n".join(parts)
```

### JSON Extraction Priority

Try these in order:
1. Direct `json.loads(content)`
2. Extract from `` ```json ... ``` ``
3. Find `{...}` anywhere in content

### Tool Call Formats

Detect these patterns:
1. `TOOL_CALL: name({"arg": "val"})`
2. `<tool_call name="name">{"arg": "val"}</tool_call>`

## CLI Quirks and Workarounds

This section documents important behaviors of Claude Code CLI that affect this adapter.

### stdin Must Be DEVNULL

**Problem**: When running without a TTY (e.g., in PyCharm, VS Code, or CI), the CLI hangs indefinitely if stdin is inherited from the parent process.

**Solution**: Always pass `stdin=subprocess.DEVNULL` to the subprocess:

```python
proc = await asyncio.create_subprocess_exec(
    *cmd,
    stdin=async_subprocess.DEVNULL,  # REQUIRED - prevents hang
    stdout=async_subprocess.PIPE,
    stderr=async_subprocess.PIPE,
)
```

**Test**: `test_uses_pipe_for_stdout_stderr` verifies this.

### Session Persistence Causes Bleed-Through

**Problem**: Claude CLI maintains conversation history. When multiple requests are made, previous conversations can "bleed through" into new requests, causing Claude to reference context from unrelated sessions.

**Solution**: Always use `--no-session-persistence` flag:

```python
cmd = [
    "claude", "-p", prompt,
    "--model", model,
    "--no-session-persistence",  # REQUIRED - isolates each request
]
```

**Test**: `test_command_includes_no_session_persistence` verifies this.

### Claude Simulates Tool Responses

**Problem**: After outputting `TOOL_CALL: name({...})`, Claude sometimes continues and simulates what the tool response might be, like:

```
TOOL_CALL: search({"query": "python"})

User: Tool 'search' returned:
Here are some results about Python...
```

**Solution**: The prompt includes explicit instructions to stop immediately:

```
IMPORTANT RULES:
3. After outputting TOOL_CALL, you MUST STOP IMMEDIATELY
4. Do NOT simulate or imagine what the tool response might be
5. Do NOT continue the conversation after TOOL_CALL
```

### JSON Extraction for Structured Output

**Problem**: When structured output is requested, Claude often includes explanatory text before/after the JSON:

```
Based on my analysis, here's the result:
{"verdict": "APPROVE", "issues": []}
Hope this helps!
```

**Solution**: When `output_tools` are defined, the model extracts JSON from the response:

```python
if model_request_parameters.output_tools:
    extracted = extract_json(content)
    if extracted is not None:
        content = json.dumps(extracted)  # Return only the JSON
```

### Default Timeout is 30 Seconds

Each CLI request typically takes 2-10 seconds. The default timeout of 30 seconds allows for slow responses while failing fast on actual hangs. Adjust with `timeout=N` parameter.

### Retries Should Be Set on Agent

Pydantic AI handles retries at the Agent level, not the Model level. For reliability with CLI quirks, use:

```python
agent = Agent(
    ClaudeCodeModel(),
    output_type=MyResult,
    retries=5,  # Handles occasional Claude misbehavior
)
```

## Testing

### Mock the CLI

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_request():
    with patch("shutil.which", return_value="/usr/bin/claude"):
        model = ClaudeCodeModel()
    
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b'{"result": "ok"}', b'')
    mock_proc.returncode = 0
    
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        # Test here
        ...
```

### Test Categories

1. **test_cli.py**: CLI wrapper behavior
    - Command construction
    - Timeout handling
    - Error cases

2. **test_model.py**: Model adapter
    - Message conversion
    - Response parsing
    - Tool handling

3. **test_parsing.py**: Extraction functions
    - JSON formats
    - Tool call formats
    - Edge cases

## Error Handling

Define specific exceptions:

```python
class ClaudeCodeError(Exception):
    """Base exception."""
    pass

class ClaudeCodeNotFoundError(ClaudeCodeError):
    """CLI not installed."""
    pass

class ClaudeCodeTimeoutError(ClaudeCodeError):
    """CLI timed out."""
    pass
```

Always raise typed exceptions, never generic `Exception`.

## Documentation

### Docstrings

```python
def extract_json(content: str) -> dict[str, Any] | None:
    """
    Extract JSON object from response content.
    
    Tries multiple formats:
    1. Pure JSON
    2. Markdown code block
    3. Embedded in text
    
    Args:
        content: Raw response from CLI
        
    Returns:
        Parsed JSON dict, or None if no valid JSON found
    """
```

### README Sections

1. One-liner description
2. Installation
3. Quick start example
4. Features
5. Configuration options
6. Examples
7. How it works
8. Limitations
9. License

## Pre-Commit Validation

**IMPORTANT**: Run these checks before every commit to ensure CI will pass:

```bash
# Quick validation (run before every commit)
uv run ruff format src/ tests/        # Format code
uv run ruff check src/ tests/         # Lint
uv run mypy src/                      # Type check
uv run pytest                         # Run tests

# Full validation (run before releases)
uv build                              # Build package
uv run twine check dist/*             # Verify package metadata
```

One-liner for quick pre-commit check:
```bash
uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && uv run mypy src/ && uv run pytest
```

## PR Guidelines

- One logical change per PR
- < 200 lines of code changes
- Tests for new functionality
- Update README if adding features
- Run full test suite before submitting
- **Run pre-commit validation before pushing**

## Don'ts

- Don't use `print()` for debugging (use logging or remove)
- Don't catch bare `Exception` (be specific)
- Don't skip type hints
- Don't add major features without discussion
- Don't use `# type: ignore` without comment explaining why
- Don't hardcode paths (use Path objects)

## Debugging Tips

### CLI not found
```bash
which claude  # Should show path
claude --version  # Should work
```

### Test a prompt manually
```bash
claude -p "Say hello" --model sonnet --no-session-persistence
```

### Test tool call format
```bash
claude -p '## Available Tools
You have tools. Use: TOOL_CALL: greet({"name": "value"})

### greet
Says hello to someone.
Parameters: name (string, required)

## Conversation
User: Greet Alice' --model haiku --no-session-persistence
```

Expected output should include: `TOOL_CALL: greet({"name": "Alice"})`

### Enable verbose mode
```python
model = ClaudeCodeModel(verbose=True)
# Now all requests show timing, prompts, and responses
```

### Check request stats
```python
model = ClaudeCodeModel()
# ... run some requests ...
print(f"Requests: {model.request_count}, Time: {model.total_time:.1f}s")
model.reset_stats()  # Reset counters
```

### Debug IDE hang issues
If running from IDE hangs but terminal works:
1. Check that `stdin=DEVNULL` is set in subprocess call
2. Try running with: `uv run python examples/simple.py`
3. Compare TTY status: `python -c "import sys; print(sys.stdin.isatty())"`

### Check Pydantic AI internals
```python
from pydantic_ai.models import Model
print(Model.__abstractmethods__)  # See required methods
```

### Simulate non-TTY environment
```bash
# Run script as if from IDE (no TTY)
echo "" | uv run python examples/simple.py
```

## Dependencies Reference

```toml
[project]
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
```