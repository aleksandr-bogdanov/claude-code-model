# claude-code-model

**Pydantic AI model adapter for Claude Code CLI — type-safe agents on your Max subscription.**

Use Pydantic AI's powerful agent framework with Claude Code CLI instead of API calls. Get structured outputs, tool calling, and full type safety without per-token costs.

## Why Use This?

| Approach | Cost | Type Safety | Structured Output | Tool Calling |
|----------|------|-------------|-------------------|--------------|
| Raw `claude -p` | $0 | ❌ | ❌ | ❌ |
| Pydantic AI + API | $$$ per token | ✅ | ✅ | ✅ |
| **claude-code-model** | $0 | ✅ | ✅ | ✅ |

If you have a Claude Max subscription ($100-200/month), you get unlimited Claude access via CLI. This adapter lets you use that access with Pydantic AI's excellent developer experience.

## Installation

```bash
# 1. Install Claude Code CLI (if not already installed)
npm install -g @anthropic-ai/claude-code

# 2. Authenticate
claude auth

# 3. Install this package
pip install claude-code-model
```

## Quick Start

### Structured Output

Get type-safe responses with automatic validation:

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from claude_code_model import ClaudeCodeModel

class ReviewResult(BaseModel):
    verdict: str  # APPROVE, REQUEST_CHANGES, COMMENT
    issues: list[str]
    suggestions: list[str]

agent = Agent(
    ClaudeCodeModel(),
    result_type=ReviewResult,
    system_prompt="You are a code reviewer. Return structured JSON."
)

result = agent.run_sync("Review this code: def add(a,b): return a+b")

# Result is fully typed!
print(result.data.verdict)  # IDE autocomplete works
print(result.data.issues)   # Type checking works
```

### Tool Calling

Give your agent functions it can call:

```python
from pydantic_ai import Agent
from claude_code_model import ClaudeCodeModel

agent = Agent(ClaudeCodeModel())

@agent.tool_plain
def read_file(path: str) -> str:
    """Read a file from disk."""
    return Path(path).read_text()

@agent.tool_plain
def list_files(directory: str = ".") -> list[str]:
    """List files in a directory."""
    return [f.name for f in Path(directory).iterdir()]

result = agent.run_sync("What Python files are in the current directory?")
# Agent automatically calls list_files() and uses the results
```

### Multi-Agent Systems

Different models for different tasks:

```python
from claude_code_model import ClaudeCodeModel

# Fast agent for quick tasks
researcher = Agent(
    ClaudeCodeModel(model="haiku"),
    system_prompt="Quick research"
)

# Powerful agent for analysis
analyst = Agent(
    ClaudeCodeModel(model="sonnet"),
    system_prompt="Deep analysis"
)

# Analyst can delegate to researcher
@analyst.tool_plain
async def research(topic: str) -> str:
    result = await researcher.run(f"Research {topic}")
    return result.data
```

## Features

- **Zero API Costs**: Uses your Claude Max subscription
- **Full Pydantic AI Compatibility**: Structured outputs, tools, deps, async
- **Type Safety**: Full IDE autocomplete and type checking
- **Multiple Models**: sonnet (default), opus, haiku
- **Tool Calling**: Give agents functions to call
- **Async Support**: Full async/await support
- **Simple**: <500 lines of code, easy to audit

## Configuration

```python
from claude_code_model import ClaudeCodeModel
from pathlib import Path

model = ClaudeCodeModel(
    model="sonnet",      # "sonnet" | "opus" | "haiku"
    timeout=300,         # seconds
    cwd=Path("/path"),   # working directory for CLI
)
```

## Examples

The `examples/` directory contains working examples:

- **`simple.py`**: Structured output for code review
- **`with_tools.py`**: File system tools with interactive chat
- **`multi_agent.py`**: Multi-agent delegation pattern

Run them:

```bash
uv run python examples/simple.py
uv run python examples/with_tools.py
```

## How It Works

```
┌─────────────────────────────────────────┐
│  Your Code                              │
│  agent = Agent(ClaudeCodeModel())       │
│  result = agent.run_sync("prompt")      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Pydantic AI                            │
│  - Manages conversation state           │
│  - Handles tool calls and retries       │
│  - Validates output against schema      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  claude-code-model                      │
│  - Converts messages → prompt string    │
│  - Adds JSON schema instructions        │
│  - Calls CLI wrapper                    │
│  - Parses response → ModelResponse      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Claude Code CLI                        │
│  $ claude -p "prompt" --model sonnet    │
│  > {"verdict": "APPROVE", "issues": []} │
└─────────────────────────────────────────┘
```

The adapter translates between Pydantic AI's message format and Claude CLI's prompt-response model. It handles:

- System prompts, user messages, tool calls, tool results
- JSON extraction from various response formats
- Tool call detection and parsing
- Structured output validation

## API Reference

### ClaudeCodeModel

Main model class implementing Pydantic AI's `Model` interface.

```python
@dataclass
class ClaudeCodeModel(Model):
    model: Literal["sonnet", "opus", "haiku"] = "sonnet"
    timeout: int = 300  # seconds
    cwd: Path | None = None
```

### Exceptions

```python
from claude_code_model import (
    ClaudeCodeError,           # Base exception
    ClaudeCodeNotFoundError,   # CLI not installed
    ClaudeCodeTimeoutError,    # Command timed out
    ClaudeCodeExecutionError,  # Non-zero exit code
)
```

## Limitations

- **No streaming**: CLI doesn't support streaming well
- **No token counting**: CLI doesn't report usage (returns 0)
- **No concurrent calls**: Run one request at a time
- **Text-only**: No image inputs (CLI limitation)
- **Rate limits**: Subject to Claude Max rate limits

## Development

```bash
# Setup
git clone https://github.com/yourusername/claude-code-model.git
cd claude-code-model
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
```

## Requirements

- Python 3.11+
- Claude Code CLI installed and authenticated
- Claude Max subscription (for unlimited CLI access)
- pydantic-ai >= 0.1.0
- pydantic >= 2.0.0

## Contributing

Contributions welcome! Please:

1. Keep changes focused and small (<200 lines)
2. Add tests for new functionality
3. Ensure `pytest`, `mypy`, and `ruff` all pass
4. Update README if adding features

See [CLAUDE.md](CLAUDE.md) for development guide.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

Built with:
- [Pydantic AI](https://ai.pydantic.dev/) - The agent framework
- [Claude Code](https://claude.ai/claude-code) - Anthropic's CLI tool
- [Pydantic](https://docs.pydantic.dev/) - Data validation

## Disclaimer

This is an unofficial adapter, not affiliated with Anthropic. Use in accordance with Claude's [Terms of Service](https://www.anthropic.com/legal/consumer-terms). You are responsible for ensuring your usage complies with Anthropic's policies.

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/claude-code-model/issues)
- Docs: [SPEC.md](SPEC.md) for technical details
- Dev Guide: [CLAUDE.md](CLAUDE.md) for contributors
