# claude-code-model

**Use Pydantic AI with your Claude Max subscription instead of paying per token.**

A drop-in replacement that routes Pydantic AI through Claude Code CLI. Same code, same features, zero API costs.

## The Problem

You want to use [Pydantic AI](https://ai.pydantic.dev/) for type-safe AI agents, but the API costs add up:

```python
# This costs $3-15 per 1M tokens
from pydantic_ai.models.anthropic import AnthropicModel
model = AnthropicModel("claude-sonnet-4-20250514")
```

Meanwhile, your Claude Max subscription ($100-200/month) gives you **unlimited** Claude access via CLI, but no programmatic interface for building agents.

## The Solution

`claude-code-model` bridges this gap. Change 2 lines and your Pydantic AI code runs on your Max subscription:

```diff
- from pydantic_ai.models.anthropic import AnthropicModel
- model = AnthropicModel("claude-sonnet-4-20250514")
+ from claude_code_model import ClaudeCodeModel
+ model = ClaudeCodeModel(model="sonnet")
```

**That's it.** Your agents, tools, schemas, and prompts stay exactly the same.

## Quick Comparison

| Feature | Pydantic AI + API | claude-code-model |
|---------|-------------------|-------------------|
| Cost | $3-15/M tokens | $0 (Max subscription) |
| Structured output | Yes | Yes |
| Tool calling | Yes | Yes |
| Type safety | Yes | Yes |
| Streaming | Yes | No |
| Concurrent requests | Yes | No |
| Migration effort | — | **2 lines** |

## Installation

```bash
# 1. Install Claude Code CLI (requires Node.js)
npm install -g @anthropic-ai/claude-code

# 2. Login to Claude
claude auth

# 3. Install this package
pip install claude-code-model
```

## Usage

### Basic Example

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from claude_code_model import ClaudeCodeModel

class MovieReview(BaseModel):
    title: str
    rating: int  # 1-10
    summary: str

agent = Agent(
    ClaudeCodeModel(),
    output_type=MovieReview,
    system_prompt="You are a film critic. Review movies concisely."
)

result = agent.run_sync("Review: The Matrix (1999)")
print(f"{result.output.title}: {result.output.rating}/10")
# The Matrix: 9/10
```

### With Tools

```python
from pydantic_ai import Agent
from claude_code_model import ClaudeCodeModel

agent = Agent(ClaudeCodeModel())

@agent.tool_plain
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Your implementation here
    return f"Weather in {city}: 72°F, sunny"

result = agent.run_sync("What's the weather in Tokyo?")
# Agent calls get_weather("Tokyo") and uses the result
```

### Multi-Agent Systems

```python
from pydantic_ai import Agent
from claude_code_model import ClaudeCodeModel

# Fast agent for simple tasks
fast_agent = Agent(
    ClaudeCodeModel(model="haiku"),
    system_prompt="Quick factual responses only."
)

# Powerful agent for complex reasoning
smart_agent = Agent(
    ClaudeCodeModel(model="sonnet"),
    system_prompt="Deep analysis and reasoning."
)

# Agents can delegate to each other via tools
@smart_agent.tool_plain
async def quick_lookup(query: str) -> str:
    """Delegate simple lookups to the fast agent."""
    result = await fast_agent.run(query)
    return result.output
```

See [`examples/multi_agent_comparison.py`](examples/multi_agent_comparison.py) for a complete multi-agent pipeline.

## Migration from Pydantic AI

### Step 1: Install

```bash
pip install claude-code-model
```

### Step 2: Replace Import

```python
# Before
from pydantic_ai.models.anthropic import AnthropicModel

# After
from claude_code_model import ClaudeCodeModel
```

### Step 3: Replace Model

```python
# Before
model = AnthropicModel("claude-sonnet-4-20250514")

# After
model = ClaudeCodeModel(model="sonnet")  # or "opus", "haiku"
```

### That's It

All your existing code works unchanged:

```python
# This code is IDENTICAL for both API and CLI
agent = Agent(
    model,  # <-- Only this changes
    output_type=MySchema,
    system_prompt="Your prompt here",
    retries=3,
)

@agent.tool_plain
def my_tool(arg: str) -> str:
    return do_something(arg)

result = agent.run_sync("Your query")
print(result.output)  # Typed result
```

## Configuration

```python
from claude_code_model import ClaudeCodeModel

model = ClaudeCodeModel(
    model="sonnet",    # "sonnet" (default), "opus", or "haiku"
    timeout=30,        # Seconds per CLI request (default: 30)
    verbose=False,     # Enable debug logging
)
```

### Model Selection

| Model | Speed | Capability | Use Case |
|-------|-------|------------|----------|
| `haiku` | Fast | Good | Simple tasks, quick lookups |
| `sonnet` | Medium | Great | Most tasks (default) |
| `opus` | Slow | Best | Complex reasoning, analysis |

### Debug Mode

```python
model = ClaudeCodeModel(verbose=True)
```

Shows request/response timing, prompts sent, tool calls detected, and JSON extraction status.

## How It Works

```
Your Code                      claude-code-model              Claude CLI
─────────────────────────────────────────────────────────────────────────
agent.run_sync("prompt")  →   Build prompt string      →   claude -p "..."
                              Add tool definitions          --model sonnet
                              Add JSON schema

result.output             ←   Parse response           ←   {"rating": 9, ...}
                              Extract JSON
                              Validate with Pydantic
```

The adapter:
1. Converts Pydantic AI messages → single prompt string
2. Adds tool definitions with format instructions
3. Adds JSON schema requirements for structured output
4. Calls `claude -p "prompt" --model sonnet` via subprocess
5. Parses response for tool calls or JSON output
6. Returns typed `ModelResponse` to Pydantic AI

## Examples

Run the examples to see it in action:

```bash
# Basic structured output
uv run python examples/simple.py

# Interactive tool usage
uv run python examples/with_tools.py

# Multi-agent pipeline with comparison
uv run python examples/multi_agent_comparison.py
```

## Limitations

**No streaming** — CLI doesn't support streaming well. Each request returns complete.

**No concurrent requests** — Run one request at a time. CLI sessions don't parallelize.

**No token counting** — CLI doesn't report token usage. `result.usage` returns zeros.

**No images** — Text-only. CLI doesn't support image inputs.

**Slower than API** — Each CLI invocation takes 2-10 seconds overhead. Multi-step tasks with tool calls take longer.

**Tool calling is prompt-based** — Unlike native API tools, this adapter instructs Claude to output `TOOL_CALL: name({args})` format. Works well but occasionally needs retries. Use `retries=3-5` on your agents.

## Exceptions

```python
from claude_code_model import (
    ClaudeCodeError,           # Base exception
    ClaudeCodeNotFoundError,   # CLI not installed
    ClaudeCodeTimeoutError,    # Request timed out
    ClaudeCodeExecutionError,  # CLI returned error
)
```

## Development

```bash
git clone https://github.com/yourusername/claude-code-model.git
cd claude-code-model
uv sync

# Run tests
uv run pytest

# Type check
uv run mypy src/

# Lint
uv run ruff check src/ tests/
```

## Requirements

- Python 3.11+
- Claude Code CLI installed and authenticated
- Claude Max subscription (for unlimited CLI access)
- pydantic-ai >= 0.1.0
- pydantic >= 2.0.0

## License

MIT — see [LICENSE](LICENSE).

## Links

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Claude Code](https://claude.ai/claude-code)
- [Development Guide](CLAUDE.md)

---

**Disclaimer**: Unofficial adapter, not affiliated with Anthropic. Use in accordance with Claude's [Terms of Service](https://www.anthropic.com/legal/consumer-terms).
