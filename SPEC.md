# claude-code-model — Technical Specification

## One-Liner

**Pydantic AI model adapter for Claude Code CLI — type-safe agents on your Max subscription.**

## Problem Statement

Claude Max subscribers ($100-200/month) get unlimited access to Claude via the Claude Code CLI (`claude -p`). However:

1. **No structured output**: CLI returns raw text, requiring fragile regex/string parsing
2. **No type safety**: No IDE autocomplete, no validation, runtime errors
3. **No retry logic**: Parse failures require manual handling
4. **No tool abstraction**: Building agents requires reinventing the wheel

Meanwhile, **Pydantic AI** solves all of these problems — but requires API access ($$$).

## Solution

A drop-in `Model` adapter that lets Pydantic AI use Claude Code CLI instead of API calls:

```python
from pydantic_ai import Agent
from claude_code_model import ClaudeCodeModel

agent = Agent(ClaudeCodeModel(), output_type=MySchema)
result = agent.run_sync("prompt")
# result.output is validated, typed, with IDE autocomplete
```

## Goals

1. **Zero API costs**: Use existing Max subscription
2. **Full Pydantic AI compatibility**: Structured outputs, tools, deps, async
3. **Simple installation**: `pip install claude-code-model`
4. **Minimal code**: <500 lines total, easy to audit and maintain

## Non-Goals

1. **Streaming**: CLI doesn't support it well, skip for v0.1
2. **Token counting**: CLI doesn't report usage, accept unknown
3. **Multiple concurrent calls**: Let user handle, don't overcomplicate
4. **GUI/TUI**: This is a library, not an application

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Code                              │
│                                                             │
│  agent = Agent(ClaudeCodeModel(), output_type=Result)       │
│  result = agent.run_sync("Review this code")                │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pydantic AI                              │
│                                                             │
│  - Manages conversation state                               │
│  - Handles tool calls and retries                           │
│  - Validates output against schema                          │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  claude-code-model                          │
│                                                             │
│  ClaudeCodeModel (implements pydantic_ai.models.Model)      │
│    │                                                        │
│    ├── Converts messages → prompt string                    │
│    ├── Adds JSON schema instructions                        │
│    ├── Calls CLI wrapper                                    │
│    └── Parses response → ModelResponse                      │
│                                                             │
│  ClaudeCodeCLI (subprocess wrapper)                         │
│    │                                                        │
│    ├── Runs: claude -p "..." --model sonnet                 │
│    ├── Handles timeout                                      │
│    └── Returns stdout/stderr/exit code                      │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Claude Code CLI                           │
│                                                             │
│  $ claude -p "prompt" --model sonnet                        │
│  > {"verdict": "APPROVE", "issues": []}                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. ClaudeCodeCLI (`cli.py`)

Low-level wrapper around the `claude` command.

```python
@dataclass
class ClaudeCodeCLI:
    model: str = "sonnet"      # sonnet | opus | haiku
    timeout: int = 300         # seconds
    cwd: Path | None = None    # working directory
    
    async def run(self, prompt: str) -> CLIResult:
        """Execute claude -p and return result."""
        ...
```

**Responsibilities**:
- Find `claude` executable
- Build command with correct flags
- Handle timeout with `asyncio.wait_for`
- Capture stdout/stderr
- Raise typed exceptions on failure

**Exceptions**:
- `ClaudeCodeNotFoundError`: CLI not installed
- `ClaudeCodeTimeoutError`: Command timed out
- `ClaudeCodeError`: Non-zero exit code

### 2. ClaudeCodeModel (`model.py`)

Pydantic AI Model implementation.

```python
@dataclass
class ClaudeCodeModel(Model):
    model: Literal["sonnet", "opus", "haiku"] = "sonnet"
    timeout: int = 300
    cwd: Path | None = None
    
    async def agent_model(self, ...) -> AgentModel:
        """Return AgentModel for this configuration."""
        ...
```

**Responsibilities**:
- Implement `pydantic_ai.models.Model` interface
- Create `ClaudeCodeAgentModel` instances

### 3. ClaudeCodeAgentModel (`model.py`)

Handles actual request/response cycle.

```python
@dataclass  
class ClaudeCodeAgentModel(AgentModel):
    async def request(self, messages, settings) -> tuple[ModelResponse, Usage]:
        """Convert messages to prompt, call CLI, parse response."""
        ...
```

**Responsibilities**:
- Convert Pydantic AI messages to prompt string
- Add tool definitions to prompt
- Add JSON schema instructions for structured output
- Call CLI
- Parse response (JSON extraction, tool call detection)
- Return `ModelResponse` and `Usage`

### 4. Message Conversion

Convert Pydantic AI message types to prompt text:

| Message Type | Prompt Format |
|--------------|---------------|
| `SystemPromptPart` | `## Instructions\n{content}` |
| `UserPromptPart` | `User: {content}` |
| `ToolReturnPart` | `Tool Result ({name}): {content}` |
| `TextPart` (assistant) | `Assistant: {content}` |
| `ToolCallPart` | `Assistant called: {name}({args})` |

### 5. JSON Extraction

Parse JSON from various response formats:

1. Pure JSON: `{"key": "value"}`
2. Markdown block: `` ```json\n{...}\n``` ``
3. Embedded: `Here is the result: {"key": "value"} as requested`

### 6. Tool Call Detection

Detect when model wants to call a tool:

1. `TOOL_CALL: function_name({"arg": "value"})`
2. `<tool_call name="function_name">{"arg": "value"}</tool_call>`

## File Structure

```
claude-code-model/
├── src/claude_code_model/
│   ├── __init__.py      # Public API: ClaudeCodeModel
│   ├── cli.py           # ClaudeCodeCLI, exceptions
│   ├── model.py         # ClaudeCodeModel, ClaudeCodeAgentModel
│   └── py.typed         # PEP 561 marker
├── tests/
│   ├── __init__.py
│   ├── test_cli.py      # CLI wrapper tests
│   ├── test_model.py    # Model adapter tests
│   └── test_parsing.py  # JSON/tool extraction tests
├── examples/
│   ├── simple.py        # Basic usage
│   ├── with_tools.py    # Tool usage
│   └── multi_agent.py   # Agent delegation
├── pyproject.toml
├── README.md
├── SPEC.md              # This file
├── CLAUDE.md            # Development guide
└── LICENSE              # MIT
```

## Dependencies

**Runtime**:
- `pydantic-ai >= 0.1.0`
- `pydantic >= 2.0.0`
- Python 3.11+

**Development**:
- `pytest >= 8.0`
- `pytest-asyncio >= 0.23`
- `ruff >= 0.4`
- `mypy >= 1.10`

**External**:
- Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)

## Testing Strategy

### Unit Tests (mocked CLI)
- Message conversion
- JSON extraction (all formats)
- Tool call detection (all formats)
- Error handling (timeout, not found, exit code)

### Integration Tests (real CLI, optional)
- Simple prompt → response
- Structured output validation
- Tool calling round-trip

Mark integration tests with `@pytest.mark.integration` and skip by default.

## Success Criteria

| Metric | Target |
|--------|--------|
| Lines of code | < 500 |
| Test coverage | > 90% |
| Examples | 3+ working |
| Install time | < 30 seconds |
| First response | < 10 seconds (sonnet) |

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pydantic AI changes internal API | Breaks adapter | Pin version, test against releases |
| Claude CLI changes flags | Breaks CLI calls | Detect version, handle gracefully |
| Anthropic ToS violation | Account ban | Document risk, user responsibility |
| CLI is slow | Bad UX | Set reasonable timeouts, document |

## Future Enhancements (Post v0.1)

1. **Streaming**: If CLI adds `--stream` flag
2. **Conversation persistence**: Save/load state
3. **Rate limiting**: Respect CLI rate limits
4. **Caching**: Skip identical prompts
5. **Logging integration**: Logfire/OpenTelemetry

## References

- [Pydantic AI Docs](https://ai.pydantic.dev/)
- [Pydantic AI GitHub](https://github.com/pydantic/pydantic-ai)
- [Claude Code Docs](https://docs.anthropic.com/en/docs/claude-code)