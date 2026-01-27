# claude-code-model

**Pydantic AI model adapter for Claude Code CLI — type-safe agents on your Max subscription.**

> ⚠️ **Work in Progress** — See SPEC.md for full design.

## Quick Start

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from claude_code_model import ClaudeCodeModel

class ReviewResult(BaseModel):
    verdict: str
    issues: list[str]

agent = Agent(ClaudeCodeModel(), output_type=ReviewResult)
result = agent.run_sync("Review this code: def add(a,b): return a+b")

print(result.output.verdict)  # Type-safe!
```

## Installation

```bash
# Requires Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Install package
pip install claude-code-model
```

## Why?

| Approach | Cost | Type Safety |
|----------|------|-------------|
| Raw `claude -p` | $0 | ❌ |
| Pydantic AI + API | $$$ | ✅ |
| **claude-code-model** | $0 | ✅ |

## License

MIT