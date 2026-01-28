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
        output_type=ReviewResult,
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
    print(f"Verdict: {result.output.verdict}")
    print(f"Issues found: {len(result.output.issues)}")
    for issue in result.output.issues:
        print(f"  - {issue}")
    print(f"Suggestions: {len(result.output.suggestions)}")
    for suggestion in result.output.suggestions:
        print(f"  - {suggestion}")


if __name__ == "__main__":
    main()
