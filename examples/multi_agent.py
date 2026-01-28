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
    output_type=ResearchResult,
    system_prompt=(
        "You are a research assistant. When given a topic, provide "
        "key points and a brief summary. Be factual and concise."
    ),
)

# Powerful agent for complex analysis (uses sonnet)
analyst: Agent[None, AnalysisResult] = Agent(
    ClaudeCodeModel(model="sonnet"),
    output_type=AnalysisResult,
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
        f"Research on '{result.output.topic}':\n"
        f"Key Points:\n" +
        "\n".join(f"- {p}" for p in result.output.key_points) +
        f"\n\nSummary: {result.output.summary}"
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
    print(f"\nConclusion: {result.output.conclusion}")
    print(f"\nConfidence: {result.output.confidence}")
    print("\nRecommendations:")
    for i, rec in enumerate(result.output.recommendations, 1):
        print(f"  {i}. {rec}")


if __name__ == "__main__":
    asyncio.run(main())
