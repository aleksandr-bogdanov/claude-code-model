"""
Multi-Agent Comparison: pydantic-ai (API) vs claude-code-model (CLI)

This example demonstrates:
1. Side-by-side syntax comparison between using the Anthropic API and Claude Code CLI
2. A practical multi-agent content creation pipeline
3. How to migrate from pydantic-ai to claude-code-model

IMPORTANT: The API version requires ANTHROPIC_API_KEY and costs money per token.
           The CLI version uses your Claude Max subscription (flat rate).

===================================================================================
SYNTAX COMPARISON: pydantic-ai vs claude-code-model
===================================================================================

The ONLY difference is the model you pass to the Agent. Everything else is identical!

┌────────────────────────────────────────────────────────────────────────────────┐
│ pydantic-ai (API) — Paid per token                                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   from pydantic_ai import Agent                                                │
│   from pydantic_ai.models.anthropic import AnthropicModel                      │
│                                                                                │
│   model = AnthropicModel("claude-sonnet-4-20250514")                           │
│   agent = Agent(model, output_type=MyResult, system_prompt="...")              │
│   result = agent.run_sync("prompt")                                            │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│ claude-code-model (CLI) — $0 with Max subscription                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   from pydantic_ai import Agent                                                │
│   from claude_code_model import ClaudeCodeModel  # <-- Only change!            │
│                                                                                │
│   model = ClaudeCodeModel(model="sonnet")                                      │
│   agent = Agent(model, output_type=MyResult, system_prompt="...")              │
│   result = agent.run_sync("prompt")                                            │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

===================================================================================
MIGRATION GUIDE: 3 Steps
===================================================================================

Step 1: Install claude-code-model
   pip install claude-code-model

Step 2: Replace import
   # Before
   from pydantic_ai.models.anthropic import AnthropicModel
   # After
   from claude_code_model import ClaudeCodeModel

Step 3: Replace model instantiation
   # Before
   model = AnthropicModel("claude-sonnet-4-20250514")
   # After
   model = ClaudeCodeModel(model="sonnet")  # or "opus", "haiku"

That's it! All your Agent code, tools, output_type schemas, and prompts stay exactly
the same. The claude-code-model implements the same Pydantic AI Model interface.

===================================================================================
"""

from __future__ import annotations

import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from claude_code_model import ClaudeCodeModel


# =============================================================================
# SCHEMAS — Identical for both API and CLI versions
# =============================================================================


class ResearchBrief(BaseModel):
    """Research findings about a topic."""

    topic: str = Field(description="The researched topic")
    key_facts: list[str] = Field(description="Key facts discovered")
    target_audience: str = Field(description="Who this content is for")
    tone_recommendation: str = Field(description="Suggested writing tone")


class ContentDraft(BaseModel):
    """Draft content created by the writer."""

    title: str = Field(description="Engaging title")
    hook: str = Field(description="Opening hook to grab attention")
    body: str = Field(description="Main content body")
    call_to_action: str = Field(description="Closing call to action")


class EditedContent(BaseModel):
    """Final edited content."""

    title: str
    content: str
    improvements_made: list[str] = Field(description="List of edits made")
    quality_score: int = Field(description="Quality score 1-10", ge=1, le=10)


# =============================================================================
# AGENTS — The magic: just swap ClaudeCodeModel for AnthropicModel
# =============================================================================

# Agent 1: Researcher (fast model for quick research)
researcher: Agent[None, ResearchBrief] = Agent(
    ClaudeCodeModel(model="haiku"),  # Fast & cheap for research
    output_type=ResearchBrief,
    system_prompt="""You are a content researcher. When given a topic:
1. Identify 3-5 key facts that would make compelling content
2. Determine the target audience
3. Recommend an appropriate tone
Be concise and actionable.""",
    retries=3,
)

# Agent 2: Writer (powerful model for creative work)
writer: Agent[None, ContentDraft] = Agent(
    ClaudeCodeModel(model="sonnet"),  # More capable for writing
    output_type=ContentDraft,
    system_prompt="""You are a skilled content writer. Given research, create:
1. An engaging title that captures attention
2. A strong opening hook
3. Well-structured body content
4. A compelling call to action
Write for the specified audience and tone.""",
    retries=3,
)

# Agent 3: Editor (most capable model for polish)
editor: Agent[None, EditedContent] = Agent(
    ClaudeCodeModel(model="sonnet"),  # Could use "opus" for highest quality
    output_type=EditedContent,
    system_prompt="""You are a senior editor. Review the draft and:
1. Fix any grammatical or stylistic issues
2. Improve clarity and flow
3. Ensure the tone matches the target audience
4. Rate the final quality 1-10
Return the polished version with a list of improvements made.""",
    retries=3,
)


# =============================================================================
# MULTI-AGENT PIPELINE — This code is identical for API and CLI!
# =============================================================================


async def create_content(topic: str) -> EditedContent:
    """
    Multi-agent content creation pipeline.

    Flow: Research → Write → Edit

    This exact same function works with both:
    - pydantic-ai + AnthropicModel (API, costs $$$)
    - claude-code-model + ClaudeCodeModel (CLI, $0 with Max)
    """
    print(f"\n{'=' * 60}")
    print(f"Creating content about: {topic}")
    print(f"{'=' * 60}")

    # Step 1: Research
    print("\n[1/3] Researching topic...")
    research = await researcher.run(f"Research this topic for a blog post: {topic}")
    print(f"  ✓ Target audience: {research.output.target_audience}")
    print(f"  ✓ Tone: {research.output.tone_recommendation}")
    print(f"  ✓ Found {len(research.output.key_facts)} key facts")

    # Step 2: Write (pass research context)
    print("\n[2/3] Writing draft...")
    writer_prompt = f"""Write content based on this research:

Topic: {research.output.topic}
Target Audience: {research.output.target_audience}
Tone: {research.output.tone_recommendation}
Key Facts:
{chr(10).join(f"- {fact}" for fact in research.output.key_facts)}
"""
    draft = await writer.run(writer_prompt)
    print(f"  ✓ Title: {draft.output.title}")
    print(f"  ✓ Draft length: {len(draft.output.body)} characters")

    # Step 3: Edit (pass draft for polishing)
    print("\n[3/3] Editing and polishing...")
    editor_prompt = f"""Edit this draft:

Title: {draft.output.title}
Hook: {draft.output.hook}
Body: {draft.output.body}
CTA: {draft.output.call_to_action}

Target audience: {research.output.target_audience}
Desired tone: {research.output.tone_recommendation}
"""
    final = await editor.run(editor_prompt)
    print(f"  ✓ Quality score: {final.output.quality_score}/10")
    print(f"  ✓ Improvements: {len(final.output.improvements_made)}")

    return final.output


# =============================================================================
# ALTERNATIVE: Orchestrator Pattern with Tool Delegation
# =============================================================================

# You can also have one agent orchestrate others using tools.
# This is useful when you want dynamic agent selection.

orchestrator: Agent[None, EditedContent] = Agent(
    ClaudeCodeModel(model="sonnet"),
    output_type=EditedContent,
    system_prompt="""You are a content creation orchestrator.
You have access to specialist agents via tools. Use them to:
1. First research the topic
2. Then write a draft based on research
3. Finally edit and polish the content
Return the final edited content.""",
    retries=3,
)


@orchestrator.tool_plain
async def do_research(topic: str) -> str:
    """
    Delegate research to the research specialist.

    Args:
        topic: The topic to research
    """
    result = await researcher.run(f"Research: {topic}")
    return f"""Research Results:
Topic: {result.output.topic}
Audience: {result.output.target_audience}
Tone: {result.output.tone_recommendation}
Facts:
{chr(10).join(f"- {f}" for f in result.output.key_facts)}"""


@orchestrator.tool_plain
async def do_writing(research_context: str) -> str:
    """
    Delegate writing to the writer specialist.

    Args:
        research_context: Research findings to base the writing on
    """
    result = await writer.run(f"Write content based on:\n{research_context}")
    return f"""Draft:
Title: {result.output.title}
Hook: {result.output.hook}
Body: {result.output.body}
CTA: {result.output.call_to_action}"""


# =============================================================================
# MAIN
# =============================================================================


async def main():
    """Run the multi-agent example."""
    print("\n" + "=" * 60)
    print("MULTI-AGENT CONTENT PIPELINE DEMO")
    print("Using: claude-code-model (Claude Code CLI)")
    print("Cost: $0 with Claude Max subscription")
    print("=" * 60)

    # Example 1: Sequential Pipeline
    topic = "Why type hints make Python code more maintainable"
    result = await create_content(topic)

    print("\n" + "=" * 60)
    print("FINAL CONTENT")
    print("=" * 60)
    print(f"\n{result.title}")
    print("-" * len(result.title))
    print(f"\n{result.content}")
    print(f"\n[Quality: {result.quality_score}/10]")
    print("\nImprovements made:")
    for improvement in result.improvements_made:
        print(f"  • {improvement}")

    # Example 2: Orchestrator Pattern (commented out to save time)
    # Uncomment to try the tool-based orchestration approach:
    #
    # print("\n\nRunning orchestrator pattern...")
    # result2 = await orchestrator.run(
    #     "Create a blog post about async/await in Python"
    # )
    # print(f"Orchestrator result: {result2.output.title}")


if __name__ == "__main__":
    asyncio.run(main())
