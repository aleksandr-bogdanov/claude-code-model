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
            print(f"\nAssistant: {result.output}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
