"""Tests for preparing retrieved chunks as generation context."""

from src.generation.generation import Generation
from src.generation.llm import LLM, SYSTEM_PROMPT
from src.models import Chunk, CodeChunk, MarkdownChunk


def test_build_context_includes_useful_code_metadata_and_content() -> None:
    chunk = Chunk(
        id="code-1",
        content="def answer():\n    return 42\n",
        file_path="vllm/engine.py",
        chunk_type="code",
        first_character_index=0,
        last_character_index=29,
        metadata=CodeChunk(
            name="answer",
            type="function",
            signature="def answer()",
            parent_context="class Engine",
            start_line=10,
            end_line=11,
        ),
    )

    context = Generation.build_context([chunk])

    assert context == [{
        "file_path": "vllm/engine.py",
        "text": (
            "Python function: answer\n"
            "Signature: def answer()\n"
            "Parent context: class Engine\n"
            "Lines: 10-11\n\n"
            "def answer():\n    return 42"
        ),
    }]


def test_build_context_includes_markdown_heading_and_lines() -> None:
    chunk = Chunk(
        id="markdown-1",
        content="Use the engine to generate text.",
        file_path="docs/usage.md",
        chunk_type="markdown",
        first_character_index=0,
        last_character_index=32,
        metadata=MarkdownChunk(
            title="Usage",
            level=2,
            start_line=4,
            end_line=6,
        ),
    )

    context = Generation.build_context([chunk])

    assert context[0]["text"] == (
        "Markdown section: Usage (level 2, lines 4-6)\n\n"
        "Use the engine to generate text."
    )


def test_prompt_includes_system_instructions_and_cited_context() -> None:
    llm = LLM.__new__(LLM)

    prompt = llm.build_qwen_prompt(
        "How does answer work?",
        [{"file_path": "vllm/engine.py", "text": "Python function: answer"}],
    )

    assert SYSTEM_PROMPT in prompt
    assert "untrusted reference data" in prompt
    assert "[1] (vllm/engine.py)" in prompt
    assert "### Question\nHow does answer work?" in prompt
