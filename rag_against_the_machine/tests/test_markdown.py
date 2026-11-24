"""Tests for Markdown chunking behavior and boundary handling."""

# Individual test names document the behavior under test.
# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest

from src.chunking.markdown import (
    _chunk_markdown_file,
    iter_markdown_corpus_chunks,
)
from src.models import MarkdownChunk


def test_empty_markdown_returns_no_chunks() -> None:
    assert not _chunk_markdown_file("", "docs/empty.md")


def test_chunk_preserves_text_and_character_indices() -> None:
    text = "# Introduction\n\nThis is the introduction.\n\n# Usage\n\nRun it."

    chunks = _chunk_markdown_file(text, "docs/guide.md", max_chunk_size=2000)

    assert [chunk.content for chunk in chunks] == [
        "# Introduction\n\nThis is the introduction.\n\n",
        "# Usage\n\nRun it.",
    ]
    assert all(isinstance(chunk.metadata, MarkdownChunk) for chunk in chunks)
    indices = [
        (chunk.first_character_index, chunk.last_character_index)
        for chunk in chunks
    ]
    assert indices == [
        (0, 42),
        (43, len(text) - 1),
    ]
    assert all(chunk.file_path == "docs/guide.md" for chunk in chunks)


def test_long_section_splits_on_newline_before_size_limit() -> None:
    text = "first line\nsecond line\nthird line"

    chunks = _chunk_markdown_file(text, "docs/guide.md", max_chunk_size=20)

    assert [chunk.content for chunk in chunks] == [
        "first line\n",
        "second line\n",
        "third line",
    ]
    assert all(len(chunk.content) <= 20 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == text


def test_corpus_writer_outputs_relative_jsonl_chunks(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    markdown_file = corpus_root / "nested" / "guide.md"
    markdown_file.parent.mkdir(parents=True)
    markdown_file.write_text("# Guide\n\nContent", encoding="utf-8")

    chunks = list(
        iter_markdown_corpus_chunks(
            project_root=tmp_path,
            corpus_root=corpus_root,
            max_chunk_size=2000,
        )
    )

    assert len(chunks) == 1
    assert chunks[0].file_path == "corpus/nested/guide.md"
    assert chunks[0].content == "# Guide\n\nContent"


@pytest.mark.parametrize("max_chunk_size", [0, 2001])
def test_corpus_writer_rejects_invalid_chunk_size(
    tmp_path: Path, max_chunk_size: int
) -> None:
    with pytest.raises(ValueError, match="between 1 and 2000"):
        list(iter_markdown_corpus_chunks(
            project_root=tmp_path,
            corpus_root=tmp_path,
            max_chunk_size=max_chunk_size,
        ))
