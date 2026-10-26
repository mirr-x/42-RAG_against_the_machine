"""Tests for Markdown chunking behavior and boundary handling."""

# Individual test names document the behavior under test.
# pylint: disable=missing-function-docstring

import pytest

from src.chunking.markdown import (
    MarkdownChunk,
    _find_boundaries,
    _find_split_point,
    _chunk_markdown_file,
)


def test_empty_markdown_returns_no_chunks() -> None:
    assert not _chunk_markdown_file("", "empty.md")


@pytest.mark.parametrize("max_chunk_size", [0, -1, -100])
def test_non_positive_chunk_size_is_rejected(max_chunk_size: int) -> None:
    with pytest.raises(ValueError, match="max_chunk_size must be at least 1"):
        _chunk_markdown_file("content", "document.md", max_chunk_size)


@pytest.mark.parametrize(
    ("text", "max_chunk_size"),
    [
        ("a", 1),
        ("exactly ten", 11),
        ("line one\nline two", 18),
    ],
)
def test_input_within_limit_is_one_unchanged_chunk(
    text: str, max_chunk_size: int
) -> None:
    chunks = _chunk_markdown_file(text, "document.md", max_chunk_size)

    assert len(chunks) == 1
    assert chunks[0] == MarkdownChunk(
        text=text,
        file_path="document.md",
        first_character_index=0,
        last_character_index=len(text) - 1,
    )


def test_unbroken_text_is_split_at_the_hard_limit() -> None:
    text = "abcdefghij"

    chunks = _chunk_markdown_file(text, "document.md", max_chunk_size=3)

    assert [chunk.text for chunk in chunks] == ["abc", "def", "ghi", "j"]
    assert [
        (chunk.first_character_index, chunk.last_character_index)
        for chunk in chunks
    ] == [
        (0, 2),
        (3, 5),
        (6, 8),
        (9, 9),
    ]


def test_split_point_prefers_paragraph_then_line_then_word_boundary() -> None:
    assert _find_split_point("first\n\nsecond", 0, 12) == 7
    assert _find_split_point("first\nsecond", 0, 10) == 6
    assert _find_split_point("first second", 0, 10) == 6


def test_split_point_falls_back_to_limit_when_no_separator_exists() -> None:
    assert _find_split_point("abcdefgh", 0, 4) == 4


def test_split_point_respects_nonzero_start_offset() -> None:
    assert _find_split_point("prefix first second", 7, 17) == 13


def test_heading_boundaries_include_document_start_and_end() -> None:
    text = "# First\nfirst body\n## Second\nsecond body"

    assert _find_boundaries(text, len(text)) == [0, 19, len(text)]


def test_headings_split_sections_without_losing_heading_text() -> None:
    text = "# First\nfirst body\n## Second\nsecond body"

    chunks = _chunk_markdown_file(text, "guide.md", max_chunk_size=len(text))

    assert [chunk.text for chunk in chunks] == [
        "# First\nfirst body\n",
        "## Second\nsecond body",
    ]
    assert [
        (chunk.first_character_index, chunk.last_character_index)
        for chunk in chunks
    ] == [
        (0, 18),
        (19, len(text) - 1),
    ]


@pytest.mark.parametrize(
    "text",
    ["heading without hash", "#no-space", "####### too many hashes"],
)
def test_heading_pattern_rejects_invalid_atx_heading_syntax(text: str) -> None:
    assert _find_boundaries(text, len(text)) == [0, len(text)]


def test_heading_pattern_accepts_tabs_after_hashes() -> None:
    text = "text\n#\tvalid heading"

    assert _find_boundaries(text, len(text)) == [0, 5, len(text)]


def test_headings_after_blank_lines_are_detected() -> None:
    text = "Intro\n\n### Details\nbody"

    assert _find_boundaries(text, len(text)) == [0, 7, len(text)]


def test_heading_at_document_start_is_not_a_duplicate_boundary() -> None:
    text = "# Heading\nbody"

    assert _find_boundaries(text, len(text)) == [0, len(text)]


def test_unicode_offsets_use_python_character_indices() -> None:
    text = "café 🚀\nbody"

    chunks = _chunk_markdown_file(text, "unicode.md", max_chunk_size=5)

    assert "".join(chunk.text for chunk in chunks) == text
    assert chunks[0].text == "café "
    assert chunks[0].first_character_index == 0
    assert chunks[0].last_character_index == 4


def test_chunks_reconstruct_text_and_have_contiguous_offsets() -> None:
    text = "# Intro\nA paragraph.\n\n## Details\nMore words here."

    chunks = _chunk_markdown_file(text, "guide.md", max_chunk_size=9)

    assert chunks
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(len(chunk.text) <= 9 for chunk in chunks)
    assert all(
        chunk.text
        == text[
            slice(chunk.first_character_index, chunk.last_character_index + 1)
        ]
        for chunk in chunks
    )
    assert all(
        current.last_character_index + 1 == following.first_character_index
        for current, following in zip(chunks, chunks[1:])
    )
    assert chunks[0].first_character_index == 0
    assert chunks[-1].last_character_index == len(text) - 1


def test_chunks_preserve_file_path_on_every_chunk() -> None:
    chunks = _chunk_markdown_file(
        "abcdefgh", "/tmp/manual.md", max_chunk_size=2
    )

    assert chunks
    assert {chunk.file_path for chunk in chunks} == {"/tmp/manual.md"}


def test_markdown_chunk_requires_expected_fields() -> None:
    chunk = MarkdownChunk(
        text="body",
        file_path="guide.md",
        first_character_index=4,
        last_character_index=7,
    )

    assert chunk.model_dump() == {
        "text": "body",
        "file_path": "guide.md",
        "first_character_index": 4,
        "last_character_index": 7,
    }
