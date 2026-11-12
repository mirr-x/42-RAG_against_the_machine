"""  A module for chunking Markdown text into smaller, manageable pieces. """
# pylint: disable=no-name-in-module

import re
import uuid
from pathlib import Path
from typing import Iterator
from scandir_rs import Walk

from src.chunking.models import Chunk, MarkdownChunk


HEADING_PATTERN = re.compile(r"(?m)^#{1,6}[ \t]+.*$")


def _find_split_point(text: str, start: int, limit: int) -> int:
    """Find a useful boundary before the maximum character limit."""
    window = text[start:limit]

    for separator in ("\n\n", "\n", " "):
        position = window.rfind(separator)
        if position > 0:
            return start + position + len(separator)

    return limit


def _find_boundaries(text: str, text_length: int) -> list[int]:
    """Find the beginning of each Markdown heading."""

    headings = []
    for match in HEADING_PATTERN.finditer(text):
        indices = match.start()
        if indices != 0:
            headings.append(indices)

    return [0, *headings, text_length]


def _heading_metadata(text: str, section_start: int) -> tuple[str, int]:
    """Return the heading title and level for a Markdown section."""
    match = re.search(
        r"(?m)^(#{1,6})[ \t]+(.+?)\s*$",
        text[section_start:],
    )
    if match is None or match.start() != 0:
        return "", 0
    return match.group(2), len(match.group(1))


def _chunk_markdown_file(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000,
        ) -> list[Chunk]:
    """Split Markdown into heading-aware chunks within the size limit.

    Args:
        text (str): The Markdown text to be chunked.
        file_path (str): The path to the original Markdown file.
        max_chunk_size (int): The max number of characters per chunk.
    Returns:
        list[Chunk]: normalized Markdown chunks.
    """

    if not text:
        return []

    text_length = len(text)
    boundaries = _find_boundaries(text, text_length)
    markdown_chunks: list[Chunk] = []

    for section_start, section_end in zip(boundaries, boundaries[1:]):
        title, level = _heading_metadata(text, section_start)
        start = section_start

        while start < section_end:
            limit = min(start + max_chunk_size, section_end)

            if limit == section_end:
                end = section_end
            else:
                end = _find_split_point(text, start, limit)

            # Prevent a split from creating an empty chunk.
            if end <= start:
                end = limit

            start_line = text.count("\n", 0, start) + 1
            end_line = text.count("\n", 0, end - 1) + 1
            markdown_chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    content=text[start:end],
                    file_path=file_path,
                    chunk_type="markdown",
                    first_character_index=start,
                    last_character_index=end - 1,
                    metadata=MarkdownChunk(
                        title=title,
                        level=level,
                        start_line=start_line,
                        end_line=end_line,
                    ),
                ),
            )
            start = end

    return markdown_chunks


def iter_markdown_corpus_chunks(
    project_root: Path,
    corpus_root: Path,
    max_chunk_size: int = 2000,
        ) -> Iterator[Chunk]:
    """Yield normalized Markdown chunks from a corpus."""
    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("max_chunk_size must be between 1 and 2000")

    walk_root = project_root / corpus_root
    try:
        output_corpus_root = walk_root.relative_to(project_root)
    except ValueError:
        output_corpus_root = corpus_root
    try:
        walker = Walk(str(walk_root), store=False)
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Walk "
            f"for corpus root '{walk_root}': "
            f"{type(e).__name__}: {e}"
        ) from e

    for cur_dir, _, files in walker:
        for filename in sorted(files):
            if not filename.lower().endswith(".md"):
                continue

            file_path = project_root / corpus_root / cur_dir / filename
            text = file_path.read_text(encoding="utf-8")
            relative_file_path = str(output_corpus_root / cur_dir / filename)

            for chunk in _chunk_markdown_file(
                text=text,
                file_path=relative_file_path,
                max_chunk_size=max_chunk_size,
            ):
                yield chunk
