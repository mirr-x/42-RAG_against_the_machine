"""  A module for chunking Markdown text into smaller, manageable pieces. """

import json
import re
import uuid
from pathlib import Path
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


def _chunk_markdown_file(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000,
    ) -> list[MarkdownChunk]:
    """Split Markdown into heading-aware chunks within the size limit.

    Args:
        text (str): The Markdown text to be chunked.
        file_path (str): The path to the original Markdown file.
        max_chunk_size (int): The max number of characters per chunk.
    Returns:
        list[MarkdownChunk]: list of MarkdownChunk objs represent's the chunks.
    """

    if not text:
        return []

    text_length = len(text)
    boundaries = _find_boundaries(text, text_length)
    markdown_chunks: list[MarkdownChunk] = []

    for section_start, section_end in zip(boundaries, boundaries[1:]):
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

            markdown_chunks.append(
                MarkdownChunk(
                    text=text[start:end],
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end - 1,
                )
            )
            start = end

    return markdown_chunks


def iter_markdown_corpus_chunks(
    project_root: Path,
    corpus_root: Path,
    max_chunk_size: int = 2000,
    ):
    """Yield normalized Markdown chunks from a corpus."""
    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("max_chunk_size must be between 1 and 2000")

    walk_root = project_root / corpus_root
    try:
        output_corpus_root = corpus_root.relative_to(project_root)
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

            for markdown_chunk in _chunk_markdown_file(
                text=text,
                file_path=relative_file_path,
                max_chunk_size=max_chunk_size,
            ):
                yield Chunk(
                    id=str(uuid.uuid4()),
                    content=markdown_chunk.text,
                    file_path=markdown_chunk.file_path,
                    chunk_type="markdown",
                    first_character_index=markdown_chunk.first_character_index,
                    last_character_index=markdown_chunk.last_character_index,
                    metadata=markdown_chunk,
                )
