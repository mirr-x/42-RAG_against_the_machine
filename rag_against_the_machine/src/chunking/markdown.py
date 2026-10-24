
import re

from pydantic import BaseModel, Field


class MarkdownChunk(BaseModel):
    """A chunk of Markdown linked to its original file."""

    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


def _find_split_point(text: str, start: int, limit: int) -> int:
    """Find a useful boundary before the maximum character limit."""
    window = text[start:limit]

    for separator in ("\n\n", "\n", " "):
        position = window.rfind(separator)
        if position > 0:
            return start + position + len(separator)

    return limit


def chunk_markdown(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> list[MarkdownChunk]:
    """Split Markdown into heading-aware chunks within the size limit."""
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be at least 1")

    if not text:
        return []

    # Find the beginning of each Markdown heading.
    heading_pattern = re.compile(r"(?m)^#{1,6}[ \t]+.*$")
    headings = [match.start() for match in heading_pattern.finditer(text)]

    # Use headings as preferred section boundaries.
    boundaries = sorted(set([0, *headings, len(text)]))
    chunks: list[MarkdownChunk] = []

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

            chunks.append(
                MarkdownChunk(
                    text=text[start:end],
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end - 1,
                )
            )
            start = end

    return chunks
