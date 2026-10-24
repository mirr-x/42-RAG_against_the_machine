"""  A module for chunking Markdown text into smaller, manageable pieces. """

import re
from pydantic import BaseModel

HEADING_PATTERN = re.compile(r"(?m)^#{1,6}[ \t]+.*$")


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


def _find_boundaries(text: str, text_length: int) -> list[int]:
    """Find the beginning of each Markdown heading."""

    headings = []
    for match in HEADING_PATTERN.finditer(text):
        indices = match.start()
        if indices != 0:
            headings.append(indices)

    return [0, *headings, text_length]


def chunk_markdown(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> list[MarkdownChunk]:
    """Split Markdown into heading-aware chunks within the size limit.

    Args:
        text (str): The Markdown text to be chunked.
        file_path (str): The path to the original Markdown file.
        max_chunk_size (int): The maximum number of characters per chunk.
    Returns:
        list[MarkdownChunk]: list of MarkdownChunk objs represent's the chunks.
    """
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be at least 1")

    if not text:
        return []

    text_length = len(text)
    boundaries = _find_boundaries(text, text_length)

    # Use headings as preferred section boundaries.
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


if __name__ == "__main__":
    sample_text = """# Heading 1
This is some sample Markdown text.
## Heading 2
```python
print("Hello, World!")
```
### Heading 3
```python
def add(a, b):
    return a + b
```
# Heading 4
This is the last section of the sample Markdown text.
"""
    chunks = chunk_markdown(sample_text, "sample.md", max_chunk_size=50)
    print(f"Total Chunks: {len(chunks)}")
    print(chunks)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"File Path: {chunk.file_path}")
        print(f"First Character Index: {chunk.first_character_index}")
        print(f"Last Character Index: {chunk.last_character_index}")
        print("Text:")
        print(chunk.text)
        print("-" * 40)
