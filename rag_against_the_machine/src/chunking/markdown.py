"""  A module for chunking Markdown text into smaller, manageable pieces. """

import re
from pathlib import Path
from scandir_rs import Walk
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
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be at least 1")

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


def chunk_markdown_corpus_with_rust(
    corpus_root: Path,
    output_path: Path,
    max_chunk_size: int = 2000
) -> int:
    """ Chunk a corpus of Markdown files using Rust for performance.

    Args:
        corpus_root (Path): root dir containing the md files to be chunked.
        project_root (Path): The root directory of the project.
        output_path (Path): path to the output file where chunks will be saved.
        max_chunk_size (int, optional): The max number of characters per chunk.

    Returns:
        int: The number of chunks created.
    """
    if output_path.exists():
        print(f"'{output_path}' already exists. Skipping chunking step.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_count = 0

    walker = Walk(str(corpus_root), store=False)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for cur_dir, _, files in walker:
            for filename in sorted(files):
                if filename.lower().endswith('.md'):
                    file_path = corpus_root / cur_dir / filename
                    
                    print(f"Chunking '{file_path}'...")
                    text = file_path.read_text(encoding='utf-8')
                    chunks = _chunk_markdown_file(
                        text=text,
                        file_path=str(file_path),
                        max_chunk_size=max_chunk_size
                    )
                    for chunk in chunks:
                        output_file.write(chunk.model_dump_json() + '\n')
                        chunk_count += 1
    return chunk_count
