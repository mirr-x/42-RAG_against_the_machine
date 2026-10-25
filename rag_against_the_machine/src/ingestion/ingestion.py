
from pathlib import Path

from src.chunking.markdown import MarkdownChunk, chunk_markdown


def discover_markdown_files(corpus_root: Path) -> list[Path]:
    """Find Markdown documentation files, without scanning source code."""
    docs_dir = corpus_root / "docs"
    readme = corpus_root / "README.md"

    files: list[Path] = []

    if docs_dir.is_dir():
        files.extend(docs_dir.rglob("*.md"))

    if readme.is_file():
        files.append(readme)

    return sorted(files)


def read_markdown_corpus(
    corpus_root: Path,
    project_root: Path,
    max_chunk_size: int = 2000,
) -> list[MarkdownChunk]:
    """ Read Markdown files from the corpus and split them into chunks.

    Args:
        corpus_root (Path): root directory of the corpus containing md files.
        project_root (Path): The root directory of the project.
        max_chunk_size (int, optional): The max size of each chunk.

    Returns:
        list[MarkdownChunk]: A list of Markdown chunks.
    """
    files = discover_markdown_files(corpus_root)
    chunks: list[MarkdownChunk] = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")

        # Keep the path relative to our assignment's project root.
        relative_path = file_path.relative_to(project_root).as_posix()

        file_chunks = chunk_markdown(
            text=text,
            file_path=relative_path,
            max_chunk_size=max_chunk_size,
        )

        chunks.extend(file_chunks)

    return chunks
