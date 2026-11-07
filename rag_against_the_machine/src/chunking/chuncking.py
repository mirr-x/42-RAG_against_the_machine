""" A module for chunking code and text into smaller pieces. """

from pydantic import BaseModel

from src.chunking.models import CodeChunk, MarkdownChunk


class Chunk(BaseModel):
    """ A standardized chunk representation both Python and Markdown files."""
    id: str
    content: str
    file_path: str
    chunk_type: str
    start_line: int
    end_line: int
    metadata: MarkdownChunk | CodeChunk


class Chunking:
    """A class for chunking Python code and Markdown files into chunks."""
    pass
