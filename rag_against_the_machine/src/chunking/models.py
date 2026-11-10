"""Pydantic models used by code and Markdown chunkers."""

from typing import Optional

from pydantic import BaseModel


class MarkdownChunk(BaseModel):
    """A chunk of Markdown linked to its original file."""

    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


class CodeChunk(BaseModel):
    """A chunk of Python code linked to its original file."""

    name: str
    type: str
    signature: str
    parent_context: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class Chunk(BaseModel):
    """A standardized chunk representation for Python and Markdown files."""

    id: str
    content: str
    file_path: str
    chunk_type: str
    first_character_index: int
    last_character_index: int
    metadata: MarkdownChunk | CodeChunk
