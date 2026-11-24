"""Tests for metadata-aware BM25 indexing."""

from pathlib import Path

import pytest

from src.indexing.indexing import Indexer
from src.models import Chunk, CodeChunk, MarkdownChunk


def _chunk(content: str, metadata: CodeChunk | MarkdownChunk) -> Chunk:
    return Chunk(
        id="chunk",
        content=content,
        file_path="source",
        chunk_type="code" if isinstance(metadata, CodeChunk) else "markdown",
        first_character_index=0,
        last_character_index=len(content) - 1,
        metadata=metadata,
    )


def _index_documents(
    chunks: list[Chunk],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> list[list[str]]:
    documents: list[list[str]] = []

    def capture_documents(
        tokenized_documents: list[list[str]],
        *,
        b: float = 0.75,
    ) -> object:
        documents.extend(tokenized_documents)
        return object()

    monkeypatch.setattr(
        "src.indexing.indexing.BM25Okapi",
        capture_documents,
    )
    Indexer(
        project_root=tmp_path,
        corpus_docs_root=tmp_path,
        corpus_code_root=tmp_path,
        chunks_file=tmp_path / "chunks.jsonl",
        index_file=tmp_path / "index.pkl",
    )._build_bm25(chunks)
    return documents


def test_bm25_document_includes_python_metadata_and_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk(
        "return value",
        CodeChunk(
            name="lookup_user",
            type="function",
            signature="def lookup_user(user_id: int)",
            parent_context="class UserStore",
        ),
    )

    documents = _index_documents([chunk], tmp_path, monkeypatch)

    assert documents == [
        [
            "lookup_user",
            "lookup",
            "user",
            "def",
            "lookup_user",
            "lookup",
            "user",
            "user_id",
            "user",
            "id",
            "int",
            "class",
            "userstore",
            "return",
            "value",
        ]
    ]


def test_bm25_document_includes_markdown_heading_and_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk(
        "## Installation\nInstall the package.",
        MarkdownChunk(title="Installation", level=2, start_line=1, end_line=2),
    )

    documents = _index_documents([chunk], tmp_path, monkeypatch)

    assert documents == [
        [
            "heading",
            "level",
            "2",
            "installation",
            "installation",
            "install",
            "package",
        ]
    ]


def test_module_source_keeps_imports_searchable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk(
        "import pathlib\nfrom collections import Counter",
        CodeChunk(name="<module>", type="module", signature=""),
    )

    documents = _index_documents([chunk], tmp_path, monkeypatch)

    assert "pathlib" in documents[0]
    assert "collections" in documents[0]
    assert "counter" in documents[0]
