"""Tests for BM25 retrieval and dataset search output."""

import json
import pickle
import uuid
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi

from src.indexing.indexing import Indexer
from src.models import Chunk, CodeChunk, MarkdownChunk, MinimalSource
from src.retrieval.retrieval import Retriever


def _write_index(path: Path) -> None:
    chunks = [
        Chunk(
            id="one",
            content="alpha beta",
            file_path="vllm/one.py",
            chunk_type="code",
            first_character_index=10,
            last_character_index=20,
            metadata=CodeChunk(
                name="one",
                type="function",
                signature="def one()",
            ),
        ),
        Chunk(
            id="two",
            content="gamma delta",
            file_path="vllm/two.py",
            chunk_type="code",
            first_character_index=30,
            last_character_index=40,
            metadata=CodeChunk(
                name="two",
                type="function",
                signature="def two()",
            ),
        ),
        Chunk(
            id="three",
            content="epsilon zeta",
            file_path="vllm/three.py",
            chunk_type="code",
            first_character_index=50,
            last_character_index=60,
            metadata=CodeChunk(
                name="three",
                type="function",
                signature="def three()",
            ),
        ),
    ]
    tokenized_chunks = [Indexer.tokenize(chunk.content) for chunk in chunks]
    with path.open("wb") as file:
        pickle.dump(
            {
                "version": 3,
                "algorithm": "rank-bm25",
                "bm25_by_type": {"code": BM25Okapi(tokenized_chunks)},
                "chunks_by_type": {"code": chunks},
                "bm25_all": BM25Okapi(tokenized_chunks),
                "chunks_all": chunks,
            },
            file,
        )


def _write_typed_index(path: Path) -> None:
    chunks = [
        Chunk(
            id="code",
            content="codeonly",
            file_path="vllm/code.py",
            chunk_type="code",
            first_character_index=0,
            last_character_index=7,
            metadata=CodeChunk(
                name="code_function",
                type="function",
                signature="def code_function()",
            ),
        ),
        Chunk(
            id="markdown",
            content="docsneedle",
            file_path="docs/guide.md",
            chunk_type="markdown",
            first_character_index=0,
            last_character_index=9,
            metadata=MarkdownChunk(
                title="Guide",
                level=1,
                start_line=1,
                end_line=1,
            ),
        ),
    ]
    index = Indexer(
        project_root=path.parent,
        corpus_docs_root=path.parent,
        corpus_code_root=path.parent,
        chunks_file=path.parent / "chunks.jsonl",
        index_file=path,
    )._build_bm25(chunks)
    with path.open("wb") as file:
        pickle.dump(index, file)


def test_query_returns_ranked_sources_and_offsets(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25.pkl"
    _write_index(index_path)

    results = Retriever(index_path).query("alpha", k=1)

    assert len(results) == 1
    assert results[0].model_dump() == {
        "file_path": "vllm/one.py",
        "first_character_index": 10,
        "last_character_index": 20,
    }


def test_query_can_restrict_results_to_code_or_markdown(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "typed-index.pkl"
    _write_typed_index(index_path)
    retriever = Retriever(index_path)

    code_results = retriever.query(
        "docsneedle",
        k=1,
        chunk_type="code",
    )
    markdown_results = retriever.query(
        "codeonly",
        k=1,
        chunk_type="markdown",
    )

    assert code_results[0].file_path == "vllm/code.py"
    assert markdown_results[0].file_path == "docs/guide.md"


def test_unfiltered_query_merges_typed_indexes(tmp_path: Path) -> None:
    index_path = tmp_path / "typed-index.pkl"
    _write_typed_index(index_path)

    results = Retriever(index_path).query("codeonly docsneedle", k=2)

    assert {result.file_path for result in results} == {
        "vllm/code.py",
        "docs/guide.md",
    }


def test_typed_index_loading_does_not_revalidate_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path = tmp_path / "typed-index.pkl"
    _write_typed_index(index_path)

    def reject_validation(*args: object, **kwargs: object) -> None:
        raise AssertionError("Persisted chunks should be loaded directly.")

    monkeypatch.setattr(Chunk, "model_validate", reject_validation)

    retriever = Retriever(index_path)

    assert len(retriever.chunks) == 2


def test_resolve_sources_maps_search_offsets_to_chunks(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "bm25.pkl"
    _write_index(index_path)

    chunks = Retriever(index_path).resolve_sources(
        [
            MinimalSource(
                file_path="vllm/one.py",
                first_character_index=10,
                last_character_index=20,
            )
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].id == "one"
    assert chunks[0].content == "alpha beta"


def test_resolve_sources_rejects_unknown_search_offsets(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "bm25.pkl"
    _write_index(index_path)

    with pytest.raises(ValueError, match="does not match any indexed chunk"):
        Retriever(index_path).resolve_sources(
            [
                MinimalSource(
                    file_path="vllm/missing.py",
                    first_character_index=0,
                    last_character_index=10,
                )
            ]
        )


@pytest.mark.parametrize(
    ("query", "k", "error"),
    [
        ("", 1, "cannot be empty"),
        ("the and", 1, "at least one valid token"),
        ("alpha", 0, "positive integer"),
        ("alpha", 4, "greater than the number of chunks"),
    ],
)
def test_query_rejects_invalid_inputs(
    tmp_path: Path,
    query: str,
    k: int,
    error: str,
) -> None:
    index_path = tmp_path / "bm25.pkl"
    _write_index(index_path)

    with pytest.raises(ValueError, match=error):
        Retriever(index_path).query(query, k)


def test_query_rejects_missing_index(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError,
        match=r"Index file not found: .*Build it first with",
    ):
        Retriever(tmp_path / "missing.pkl").query("alpha")


def test_query_accepts_k_larger_than_twenty_five(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25.pkl"
    chunks = [
        Chunk(
            id=str(index),
            content=f"term{index}",
            file_path=f"vllm/file{index}.py",
            chunk_type="code",
            first_character_index=0,
            last_character_index=5,
            metadata=CodeChunk(
                name=f"function{index}",
                type="function",
                signature=f"def function{index}():",
            ),
        )
        for index in range(30)
    ]
    tokenized_chunks = [Indexer.tokenize(chunk.content) for chunk in chunks]
    with index_path.open("wb") as file:
        pickle.dump(
            {
                "version": 3,
                "algorithm": "rank-bm25",
                "bm25_by_type": {"code": BM25Okapi(tokenized_chunks)},
                "chunks_by_type": {"code": chunks},
                "bm25_all": BM25Okapi(tokenized_chunks),
                "chunks_all": chunks,
            },
            file,
        )

    results = Retriever(index_path).query("term10", k=30)

    assert len(results) == 30


def test_query_batch_preserves_ids_and_saves_json(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25.pkl"
    _write_index(index_path)
    dataset_path = tmp_path / "dataset.json"
    question_id = str(uuid.uuid4())
    dataset_path.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {
                        "question_id": question_id,
                        "question": "alpha",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result_path = Retriever(index_path).query_batch(
        dataset_path,
        tmp_path / "results",
        k=1,
    )

    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_path.name == dataset_path.name
    assert output["k"] == 1
    assert output["search_results"][0]["question_id"] == question_id
    assert output["search_results"][0]["retrieved_sources"][0][
        "file_path"
    ] == "vllm/one.py"


