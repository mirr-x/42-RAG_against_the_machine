"""Tests for corpus-level chunking."""

from pathlib import Path

import pytest

from src.chunking.chuncking import Chunking


@pytest.mark.parametrize("missing_corpus", ["python", "markdown"])
def test_missing_corpus_fails_before_creating_output(
    tmp_path: Path,
    missing_corpus: str,
) -> None:
    python_corpus = Path("data/raw/vllm-0.10.1/vllm")
    markdown_corpus = Path("data/raw/vllm-0.10.1/docs")
    missing_root = (
        python_corpus if missing_corpus == "python" else markdown_corpus
    )
    existing_root = (
        markdown_corpus if missing_corpus == "python" else python_corpus
    )
    (tmp_path / existing_root).mkdir(parents=True)

    output_path = tmp_path / "data/processed/chunks.jsonl"
    chunking = Chunking(
        project_root=tmp_path,
        corpus_docs_root=markdown_corpus,
        corpus_code_root=python_corpus,
        output_path=output_path,
    )

    with pytest.raises(FileNotFoundError, match="corpus directory does not exist"):
        chunking.run()

    assert not output_path.exists()
