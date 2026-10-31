"""Validation tests for the generated Markdown chunk corpus."""

import json
from pathlib import Path

from src.chunking.validation import format_validation_report, validate_chunk_jsonl


ROOT = Path(__file__).parents[1]
CORPUS_ROOT = ROOT / "data/raw/vllm-0.10.1/docs"
CHUNKS_PATH = ROOT / "data/raw/processed/chunks.jsonl"


def test_generated_chunks_validate() -> None:
    report = validate_chunk_jsonl(ROOT, CORPUS_ROOT, CHUNKS_PATH)

    assert not report.errors, format_validation_report(report)
    assert report.total_chunks > 0
    assert report.source_files > 0
    assert not report.files_with_reconstruction_problems


def test_empty_source_file_requires_no_chunks(tmp_path: Path) -> None:
    corpus_root = tmp_path / "docs"
    source_path = corpus_root / "empty.md"
    source_path.parent.mkdir()
    source_path.write_text("", encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    report = validate_chunk_jsonl(tmp_path, corpus_root, chunks_path)

    assert not report.errors, format_validation_report(report)
    assert report.total_chunks == 0
    assert report.source_files == 1


def test_malformed_records_are_reported_with_line_numbers(tmp_path: Path) -> None:
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "guide.md").write_text("content", encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "not json\n"
        + json.dumps({"text": "content", "file_path": "docs/guide.md"})
        + "\n",
        encoding="utf-8",
    )

    report = validate_chunk_jsonl(tmp_path, corpus_root, chunks_path)

    assert report.validation_error_count == 2
    assert "line 1: invalid JSON" in report.errors
    assert "line 2: missing fields" in report.errors