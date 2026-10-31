"""Validation helpers for generated Markdown chunks."""

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.chunking.markdown import MarkdownChunk


@dataclass
class ValidationReport:
    """Summary of JSONL chunk validation."""

    total_chunks: int = 0
    source_files: int = 0
    minimum_chunk_size: int = 0
    maximum_chunk_size: int = 0
    average_chunk_size: float = 0.0
    errors: list[str] = field(default_factory=list)
    files_with_reconstruction_problems: set[str] = field(default_factory=set)

    @property
    def validation_error_count(self) -> int:
        """Return the number of validation errors."""
        return len(self.errors)


def _record_is_valid(record: Any, line_number: int, report: ValidationReport) -> bool:
    if not isinstance(record, dict):
        report.errors.append(f"line {line_number}: expected a JSON object")
        return False

    required_fields = {
        "text",
        "file_path",
        "first_character_index",
        "last_character_index",
    }
    missing_fields = required_fields - record.keys()
    if missing_fields:
        report.errors.append(
            f"line {line_number}: missing fields {sorted(missing_fields)}"
        )
        return False

    if not isinstance(record["text"], str):
        report.errors.append(f"line {line_number}: text must be a string")
        return False
    if not isinstance(record["file_path"], str):
        report.errors.append(f"line {line_number}: file_path must be a string")
        return False
    for field_name in ("first_character_index", "last_character_index"):
        if isinstance(record[field_name], bool) or not isinstance(
            record[field_name], int
        ):
            report.errors.append(
                f"line {line_number}: {field_name} must be an integer"
            )
            return False

    try:
        MarkdownChunk.model_validate(record)
    except ValidationError as error:
        report.errors.append(f"line {line_number}: invalid chunk: {error}")
        return False
    return True


def validate_chunk_jsonl(
    project_root: Path,
    corpus_root: Path,
    chunks_path: Path,
    max_chunk_size: int = 2000,
) -> ValidationReport:
    """Validate generated chunks against their Markdown source files."""
    report = ValidationReport()
    source_files = {
        path.relative_to(project_root).as_posix(): path
        for path in corpus_root.rglob("*.md")
    }
    report.source_files = len(source_files)
    chunks_by_file: dict[str, list[MarkdownChunk]] = defaultdict(list)

    try:
        lines = chunks_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        report.errors.append(f"could not read {chunks_path}: {error}")
        return report

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            report.errors.append(f"line {line_number}: blank JSONL line")
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            report.errors.append(
                f"line {line_number}: invalid JSON at column {error.colno}"
            )
            continue
        if not _record_is_valid(record, line_number, report):
            continue

        chunk = MarkdownChunk.model_validate(record)
        if not chunk.text:
            report.errors.append(f"line {line_number}: chunk text is empty")
            continue
        if len(chunk.text) > max_chunk_size:
            report.errors.append(
                f"line {line_number}: chunk is {len(chunk.text)} characters; "
                f"maximum is {max_chunk_size}"
            )
        if Path(chunk.file_path).is_absolute() or ".." in Path(chunk.file_path).parts:
            report.errors.append(
                f"line {line_number}: file_path must be project-relative: "
                f"{chunk.file_path}"
            )
            continue
        source_path = project_root / chunk.file_path
        if not source_path.is_file():
            report.errors.append(
                f"line {line_number}: source file does not exist: "
                f"{chunk.file_path}"
            )
            continue
        chunks_by_file[chunk.file_path].append(chunk)

    sizes = [len(chunk.text) for chunks in chunks_by_file.values() for chunk in chunks]
    report.total_chunks = len(sizes)
    if sizes:
        report.minimum_chunk_size = min(sizes)
        report.maximum_chunk_size = max(sizes)
        report.average_chunk_size = sum(sizes) / len(sizes)

    for file_path, source_path in source_files.items():
        source_text = source_path.read_text(encoding="utf-8")
        chunks = sorted(
            chunks_by_file.get(file_path, []),
            key=lambda chunk: chunk.first_character_index,
        )
        if not chunks:
            if source_text:
                report.errors.append(f"{file_path}: no chunks found for non-empty file")
                report.files_with_reconstruction_problems.add(file_path)
            continue

        expected_start = 0
        reconstructed: list[str] = []
        file_has_problem = False
        for chunk in chunks:
            expected_length = chunk.last_character_index - chunk.first_character_index + 1
            if expected_length != len(chunk.text):
                report.errors.append(
                    f"{file_path}: offset length does not match chunk text length"
                )
                file_has_problem = True
            if chunk.first_character_index != expected_start:
                report.errors.append(
                    f"{file_path}: overlap or gap before offset "
                    f"{chunk.first_character_index}; expected {expected_start}"
                )
                file_has_problem = True
            if (
                chunk.first_character_index < 0
                or chunk.last_character_index >= len(source_text)
                or chunk.first_character_index > chunk.last_character_index
            ):
                report.errors.append(f"{file_path}: chunk offsets are outside source text")
                file_has_problem = True
                continue
            source_slice = source_text[
                chunk.first_character_index : chunk.last_character_index + 1
            ]
            if source_slice != chunk.text:
                report.errors.append(
                    f"{file_path}: chunk text differs from source at offset "
                    f"{chunk.first_character_index}"
                )
                file_has_problem = True
            reconstructed.append(chunk.text)
            expected_start = chunk.last_character_index + 1

        if expected_start != len(source_text) or "".join(reconstructed) != source_text:
            report.errors.append(f"{file_path}: source reconstruction does not match")
            file_has_problem = True
        if file_has_problem:
            report.files_with_reconstruction_problems.add(file_path)

    return report


def format_validation_report(report: ValidationReport) -> str:
    """Format the validation metrics for a test failure or CLI report."""
    return "\n".join(
        [
            f"Total chunks: {report.total_chunks}",
            f"Source files: {report.source_files}",
            f"Smallest chunk: {report.minimum_chunk_size}",
            f"Largest chunk: {report.maximum_chunk_size}",
            f"Average chunk: {report.average_chunk_size:.2f}",
            f"Validation errors: {report.validation_error_count}",
            "Errors:",
            *(report.errors or ["none"]),
        ]
    )