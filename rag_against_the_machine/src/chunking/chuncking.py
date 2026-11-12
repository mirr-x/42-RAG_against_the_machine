""" A module for chunking code and text into smaller pieces. """

from typing import Iterator
from pathlib import Path
from tqdm import tqdm

from src.chunking.markdown import iter_markdown_corpus_chunks
from src.chunking.python import iter_python_corpus_chunks


class Chunking:
    """A class for chunking Python code and Markdown files into chunks."""

    def __init__(
        self,
        project_root: Path,
        corpus_docs_root: Path,
        corpus_code_root: Path,
        output_path: Path,
        max_chunk_size: int = 1000
            ) -> None:
        self.project_root = project_root
        self.corpus_docs_root = corpus_docs_root
        self.corpus_code_root = corpus_code_root
        self.output_path = output_path
        self.max_chunk_size = max_chunk_size

    def run(self) -> None:
        """Run the chunking process for both Python and Markdown files."""
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            code_chunk_count = 0
            markdown_chunk_count = 0
            code_chunks_iterator: Iterator = iter_python_corpus_chunks(
                self.project_root,
                self.corpus_code_root,
                self.max_chunk_size,
            )
            docs_chunks_iterator: Iterator = iter_markdown_corpus_chunks(
                self.project_root,
                self.corpus_docs_root,
                self.max_chunk_size,
            )

            with self.output_path.open("w", encoding="utf-8") as output_file:
                for chunk in tqdm(
                    code_chunks_iterator, desc="Chunking code", unit=" chunk"
                ):
                    output_file.write(
                        chunk.model_dump_json(exclude_none=True) + "\n"
                    )
                    code_chunk_count += 1

                for chunk in tqdm(
                    docs_chunks_iterator, desc="Chunking docs", unit=" chunk"
                ):
                    output_file.write(
                        chunk.model_dump_json(exclude_none=True) + "\n"
                    )
                    markdown_chunk_count += 1

        except (OSError, UnicodeError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"Chunking failed while writing '{self.output_path}': "
                f"{type(error).__name__}: {error}"
            ) from error
