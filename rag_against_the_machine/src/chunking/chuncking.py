""" A module for chunking code and text into smaller pieces. """

from pydantic import BaseModel
from pathlib import Path

from src.chunking.models import Chunk
from src.chunking.markdown import chunk_markdown_corpus_with_rust
from src.chunking.python import chunk_python_corpus_with_rust


class Chunking:
    """A class for chunking Python code and Markdown files into chunks."""
    pass
    def __init__(
        self,
        project_root: Path,
        corpus_docs_root: Path,
        corpus_code_root: Path,
        output_path: Path,
        max_chunk_size=1000
    ) -> None:
        self.project_root = project_root
        self.corpus_docs_root = corpus_docs_root
        self.corpus_code_root = corpus_code_root
        self.output_path = output_path
        self.max_chunk_size = max_chunk_size

    def run(self) -> None:
        """Run the chunking process for both Python and Markdown files."""
        try:
            code_chunk_count = chunk_python_corpus_with_rust(
                project_root=self.project_root,
                corpus_root=self.corpus_code_root,
                output_path=self.output_path,
                max_chunk_size=self.max_chunk_size
            )
            print(f"Code chunks written: {code_chunk_count}")

            # markdown_chunk_count = chunk_markdown_corpus_with_rust(
            #     project_root=self.project_root,
            #     corpus_root=self.corpus_docs_root,
            #     output_path=self.output_path,
            #     max_chunk_size=self.max_chunk_size
            # )
            # print(f"Markdown chunks written: {markdown_chunk_count}")

        except ValueError as e:
            raise ValueError(f"Error during chunking: {e}")
        
        
