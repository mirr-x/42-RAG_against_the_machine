"""Build and persist a BM25 index for the chunk corpus."""

import re
import pickle

from rank_bm25 import BM25Okapi
from pathlib import Path
from tqdm import tqdm

from src.chunking.chuncking import Chunking
from src.chunking.models import Chunk

TOKEN_PATTERN = re.compile(r'[A-Za-z0-9_]+')

USELESS_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}


class Indexer:
    """Create the chunk corpus and persist a searchable BM25 index."""

    def __init__(
        self,
        project_root: Path,
        corpus_docs_root: Path,
        corpus_code_root: Path,
        chunks_file: Path,
        index_file: Path,
        max_chunk_size: int = 2000,
            ) -> None:
        self.project_root = project_root
        self.corpus_docs_root = corpus_docs_root
        self.corpus_code_root = corpus_code_root
        self.chunks_file = chunks_file
        self.index_file = index_file
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize content by keep related words and discard useless words"""
        tokens: list[str] = []

        for word in TOKEN_PATTERN.findall(text.lower()):
            if word not in USELESS_WORDS:
                tokens.append(word)

            if '_' in word:
                for part in word.split("_"):
                    if part and part not in USELESS_WORDS:
                        tokens.append(part)

        return tokens

    def _build_bm25(self, chunks: list[Chunk]) -> dict[str, object]:
        """Build a ready-to-search BM25 index with ``rank-bm25``."""
        print('')
        tokenized_documents = [
            self._tokenize(chunk.content)
            for chunk in tqdm(
                chunks, desc="Tokenizing chunks", unit=" chunk", ascii="-#"
            )
        ]
        bm25 = BM25Okapi(tokenized_documents)
        return {
            "version": 1,
            "algorithm": "rank-bm25",
            "bm25": bm25,
            "chunks": chunks,
        }

    def _load_chunks(self) -> list[Chunk]:
        """Load and validate one JSONL chunk record per line."""
        if not self.chunks_file.is_file():
            raise FileNotFoundError(
                f"Chunks file does not exist: '{self.chunks_file}'."
            )

        chunks: list[Chunk] = []
        try:
            with self.chunks_file.open(encoding="utf-8") as chunks_file:
                for line_num, json_record in enumerate(chunks_file, start=1):
                    if not json_record.strip():
                        continue
                    try:
                        chunks.append(
                            Chunk.model_validate_json(json_record)
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"Invalid chunk at line {line_num} in "
                            f"'{self.chunks_file}': {error}"
                        ) from error
        except OSError as error:
            raise OSError(
                f"Could not read chunks file '{self.chunks_file}': {error}"
            ) from error

        if not chunks:
            raise ValueError(f"Chunks file is empty: '{self.chunks_file}'.")
        return chunks

    def build(self) -> int:
        """Chunk the corpus, build BM25, and save the persisted index."""
        Chunking(
            project_root=self.project_root,
            corpus_docs_root=self.corpus_docs_root,
            corpus_code_root=self.corpus_code_root,
            output_path=self.chunks_file,
            max_chunk_size=self.max_chunk_size,
        ).run()

        chunks = self._load_chunks()
        index = self._build_bm25(chunks)
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        with self.index_file.open('wb') as index_file:
            pickle.dump(index, index_file, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            f"Ingestion completed. Indexed {len(chunks)} chunks under "
            f"'{self.index_file}'."
        )
        return len(chunks)
