"""Build and persist a BM25 index for the chunk corpus."""

import re
import pickle

from rank_bm25 import BM25Okapi
from pathlib import Path
from tqdm import tqdm

from src.chunking.chuncking import Chunking
from src.models import Chunk, CodeChunk, MarkdownChunk

TOKEN_PATTERN = re.compile(r'[A-Za-z0-9_]+')
IDENTIFIER_PART_PATTERN = re.compile(
    r'[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+'
)

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
    def tokenize(text: str, split_camel_case: bool = False) -> list[str]:
        """Tokenize content by keep related words and discard useless words"""
        tokens: list[str] = []

        for original_word in TOKEN_PATTERN.findall(text):
            word = original_word.lower()
            if word not in USELESS_WORDS:
                tokens.append(word)

            if '_' in word:
                for part in word.split("_"):
                    if part and part not in USELESS_WORDS:
                        tokens.append(part)

            if split_camel_case and any(
                char.isupper() for char in original_word
            ):
                for part in IDENTIFIER_PART_PATTERN.findall(original_word):
                    normalized_part = part.lower()
                    if (
                        normalized_part != word
                        and normalized_part not in USELESS_WORDS
                    ):
                        tokens.append(normalized_part)

        return tokens

    @staticmethod
    def build_search_document(chunk: Chunk) -> str:
        """Include searchable metadata alongside each chunk's source text."""
        metadata = chunk.metadata
        if isinstance(metadata, CodeChunk):
            filename = Path(chunk.file_path).stem
            fields = [
                filename,
                filename.replace("_", "") if "_" in filename else "",
                metadata.name,
                metadata.signature,
                metadata.parent_context or "",
                chunk.content,
            ]
        elif isinstance(metadata, MarkdownChunk):
            fields = [
                f"heading level {metadata.level}",
                metadata.title,
                chunk.content,
            ]
        else:
            raise TypeError(
                f"Unsupported chunk metadata type: {type(metadata).__name__}."
            )
        return " ".join(field for field in fields if field)

    def _build_bm25(self, chunks: list[Chunk]) -> dict[str, object]:
        """Build typed BM25 indexes and a unified fallback index."""
        chunks_by_type: dict[str, list[Chunk]] = {
            "code": [],
            "markdown": [],
        }
        for chunk in chunks:
            if isinstance(chunk.metadata, CodeChunk):
                chunks_by_type["code"].append(chunk)
            elif isinstance(chunk.metadata, MarkdownChunk):
                chunks_by_type["markdown"].append(chunk)
            else:
                raise TypeError(
                    "Unsupported chunk metadata type: "
                    f"{type(chunk.metadata).__name__}."
                )

        bm25_by_type: dict[str, BM25Okapi] = {}
        for chunk_type, typed_chunks in chunks_by_type.items():
            if not typed_chunks:
                continue
            tokenized_documents = [
                self.tokenize(
                    self.build_search_document(chunk),
                    split_camel_case=chunk_type == "code",
                )
                for chunk in tqdm(
                    typed_chunks,
                    desc=f"Tokenizing {chunk_type} chunks",
                    unit=" chunk",
                    ascii="-#",
                )
            ]
            bm25_by_type[chunk_type] = BM25Okapi(
                tokenized_documents,
                b=0.3,
                epsilon=0.5,
            )

        unified_chunks = chunks
        unified_documents = [
            self.tokenize(
                self.build_search_document(chunk),
                split_camel_case=True,
            )
            for chunk in tqdm(
                unified_chunks,
                desc="Tokenizing unified chunks",
                unit=" chunk",
                ascii="-#",
            )
        ]
        bm25_all = BM25Okapi(
            unified_documents,
            b=0.3,
            epsilon=0.5,
        )

        return {
            "version": 3,
            "algorithm": "rank-bm25",
            "bm25_by_type": bm25_by_type,
            "chunks_by_type": {
                chunk_type: typed_chunks
                for chunk_type, typed_chunks in chunks_by_type.items()
                if typed_chunks
            },
            "bm25_all": bm25_all,
            "chunks_all": unified_chunks,
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
