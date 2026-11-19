"""Retrieval and search functionality for the RAG system."""

from pathlib import Path
from typing import List

import json
import pickle
import numpy as np
from pydantic import ValidationError
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.models import (
    Chunk,
    MinimalSource,
    StudentSearchResults,
    RagDataset,
    MinimalSearchResults
)
from src.indexing.indexing import Indexer


class Retriever:
    """Retriever class for searching the index."""

    def __init__(self, index_file: Path) -> None:
        self.index_file = index_file
        if not self.index_file.exists():
            raise FileNotFoundError(
                f"Index file not found: '{self.index_file}'. "
                "Build it first with `python -m src index`."
            )

    def query(
            self,
            query: str,
            k: int = 5,
            data_type: MinimalSource | Chunk = MinimalSource,
            ) -> List[MinimalSource | Chunk]:
        """Retrieve the top-k results for the given query."""
        if not self.index_file.exists():
            raise FileNotFoundError(
                f"Index file not found: '{self.index_file}'. "
                "Build it first with `python -m src index`."
            )
            
        if (
            not query
            or not isinstance(query, str)
            or query.strip() == ""
            or len(query) > 1000
            or len(query) < 5
        ):
            raise ValueError(
                "query cannot be empty and must be between 5 and "
                "1000 characters."
            )
        if not isinstance(k, int) or k <= 0 or k > 25:
            raise ValueError("k must be a positive integer.")

        # load the index from the index_file
        with open(self.index_file, 'rb') as index_file:
            try:
                bm25_file = pickle.load(index_file)
            except (
                EOFError,
                OSError,
                pickle.PickleError,
                AttributeError,
                TypeError,
            ) as error:
                raise ValueError(
                    f"Error loading index file: {error}"
                ) from error

        bm25_index: BM25Okapi = bm25_file['bm25']
        if k > len(bm25_index.doc_freqs):
            raise ValueError("k cannot be greater than the number of chunks.")

        # Apply the same tokenizer used by Indexer.
        tokenized_query = Indexer.tokenize(query)
        if not tokenized_query:
            raise ValueError(
                "Query string must contain at least one valid token "
                "after tokenization."
            )

        # Calculate BM25 scores.
        scores = bm25_index.get_scores(tokenized_query)

        # Select the top k chunks and select the k
        top_k_unsorted = np.argpartition(scores, -k)[-k:]
        top_k_sorted = sorted(
            top_k_unsorted,
            key=lambda i: scores[i],
            reverse=True,
        )

        # Return the top k results as models instead of printing them.
        top_k_results = []
        for i in top_k_sorted:
            chunk = bm25_file['chunks'][i]
            if chunk.last_character_index < chunk.first_character_index:
                raise ValueError(
                    "Chunk end index cannot precede its start index.")
            if (
                chunk.last_character_index - chunk.first_character_index + 1
                > 2000
            ):
                raise ValueError("Chunk width cannot exceed 2,000 characters.")

            if data_type == Chunk:
                top_k_results.append(Chunk.model_validate(chunk))
            elif data_type == MinimalSource:
                top_k_results.append(MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index
                ))
        return top_k_results

    def query_batch(
        self,
        dataset_path: Path,
        save_directory: Path,
        k: int = 5
    ) -> Path:
        """Retrieve the top-k results for each question in a dataset."""
        if not dataset_path.exists():
            raise FileNotFoundError("Dataset file not found.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        try:
            save_directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ValueError(
                f"Error creating saved directory: {error}"
            ) from error

        # Load the dataset
        try:
            with dataset_path.open('r', encoding='utf-8') as file:
                rag_dataset = RagDataset.model_validate(json.load(file))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValidationError,
        ) as error:
            raise ValueError(
                f"Error loading or validating dataset file: {error}"
            ) from error

        results: List[MinimalSearchResults] = []
        for question in tqdm(
            rag_dataset.rag_questions,
            desc="Retrieving dataset questions",
            unit=" question",
        ):
            results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=self.query(question.question, k)
            ))
        try:
            search_result = StudentSearchResults(search_results=results, k=k)
            result_path = save_directory / dataset_path.name
            with result_path.open('w', encoding='utf-8') as f:
                json.dump(search_result.model_dump(), f, indent=4)
        except (OSError, TypeError, ValueError) as e:
            raise ValueError(f"Error saving search results: {e}") from e
        return result_path

    