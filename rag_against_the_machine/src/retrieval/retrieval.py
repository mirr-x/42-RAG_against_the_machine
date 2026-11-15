"""Retrieval and search functionality for the RAG system."""

from pathlib import Path
import pickle
import numpy as np
from rank_bm25 import BM25Okapi

from src.indexing.indexing import Indexer


class Retriever:
    """Retriever class for searching the index."""

    def __init__(self, index_file: Path) -> None:
        self.index_file = index_file

    def query(self, query: str, k: int = 5) -> None:
        """Retrieve the top-k results for the given query."""
        if not self.index_file.exists():
            raise FileNotFoundError("Index file not found.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if not query.strip():
            raise ValueError("Query string cannot be empty.")

        # load the index from the index_file
        with open(self.index_file, 'rb') as index_file:
            try:
                bm25_file = pickle.load(index_file)
            except Exception as e:
                raise ValueError(f"Error loading index file: {e}") from e

        bm25_index: BM25Okapi = bm25_file['bm25']
        if k > len(bm25_index.doc_freqs):
            raise ValueError("k cannot be greater than the number of chunks.")

        # Apply the same tokenizer used by Indexer.
        tokenized_query = Indexer.tokenize(query)
        if not tokenized_query:
            raise ValueError("Query string must contain at least one valid token after tokenization.")

        # Calculate BM25 scores.
        scores = bm25_index.get_scores(tokenized_query)

        # Select the top k chunks and select the k
        top_k_unsorted = np.argpartition(scores, -k)[-k:]
        top_k_sorted = sorted(
            top_k_unsorted,
            key=lambda i: scores[i],
            reverse=True,
        )
        # TODO: Return the top k results as models instead of printing them.
        print(scores)
        print('*' * 9)
        print(scores[top_k_sorted])
        print(top_k_sorted)
