"""Retrieval and search functionality for the RAG system."""

import pickle
import numpy as np
from rank_bm25 import BM25Okapi

from src.indexing.indexing import Indexer


class Retriever:
    """Retriever class for searching the index."""

    def __init__(self, index_file: str):
        self.index_file = index_file

    def query(self, query: str, k: int = 5):
        """Retrieve the top-k results for the given query."""
        # load the index from the index_file
        with open(self.index_file, 'rb') as index_file:
            bm25_file = pickle.load(index_file)
            bm25_index: BM25Okapi = bm25_file['bm25']
        # Apply the same tokenizer used by Indexer.
        tokenized_query = Indexer.tokenize(query)

        # Calculate BM25 scores.
        scores = bm25_index.get_scores(tokenized_query)
        scores = bm25_index.get_scores(tokenized_query)

        np.set_printoptions(threshold=np.inf)
        print(scores)
