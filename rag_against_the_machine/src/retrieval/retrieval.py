"""Retrieval and search functionality for the RAG system."""


class Retriever:
    """Retriever class for searching the index."""

    def __init__(self, index_file: str):
        self.index_file = index_file
