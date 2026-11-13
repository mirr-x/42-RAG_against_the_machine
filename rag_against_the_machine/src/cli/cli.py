""" CLI for the RAG system """

from src.config import (
    CHUNKS_FILE,
    CORPUS_CODE_ROOT,
    CORPUS_DOCS_ROOT,
    INDEX_FILE,
    ROOT,
)
from src.indexing.indexing import Indexer
from src.retrieval.retrieval import Retriever


class CLI:
    """ CLI for the RAG system """
    def index(self, max_chunk_size: int = 2000) -> None:
        """ Index the documentation and code corpus """

        Indexer(
            project_root=ROOT,
            corpus_docs_root=CORPUS_DOCS_ROOT,
            corpus_code_root=CORPUS_CODE_ROOT,
            chunks_file=CHUNKS_FILE,
            index_file=INDEX_FILE,
            max_chunk_size=max_chunk_size
        ).build()

    def search(self, query: str, k: int = 5):
        """ Search the index for the given query """
        pass  # TODO: Implement search functionality