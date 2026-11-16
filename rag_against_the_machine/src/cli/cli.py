""" CLI for the RAG system """

from pathlib import Path

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

    def search(self, query: str, k: int = 5) -> None:
        """ Search the index for the given query """
        results = Retriever(index_file=INDEX_FILE).query(query, k)
        for result in results:
            print(
                f'{result.file_path} '
                f'[{result.first_character_index}:'
                f'{result.last_character_index}]'
            )

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 5,
    ) -> None:
        """ Answer questions from a dataset and save the results to a file """
        dataset_path: Path = Path(dataset_path)
        save_directory: Path = Path(save_directory)

        print(
            'Saved student_search_results to ',
            Retriever(index_file=INDEX_FILE).query_batch(
                dataset_path=dataset_path,
                save_directory=save_directory,
                k=k
        ))
