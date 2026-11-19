""" CLI for the RAG system """

from pathlib import Path
from typing import List

from src.config import (
    CHUNKS_FILE,
    CORPUS_CODE_ROOT,
    CORPUS_DOCS_ROOT,
    INDEX_FILE,
    ROOT,
)
from src.generation.generation import Generation
from src.retrieval.retrieval import Retriever
from src.indexing.indexing import Indexer
from src.models import Chunk, MinimalSource
from src.generation.llm import LLM


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
        retriver = Retriever(index_file=INDEX_FILE)
        results: List[MinimalSource] = retriver.query(query, k, MinimalSource)
        for result in results:
            print(
                f'{result.file_path} '
                f'[{result.first_character_index}:'
                f'{result.last_character_index}]'
            )

    def search_dataset(
        self,
        dataset_path_: str,
        save_directory_: str,
        k: int = 5,
    ) -> None:
        """ Answer questions from a dataset and save the results to a file """
        dataset_path: Path = Path(dataset_path_)
        save_directory: Path = Path(save_directory_)

        print(
            'Saved student_search_results to ',
            Retriever(index_file=INDEX_FILE).query_batch(
                dataset_path=dataset_path,
                save_directory=save_directory,
                k=k
            ))

    def answer(self, question: str, k: int = 5) -> None:
        """ Answer a question using the RAG system """
        # TODO: Step 5: Add the answer CLI command
        results: List[Chunk] = Retriever(index_file=INDEX_FILE).query(question, k, Chunk)
        print(results)
        for result in results:
            print(
                f'{result.file_path} '
                f'[{result.first_character_index}:'
                f'{result.last_character_index}]'
            )
        # qwen_llm = LLM(model_name="Qwen/Qwen3-0.6B")
        # generation = Generation(llm=qwen_llm)
        # print(generation.generate(question=question, retrieved_sources=results))
        pass
