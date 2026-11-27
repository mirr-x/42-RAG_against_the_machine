""" CLI for the RAG system """

from pathlib import Path
from time import perf_counter
from typing import List, Literal

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
        results: List[MinimalSource] = retriver.query(
            query,
            k,
            MinimalSource
        )
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
        chunk_type: Literal["code", "markdown"] | None = None,
    ) -> None:
        """ Answer questions from a dataset and save the results to a file """
        dataset_path_obj: Path = Path(dataset_path)
        save_directory_obj: Path = Path(save_directory)

        print(
            'Saved student_search_results to ',
            Retriever(index_file=INDEX_FILE).query_batch(
                dataset_path=dataset_path_obj,
                save_directory=save_directory_obj,
                k=k,
                chunk_type=chunk_type,
            ))

    def answer(
        self,
        question: str,
        k: int = 5,
        context: Literal["code", "markdown"] | None = None
    ) -> None:
        """ Answer a question using the RAG system """
        start_time = perf_counter()
        sources: List[Chunk] = Retriever(index_file=INDEX_FILE).query(
            question,
            k,
            Chunk,
            chunk_type=context
        )
        for source in sources:
            print(
                f'{source.file_path} '
                f'[{source.first_character_index}:'
                f'{source.last_character_index}]\n'
            )
        qwen_llm = LLM(model_name="Qwen/Qwen3-0.6B")
        generation = Generation(llm=qwen_llm)
        answer = generation.generate(
            question=question,
            retrieved_sources=sources
        )
        print(answer.answer)
        print(f"Time: {perf_counter() - start_time:.2f} seconds")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        k: int | None = None,
    ) -> None:
        """Generate answers for student search results and save them."""
        start_time = perf_counter()
        dataset_path: Path = Path(student_search_results_path)
        save_directory_path: Path = Path(save_directory)

        retriever = Retriever(index_file=INDEX_FILE)
        qwen_llm = LLM(model_name="Qwen/Qwen3-0.6B")
        generation = Generation(llm=qwen_llm)
        print(
            'Saved student_search_results_and_answer to ',
            generation.generate_batch(
                dataset_path=dataset_path,
                save_directory=save_directory_path,
                retriever=retriever,
                k=k
            )
        )
        print(f"Time: {perf_counter() - start_time:.2f} seconds")
