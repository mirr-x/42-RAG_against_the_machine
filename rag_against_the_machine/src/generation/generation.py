""" Generation module for the RAG system. """

import sys
import uuid
from typing import Any, List
from pathlib import Path


import json
from tqdm import tqdm
from pydantic import ValidationError

from src.models import (
    Chunk,
    CodeChunk,
    MarkdownChunk,
    MinimalAnswer,
    MinimalSource,
    RagDataset,
    StudentSearchResultsAndAnswer,
)
from src.generation.llm import LLM
from src.retrieval.retrieval import Retriever


class Generation:
    """Class for generating responses in the RAG system."""

    def __init__(self, llm: LLM) -> None:
        """
        Initialize the Generation class with a model.

        Args:
            llm: The LLM used for generating responses.
        """
        self.llm = llm

    @staticmethod
    def format_chunks_for_llm_as_xml(chunks: list[Chunk]) -> str:
        """Format retrieved chunks as XML documents for the language model.

        Args:
            chunks: The chunks to serialize, including their metadata.

        Returns:
            A string containing one XML document block per chunk.
        """
        formatted_blocks = []

        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            # Build specific attributes based on the metadata type
            if isinstance(meta, CodeChunk):
                header_info = (
                    f'type="code" file="{chunk.file_path}" '
                    f'symbol="{meta.name}" kind="{meta.type}" '
                    f'parent="{meta.parent_context or "None"}" '
                    f'lines="{meta.start_line}-{meta.end_line}"'
                )
            elif isinstance(meta, MarkdownChunk):
                header_info = (
                    f'type="markdown" file="{chunk.file_path}" '
                    f'section="{meta.title}" '
                    f'lines="{meta.start_line}-{meta.end_line}"'
                )
            else:
                header_info = f'file="{chunk.file_path}"'

            # Wrap in XML tags
            block = (
                f"<document index=\"{i}\" {header_info}>\n"
                f"{chunk.content}\n</document>"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def generate(
        self,
        question: str,
        retrieved_sources: list[Chunk],
    ) -> MinimalAnswer:
        """Generate a structured, source-linked answer for one question."""
        contexts: str = self.format_chunks_for_llm_as_xml(retrieved_sources)
        prompt: str = self.llm.build_qwen_prompt(question, contexts)
        print(
            "Generating the answer on CPU; this may take a few minutes...",
            file=sys.stderr,
            flush=True,
        )
        generated: Any = self.llm.pipeline(
            prompt,
            return_full_text=False,
            max_new_tokens=128,
            do_sample=False,
        )
        if (
            not isinstance(generated, list)
            or not generated
            or not isinstance(generated[0], dict)
            or not isinstance(generated[0].get("generated_text"), str)
        ):
            raise RuntimeError(
                "The language model returned an invalid response."
            )

        return MinimalAnswer(
            question_id=str(uuid.uuid4()),
            question=question,
            retrieved_sources=[
                MinimalSource(
                    file_path=source.file_path,
                    first_character_index=source.first_character_index,
                    last_character_index=source.last_character_index,
                )
                for source in retrieved_sources
            ],
            answer=generated[0]["generated_text"],
        )

    def generate_batch(
        self,
        dataset_path: Path,
        save_directory: Path,
        retriever: Retriever,
        k: int = 5,
    ) -> Path:
        """Generate answers for each question in a dataset and save  results"""
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

        # load the dataset
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

        results: list[MinimalAnswer] = []
        for question in tqdm(
            rag_dataset.rag_questions,
            desc="Generating answers",
            unit=" question",
        ):
            retrieved_sources: List[Chunk] = retriever.query(
                        question.question,
                        k,
                        Chunk   # type: ignore[arg-type]
                    )
            results.append(self.generate(
                question=question.question,
                retrieved_sources=retrieved_sources
            ))
        try:
            output = StudentSearchResultsAndAnswer(
                search_results=results,
                k=k
            )
            output_file_path = save_directory / dataset_path.name
            with output_file_path.open('w', encoding='utf-8') as f:
                json.dump(output.model_dump(), f, ensure_ascii=False, indent=4)
        except (OSError, TypeError, ValueError) as e:
            raise ValueError(f"Error saving generated answers: {e}") from e
        return output_file_path
