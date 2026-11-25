""" Generation module for the RAG system. """

import sys
import uuid
from typing import Any
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
    StudentSearchResults,
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
        question_id: str | None = None,
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
            question_id=(
                question_id if question_id is not None else str(uuid.uuid4())
            ),
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
        k: int | None = None,
    ) -> Path:
        """Answer a student search-results file and save structured output."""
        if not dataset_path.exists():
            raise FileNotFoundError("Dataset file not found.")
        if k is not None and k <= 0:
            raise ValueError("k must be a positive integer.")

        try:
            save_directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ValueError(
                f"Error creating saved directory: {error}"
            ) from error

        # Load retrieval results
        try:
            with dataset_path.open('r', encoding='utf-8') as file:
                search_results = StudentSearchResults.model_validate(
                    json.load(file)
                )
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValidationError,
        ) as error:
            raise ValueError(
                f"Error loading or validating dataset file: {error}"
            ) from error

        if search_results.k <= 0:
            raise ValueError(
                "Search results must declare a positive k value."
            )
        output_k = search_results.k
        if k is not None:
            output_k = min(k, search_results.k)  # delete me using agent

        results: list[MinimalAnswer] = []
        for question in tqdm(
            search_results.search_results,
            desc="Generating answers",
            unit=" question",
        ):
            source_references = question.retrieved_sources[:output_k]
            retrieved_sources = retriever.resolve_sources(source_references)
            results.append(self.generate(
                question=question.question,
                retrieved_sources=retrieved_sources,
                question_id=question.question_id,
            ))
        try:
            output = StudentSearchResultsAndAnswer(
                search_results=results,
                k=output_k,
            )
            output_file_path = save_directory / dataset_path.name
            with output_file_path.open('w', encoding='utf-8') as f:
                json.dump(output.model_dump(), f, ensure_ascii=False, indent=4)
        except (OSError, TypeError, ValueError) as e:
            raise ValueError(f"Error saving generated answers: {e}") from e
        return output_file_path
