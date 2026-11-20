""" Generation module for the RAG system. """

import sys
from typing import Any
import uuid

from src.models import (
    Chunk,
    CodeChunk,
    MarkdownChunk,
    MinimalAnswer,
    MinimalSource,
    StudentSearchResultsAndAnswer,
)
from src.generation.llm import LLM


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
    def format_chunks_for_llm(chunks: list[Chunk]) -> str:
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
            block = f"<document index=\"{i}\" {header_info}>\n{chunk.content}\n</document>"
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def generate(
        self,
        question: str,
        retrieved_sources: list[Chunk],
    ) -> StudentSearchResultsAndAnswer:
        """Generate a structured, source-linked answer for one question."""
        contexts: str = self.format_chunks_for_llm(retrieved_sources)
        prompt: str = self.llm.build_qwen_prompt(question, contexts)
        print(
            "Generating a concise answer on CPU; this may take a few minutes...",
            file=sys.stderr,
            flush=True,
        )
        generated: Any = self.llm.pipeline(
            prompt,
            return_full_text=False,
            max_new_tokens=64,
            do_sample=False,
        )
        if (
            not isinstance(generated, list)
            or not generated
            or not isinstance(generated[0], dict)
            or not isinstance(generated[0].get("generated_text"), str)
        ):
            raise RuntimeError("The language model returned an invalid response.")

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
        # return StudentSearchResultsAndAnswer(
        #     search_results=[answer],
        #     k=len(retrieved_sources),
        # )
