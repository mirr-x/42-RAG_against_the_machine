""" Generation module for the RAG system. """

from typing import Any

from src.models import StudentSearchResultsAndAnswer, MinimalAnswer, Chunk
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

    def generate(
        self,
        question: str,
        retrieved_sources: list[Chunk],
    ) -> StudentSearchResultsAndAnswer:
        """Generate an answer and return it with the original source metadata."""
        contexts = self.source_context_loader(retrieved_sources)
        prompt = self.llm.build_qwen_prompt(question, contexts)
        generated: Any = self.llm.pipeline(prompt, return_full_text=False)
        if (
            not isinstance(generated, list)
            or not generated
            or not isinstance(generated[0], dict)
            or not isinstance(generated[0].get("generated_text"), str)
        ):
            raise RuntimeError("The language model returned an invalid response.")

        answer = MinimalAnswer(
            answer=generated[0]["generated_text"],
        )
        return StudentSearchResultsAndAnswer(
            search_results=[answer],
            k=len(retrieved_sources),
        )
