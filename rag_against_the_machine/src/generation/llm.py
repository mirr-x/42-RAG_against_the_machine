"""LLM module for the RAG system."""

from typing import Any

from transformers import A

SYSTEM_PROMPT: str = (
    "You are a precise vLLM codebase assistant. Answer the user's question "
    "using ONLY the provided sources.\n"
    "Rules:\n"
    "1. Rely strictly on the facts in the sources. Do not guess or use "
    "outside knowledge.\n"
    "2. If the sources do not contain the answer, reply ONLY with: "
    '"I cannot find the answer in the provided sources."\n'
    "3. Cite the snippet number in brackets immediately after every fact "
    "(e.g., [1], [2]).\n"
    "4. Keep the answer concise, direct, and factual."
)


class LLM:
    """Build prompts for the configured language model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForCausalLm.from_pretrained(self.model_name)

    def build_qwen_prompt(
        self, question: str, retrieved_sources: list[dict[str, Any]]
    ) -> str:
        """Build a Qwen model sys prompt based on the question and sources.

        Args:
            @param question: The user's question.
            @param retrieved_sources: A list of dictionaries containing snippet
                information.
        """
        if not retrieved_sources:
            context_text = "No retrieved_sources provided."
        else:
            formatted_chunks: list[str] = []
            for idx, item in enumerate(retrieved_sources, start=1):
                file_path = str(item.get("file_path", "unknown"))
                text = str(item.get("text", "")).strip()
                formatted_chunks.append(f"[{idx}] ({file_path})\n{text}")
            context_text = "\n\n---\n\n".join(formatted_chunks)

        return (
            f"### Context\n{context_text}\n\n"
            f"### Question\n{question}\n/no_think\n\n"
            "### Answer\n"
        )