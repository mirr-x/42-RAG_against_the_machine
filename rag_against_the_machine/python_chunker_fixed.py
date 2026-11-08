"""AST-based Python source chunking."""
# pylint: disable=no-name-in-module
# pylint: disable=too-few-public-methods
import ast
import uuid
from pathlib import Path
from typing import Iterator

from scandir_rs import Walk

from src.chunking.chuncking import Chunk
from src.chunking.models import CodeChunk


DefinitionNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


class ASTCodeChunker:
    """Parse a Python file and produce structured code chunks."""

    def __init__(
        self,
        file_path: Path,
        max_chunk_size: int = 2000,
        display_path: str | None = None,
    ) -> None:
        if not 1 <= max_chunk_size <= 2000:
            raise ValueError("max_chunk_size must be between 1 and 2000")

        self.source_path = Path(file_path)
        self.file_path = display_path or str(file_path)
        self.max_chunk_size = max_chunk_size
        self.source_code = self.source_path.read_text(encoding="utf-8")
        self.source_lines = self.source_code.splitlines(keepends=True)


    def _create_class_chunks(
        self,
        node: ast.ClassDef,
        parent_context: str | None = None,
    ) -> list[Chunk]:
        """Create class-context chunks without copying method implementations."""
        metadata = CodeChunk(
            name=node.name,
            type="class",
            signature=self._extract_signature(node),
            parent_context=parent_context,
        )

        chunks: list[Chunk] = []
        class_start = self._definition_start_line(node)
        first_body_line = node.body[0].lineno

        # Keep decorators and the class header in a separate class chunk.
        header_end = first_body_line - 1
        if header_end >= class_start:
            header = self._get_source_segment(class_start, header_end)
            chunks.extend(self._create_chunks(header, class_start, metadata))

        # Keep docstrings, attributes, assignments, and other class-level
        # statements. Methods and nested classes get their own chunks below.
        class_level_nodes: list[ast.stmt] = []

        for child in node.body:
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue

            # Do not copy a control-flow statement that contains a nested
            # definition into the class chunk. The nested definition gets
            # its own chunk during recursive traversal.
            if next(self._iter_nested_definitions(child), None) is not None:
                continue

            class_level_nodes.append(child)

        for child in class_level_nodes:
            child_start = child.lineno
            child_end = child.end_lineno or child.lineno
            content = self._get_source_segment(child_start, child_end)
            chunks.extend(
                self._create_chunks(content, child_start, metadata)
            )

        # A one-line class can have its header and body on the same line.
        # If no class content was captured, preserve the complete definition.
        if not chunks:
            content = self._get_source_segment(
                class_start,
                node.end_lineno or node.lineno,
            )
            chunks.extend(
                self._create_chunks(content, class_start, metadata)
            )

        return chunks

    def _create_definition_chunks(
        self,
        node: DefinitionNode,
        parent_context: str | None = None,
    ) -> list[Chunk]:
        """Create chunks for a definition and all nested definitions."""
        chunks: list[Chunk] = []

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.extend(self._create_function_chunks(node, parent_context))
            current_context = self._extend_context(parent_context,f"function {node.name}",)
        else:
            chunks.extend(self._create_class_chunks(node, parent_context))
            current_context = self._extend_context(parent_context, f"class {node.name}",)

        # Find definitions directly inside this definition and definitions
        # nested inside if/for/try/with and similar statements.
        for child in self._iter_nested_definitions(node):
            chunks.extend(
                self._create_definition_chunks(child, current_context)
            )

        return chunks

