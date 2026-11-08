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

    def _extract_signature(self, node: DefinitionNode) -> str:
        """Extract decorators and the definition header."""
        start_line = self._definition_start_line(node)
        first_body_line = node.body[0].lineno

        # For a one-line definition, the body starts on the header line.
        # Keep the complete line as the best available representation.
        if first_body_line <= node.lineno:
            return self._get_source_segment(
                start_line,
                node.end_lineno or node.lineno,
            ).strip()

        return self._get_source_segment(
            start_line,
            first_body_line - 1,
        ).strip()

    def _create_chunks( #? DONE
        self,
        content: str,
        start_line: int,
        metadata: CodeChunk,
    ) -> list[Chunk]:
        """Split content at line boundaries, enforcing max_chunk_size."""
        chunks: list[Chunk] = []
        current_lines: list[str] = []
        current_size = 0
        current_start = start_line
        current_end = start_line

        def save_chunk() -> None:
            nonlocal current_lines, current_size

            if not current_lines:
                return
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    content="".join(current_lines),
                    file_path=self.file_path,
                    chunk_type="code",
                    start_line=current_start,
                    end_line=current_end,
                    metadata=metadata,
                )
            )
            current_lines = []
            current_size = 0

        for line_number, line in enumerate(
            content.splitlines(keepends=True),
            start=start_line,
        ):
            line_size = len(line)

            # A single line is larger than the complete chunk limit.
            if line_size > self.max_chunk_size:
                save_chunk()

                for offset in range(0, line_size, self.max_chunk_size):
                    part = line[offset:offset + self.max_chunk_size]
                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            content=part,
                            file_path=self.file_path,
                            chunk_type="code",
                            start_line=line_number,
                            end_line=line_number,
                            metadata=metadata,
                        )
                    )
                continue

            # The next complete line will not fit in the current chunk.
            if current_size + line_size > self.max_chunk_size:
                save_chunk()

            # The first line added after a save starts a new chunk.
            if not current_lines:
                current_start = line_number

            current_lines.append(line)
            current_size += line_size
            current_end = line_number

        save_chunk()
        return chunks

    def _create_function_chunks(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_context: str | None = None,
    ) -> list[Chunk]:
        """Create one or more chunks for a function or method."""
        function_type = (
            "async_function"
            if isinstance(node, ast.AsyncFunctionDef)
            else "function"
        )

        metadata = CodeChunk(
            name=node.name,
            type=function_type,
            signature=self._extract_signature(node),
            parent_context=parent_context,
        )

        start_line = self._definition_start_line(node)
        content = self._get_source_segment(
            start_line,
            node.end_lineno or node.lineno,
        )

        return self._create_chunks(content, start_line, metadata)


    @staticmethod
    def _iter_nested_definitions(node: ast.AST) -> Iterator[DefinitionNode]:
        """Yield nested definitions using a simple flat loop."""
        definition_types = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        )

        # ast.walk visits every single node underneath without needin recursion
        for child in ast.walk(node):
            if child is not node and isinstance(child, definition_types):
                yield child

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

