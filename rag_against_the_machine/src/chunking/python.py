
"""A module for chunking Python code into smaller pieces."""
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
DefinitionTypes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class ASTCodeChunker:
    """Parse a Python file and produce structured code chunks."""

    def __init__(self, file_path: Path, max_chunk_size: int = 2000) -> None:
        if not 1 <= max_chunk_size <= 2000:
            raise ValueError("max_chunk_size must be between 1 and 2000")

        self.file_path = str(file_path)
        self.max_chunk_size = max_chunk_size
        self.source_code = file_path.read_text(encoding="utf-8")

        # Keep newline characters to preserve the original source.
        self.source_lines = self.source_code.splitlines(keepends=True)

    # @ DONE
    @staticmethod
    def _definition_start_line(node: DefinitionNode) -> int:
        """Return the first decorator line, or the definition line."""
        if node.decorator_list:
            return node.decorator_list[0].lineno
        return node.lineno

    @staticmethod
    def _extend_context(parent_context: str | None, curr_context: str) -> str:
        """Append a definition to its parent context."""

        if parent_context is None:
            return curr_context
        return f"{parent_context} > {curr_context}"

    # @ DONE
    @staticmethod
    def _travers_nested_definitions(node: ast.AST) -> Iterator[DefinitionNode]:
        """ Iterate over nested definitions within a node. """
        for child in ast.walk(node):
            if child is not node and isinstance(child, DefinitionTypes):
                yield child

    # @ DONE
    def _extract_signature(self, node: DefinitionNode) -> str:
        """Extract decorators and the definition header."""
        start_line = self._definition_start_line(node)
        first_body_line = node.body[0].lineno

        # Automatically picks the correct end line using max
        end_line = max(node.lineno, first_body_line - 1)
        return self._get_source_segment(start_line, end_line).strip()

    def _get_source_segment(self, start: int, end: int) -> str:
        """Extract original source lines."""
        return "".join(self.source_lines[start - 1:end])

    def _create_chunks(
        self,
        content: str,
        start_line: int,
        metadata: CodeChunk
    ) -> list[Chunk]:
        """ Split code into chunks without exceeding the size limit """

        chunks: list[Chunk] = []
        curr_lines = []
        curr_size = 0
        curr_start = start_line
        curr_end = start_line

        def save_whatever_lines_were_gathered_before():
            nonlocal curr_lines, curr_size

            if not curr_lines:
                return
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    content=''.join(curr_lines),
                    file_path=self.file_path,
                    chunk_type='code',
                    start_line=curr_start,
                    end_line=curr_end,
                    metadata=metadata
                )
            )
            curr_lines = []
            curr_size = 0

        source_lines = content.splitlines(keepends=True)
        for line_number, line in enumerate(source_lines, start_line):
            line_size = len(line)

            # Rare case
            # A single line is larger than the complete chunk limit.
            if line_size > self.max_chunk_size:
                save_whatever_lines_were_gathered_before()
                for offset in range(0, line_size, self.max_chunk_size):
                    part = line[offset:offset + self.max_chunk_size]
                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            content=part,
                            file_path=self.file_path,
                            chunk_type='code',
                            start_line=line_number,
                            end_line=line_number,
                            metadata=metadata
                        )
                    )
                continue

            if line_size + curr_size > self.max_chunk_size:
                save_whatever_lines_were_gathered_before()
            if not curr_lines:
                curr_start = line_number

            curr_size += line_size
            curr_end = line_number
            curr_lines.append(line)

        save_whatever_lines_were_gathered_before()
        return chunks

    def _create_function_chunks(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_context: str | None = None,
    ) -> list[Chunk]:
        """Create one or more chunks for a function or method."""

        type_ = (
            "async_function"
            if isinstance(node, ast.AsyncFunctionDef)
            else "function"
        )

        metadata = CodeChunk(
            name=node.name,
            type=type_,
            signature=self._extract_signature(node),
            parent_context=parent_context,
        )

        content = self._get_source_segment(
            node.lineno,
            node.end_lineno or node.lineno,
        )

        return self._create_chunks(content, node.lineno, metadata)


    def _create_definition_chunks(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        parent_context: str | None = None,
    ) -> list[Chunk]:
        """Create chunks for a definition and nested definitions."""

        chunks: list[Chunk] = []

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.extend(
                self._create_function_chunks(node, parent_context)
            )
            child_context = self._extend_context(
                parent_context,
                f"function {node.name}",
            )
            child_definitions = (
                child for child in node.body
                if isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                    ),
                )
            )
            for child in child_definitions:
                chunks.extend(
                    self._create_definition_chunks(child, child_context)
                )
            return chunks

        direct_methods = [
            child for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        class_context = self._extend_context(
            parent_context,
            f"class {node.name}",
        )

        if not direct_methods or parent_context is not None:
            chunks.extend(
                self._create_empty_class_chunks(node, parent_context)
            )

        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.extend(
                    self._create_definition_chunks(child, class_context)
                )
            elif isinstance(child, ast.ClassDef):
                chunks.extend(
                    self._create_definition_chunks(child, class_context)
                )

        return chunks

    def _create_empty_class_chunks(
        self,
        node: ast.ClassDef,
        parent_context: str | None = None,
    ) -> list[Chunk]:
        """Create chunks for a class without methods."""

        metadata = CodeChunk(
            name=node.name,
            type="class",
            signature=self._extract_signature(node),
            parent_context=parent_context,
        )

        content = self._get_source_segment(
            node.lineno,
            node.end_lineno or node.lineno,
        )

        return self._create_chunks(
            content=content,
            start_line=node.lineno,
            metadata=metadata,
        )

    # @ DONE
    def _create_module_chunks(self, nodes: list[ast.stmt]) -> list[Chunk]:
        if not nodes:
            return []

        first_node = nodes[0]
        last_node = nodes[-1]
        metadata = CodeChunk(
            name='<module>',
            type='module',
            signature=''
        )
        content = self._get_source_segment(
                    first_node.lineno,
                    last_node.end_lineno or last_node.lineno,
                )
        return self._create_chunks(
            content=content,
            start_line=first_node.lineno,
            metadata=metadata
        )

    def chunks_python_file(self) -> list[Chunk]:
        """Parse Python file and produce structured code chunks."""
        try:
            tree = ast.parse(self.source_code)
        except SyntaxError as error:
            print(
                f"Skipping file {self.file_path} due to SyntaxError: {error}")
            return []

        chunks: list[Chunk] = []
        module_nodes: list[ast.stmt] = []

        for node in tree.body:
            if isinstance(node, DefinitionTypes):
                chunks.extend(self._create_module_chunks(module_nodes))
                module_nodes = []
                chunks.extend(self._create_definition_chunks(node))
            else:
                module_nodes.append(node)
                # Traverse nested functions and classes
                for child in self._travers_nested_definitions(node):
                    chunks.extend(
                        self._create_definition_chunks(child, '<module>'))

        chunks.extend(self._create_module_chunks(module_nodes))
        return chunks

# @ DONE
def chunk_python_corpus_with_rust(
    project_root: Path,
    corpus_root: Path,
    output_path: Path,
    max_chunk_size: int = 2000,
) -> int:
    """Chunk a corpus of Python files using Rust for traversal."""

    if not 1 <= max_chunk_size <= 2000:
        raise ValueError(
            "max_chunk_size must be between 1 and 2000"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_count = 0

    walker = Walk(
        root_path=str(project_root / corpus_root),
        store=False,
    )

    with open(output_path, "w", encoding="utf-8") as output_file:
        for root, _, files in walker:
            for filename in sorted(files):

                if not filename.lower().endswith(".py"):
                    continue

                file_path = corpus_root / root / filename

                chunker = ASTCodeChunker(
                    file_path=file_path,
                    max_chunk_size=max_chunk_size,
                )

                for chunk in chunker.chunks_python_file():
                    output_file.write(chunk.model_dump_json() + "\n")
                    chunk_count += 1

    return chunk_count
