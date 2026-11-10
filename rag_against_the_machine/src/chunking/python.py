
"""A module for chunking Python code into smaller pieces."""
# pylint: disable=no-name-in-module
# pylint: disable=too-few-public-methods
import ast
import uuid
from pathlib import Path
from typing import Iterator

from scandir_rs import Walk

from src.chunking.models import Chunk, CodeChunk


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
        self.line_offsets = []
        character_offset = 0
        for line in self.source_lines:
            self.line_offsets.append(character_offset)
            character_offset += len(line)

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

    @staticmethod
    def _travers_nested_definitions(node: ast.AST) -> Iterator[DefinitionNode]:
        """ Iterate over nested definitions within a node. """
        for child in ast.walk(node):
            if child is not node and isinstance(child, DefinitionTypes):
                yield child

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
        content_start = self.line_offsets[start_line - 1]
        curr_char_start = content_start
        curr_char_end = content_start - 1

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
                    first_character_index=curr_char_start,
                    last_character_index=curr_char_end,
                    metadata=metadata
                )
            )
            curr_lines.clear()
            curr_size = 0

        source_lines = content.splitlines(keepends=True)
        character_offset = content_start
        for line_number, line in enumerate(source_lines, start_line):
            line_size = len(line)

            # Rare case
            # A single line is larger than the complete chunk limit.
            if line_size > self.max_chunk_size:
                save_whatever_lines_were_gathered_before()
                for offset in range(0, line_size, self.max_chunk_size):
                    part = line[offset:offset + self.max_chunk_size]
                    char_start = character_offset + offset
                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            content=part,
                            file_path=self.file_path,
                            chunk_type='code',
                            start_line=line_number,
                            end_line=line_number,
                            first_character_index=char_start,
                            last_character_index=char_start + len(part) - 1,
                            metadata=metadata
                        )
                    )
                character_offset += line_size
                continue

            if line_size + curr_size > self.max_chunk_size:
                save_whatever_lines_were_gathered_before()
            if not curr_lines:
                curr_start = line_number
                curr_char_start = character_offset

            curr_size += line_size
            curr_end = line_number
            curr_char_end = character_offset + line_size - 1
            curr_lines.append(line)
            character_offset += line_size

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
        start_line = self._definition_start_line(node)
        end_line = node.end_lineno or node.lineno
        content = self._get_source_segment(start_line, end_line)
        return self._create_chunks(content, start_line, metadata)

    def _create_class_chunks(
        self,
        node: ast.ClassDef,
        parent_context: str | None = None,
    ) -> list[Chunk]:
        """Create class chunks, grouping class-level statements together."""
        metadata = CodeChunk(
            name=node.name,
            type="class",
            signature=self._extract_signature(node),
            parent_context=parent_context,
        )

        chunks: list[Chunk] = []
        class_start = self._definition_start_line(node)
        first_body_line = node.body[0].lineno
        header_end = first_body_line - 1
        class_nodes: list[ast.stmt] = []

        def flush_class_nodes():
            nonlocal class_nodes
            if class_nodes:
                start = class_nodes[0].lineno
                end = class_nodes[-1].end_lineno or class_nodes[-1].lineno
                content = self._get_source_segment(start, end)
                chunks.extend(self._create_chunks(content, start, metadata))
                class_nodes.clear()

        # 1. Extract decorators and class header as the first chunk
        if header_end >= class_start:
            header = self._get_source_segment(class_start, header_end)
            chunks.extend(self._create_chunks(header, class_start, metadata))

        # 2. Ignore non-definition class body statements (docstrings, fields)
        for child in node.body:
            if any(isinstance(descendant, DefinitionTypes) for descendant in ast.walk(child)):
                flush_class_nodes()  # Flush pending attributes/docstrings before a method
                continue
            class_nodes.append(child)

        flush_class_nodes()

        # case for single-line classes
        if not chunks:
            content = self._get_source_segment(
                class_start,
                node.end_lineno or node.lineno,
            )
            chunks.extend(self._create_chunks(content, class_start, metadata))

        return chunks

    def _generate_chunks_for_node(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        parent_context: str | None = None,
    ) -> list[Chunk]:
        """Create chunks for a definition and nested definitions."""
        chunks: list[Chunk] = []
        current_context = parent_context

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.extend(self._create_function_chunks(node, parent_context))
            current_context = self._extend_context(
                parent_context,
                f"function {node.name}",
            )
        elif isinstance(node, ast.ClassDef):
            chunks.extend(self._create_class_chunks(node, parent_context))
            current_context = self._extend_context(
                parent_context, f"class {node.name}")

        for child in self._travers_nested_definitions(node):
            chunks.extend(
                self._generate_chunks_for_node(child, current_context)
            )

        return chunks

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
            print(f"Skipping file {self.file_path} due to SyntaxError: {error}")
            return []

        chunks: list[Chunk] = []
        module_nodes: list[ast.stmt] = []

        def flush_module_nodes():
            nonlocal module_nodes
            if module_nodes:
                chunks.extend(self._create_module_chunks(module_nodes))
                module_nodes.clear()

        for node in tree.body:
            if isinstance(node, DefinitionTypes):
                flush_module_nodes()
                chunks.extend(self._generate_chunks_for_node(node))
            else:
                module_nodes.append(node)
                
        flush_module_nodes()
        
        return chunks


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

    walk_root = project_root / corpus_root
    try:
        walker = Walk(str(walk_root), store=False)
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Walk "
            f"for corpus root '{walk_root}': "
            f"{type(e).__name__}: {e}"
        ) from e

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
