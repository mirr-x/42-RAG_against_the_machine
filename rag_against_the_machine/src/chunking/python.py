
"""A module for chunking Python code into smaller pieces."""
# pylint: disable=no-name-in-module
# pylint: disable=too-few-public-methods
import ast
import uuid
from pathlib import Path

from scandir_rs import Walk

from src.chunking.chuncking import Chunk
from src.chunking.models import CodeChunk


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

    def _extract_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> str:
        """Extract the signature of a function or class."""

        sig = self.source_lines[
            node.lineno - 1: node.body[0].lineno - 1
        ]
        return "".join(sig).strip()

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
            c_line = len(line)
            # Rare case
            if c_line > self.max_chunk_size:
                save_whatever_lines_were_gathered_before()
                for offset in range(0, c_line, self.max_chunk_size):
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

            if c_line + curr_size > self.max_chunk_size:
                save_whatever_lines_were_gathered_before()

            curr_size += c_line
            curr_end = line_number
            curr_lines.append(line)

        save_whatever_lines_were_gathered_before()

        return chunks

    def _create_function_chunks(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_class: str | None = None,
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
            parent_context=(
                f"class {parent_class}" if parent_class else None
            ),
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

    def _create_empty_class_chunks(
        self,
        node: ast.ClassDef,
    ) -> list[Chunk]:
        """Create chunks for a class without methods."""

        metadata = CodeChunk(
            name=node.name,
            type="class",
            signature=self._extract_signature(node),
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



    def chunks_python_file(self) -> list[Chunk]:
        """Parse Python code into AST-based chunks."""

        try:
            tree = ast.parse(self.source_code)
        except SyntaxError as error:
            print(
                f"Skipping file {self.file_path} "
                f"due to SyntaxError: {error}"
            )
            return []

        chunks: list[Chunk] = []

        function_types = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        )
        module_nodes: list[ast.stmt] = []

        for node in tree.body:

            if isinstance(node, function_types):
                chunks.extend(self._create_module_chunks(module_nodes))
                module_nodes = []
                chunks.extend(self._create_function_chunks(node))

            elif isinstance(node, ast.ClassDef):
                chunks.extend(self._create_module_chunks(module_nodes))
                module_nodes = []
                method_found = False

                for sub_node in node.body:
                    if isinstance(sub_node, function_types):
                        method_found = True

                        chunks.extend(
                            self._create_function_chunks(
                                sub_node,
                                parent_class=node.name,
                            )
                        )

                if not method_found:
                    chunks.extend(
                        self._create_empty_class_chunks(node)
                    )
            else:
                module_nodes.append(node)

        chunks.extend(self._create_module_chunks(module_nodes))

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

                chunks = chunker.chunks_python_file()

                for chunk in chunks:
                    output_file.write(chunk.model_dump_json() + "\n")
                    chunk_count += 1

    return chunk_count
