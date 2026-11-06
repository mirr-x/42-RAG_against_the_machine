"""  A module for chunking python code into smaller, manageable pieces. """
# pylint: disable=no-name-in-module
# pylint: disable=too-few-public-methods
from typing import Optional

import ast
import uuid
from pathlib import Path
from scandir_rs import Walk

from src.chunking.chuncking import Chunk
from src.chunking.models import CodeChunk


class ASTCodeChunker:
    """Parse a Python file and produce structured code chunks."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = str(file_path)
        self.source_code = file_path.read_text(encoding='utf-8')
        self.source_lines = self.source_code.splitlines()

    def _extract_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> str:
        """Extracts the exact signature line(s) of a func or class."""
        sig = self.source_lines[node.lineno - 1: node.body[0].lineno - 1]
        return ''.join(sig).strip()

    def _get_source_segment(self, start: int, end: int) -> str:
        """ Extracts raw source lines """
        body = self.source_lines[start - 1: end]
        return ''.join(body)

    def _creat_function_chunck(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_classe: Optional[str] = None,
    ) -> Chunk:
        """ Creat a CodeChunk """

        type_ = (
            'function' if isinstance(node, ast.FunctionDef)
            else 'async_function'
        )
        code_chunk = CodeChunk(
            name=node.name,
            type=type_,
            signature=self._extract_signature(node),
            parent_context=f'class {parent_classe}' if parent_classe else None
        )
        return Chunk(
            id=str(uuid.uuid4()),
            content=self._get_source_segment(
                node.lineno, (node.end_lineno or node.lineno)),
            file_path=self.file_path,
            chunk_type='code',
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            Metadata=code_chunk
        )

    def _create_empty_class_chunk(self, node: ast.ClassDef) -> Chunk:
        """Create a chunk for a class that does not contain methods."""
        code_chunk = CodeChunk(
            name=node.name,
            type='class',
            signature=self._extract_signature(node)
        )
        return Chunk(
            id=str(uuid.uuid4()),
            content=self._get_source_segment(
                node.lineno, (node.end_lineno or node.lineno)),
            file_path=self.file_path,
            chunk_type='code',
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            Metadata=code_chunk
        )

    def chunks_python_file(self) -> list[Chunk]:
        """Parses the codebase using AST into distinct chunks."""
        try:
            tree = ast.parse(self.source_code)
        except SyntaxError as e:
            print(f"Skipping file {self.file_path} due to SyntaxError: {e}")
            return []

        chunks = []
        func_or_async_func = (ast.FunctionDef, ast.AsyncFunctionDef)
        for node in tree.body:
            if isinstance(node, func_or_async_func):
                chunks.append(self._creat_function_chunck(node))
            elif isinstance(node, ast.ClassDef):
                class_sig = self._extract_signature(node)
                method_found = False

                for sub_node in node.body:
                    if isinstance(sub_node, func_or_async_func):
                        method_found = True
                        chunks.append(
                            self._creat_function_chunck(sub_node, class_sig))
                if not method_found:
                    chunks.append(self._create_empty_class_chunk(node))
        return chunks


def chunk_python_corpus_with_rust(
    project_root: Path,
    corpus_root: Path,
    output_path: Path,
    max_chunk_size: int = 2000
) -> int:
    """
    Chunk a corpus of Python files using Rust for performance.

    Args:
        project_root (Path): The root directory of the project.
        corpus_root (Path): The directory containing the Python files to chunk.
        output_path (Path): The path where the chunked output will be saved.
        max_chunk_size (int, optional): The max number of characters per chunk.

    Returns:
        int: The number of chunks created.
    """
    if not 1 <= max_chunk_size <= 2000:
        raise ValueError('max_chunk_size must be between 1 and 2000')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_count = 0

    walker = Walk(root_path=str(project_root / corpus_root), store=False)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for root, _, files in walker:
            for filename in sorted(files):
                if filename.lower().endswith('.py'):

                    chunker = ASTCodeChunker(
                        file_path=(corpus_root / root / filename),
                    )
                    chunks = chunker.chunks_python_file()
                    for chunk in chunks:
                        output_file.write(chunk.model_dump_json() + '\n')
                        chunk_count += 1

    return chunk_count
