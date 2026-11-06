"""  A module for chunking python code into smaller, manageable pieces. """

import ast
from pathlib import Path
from typing import Optional
from scandir_rs import Walk
from pydantic import BaseModel

class CodeChunk(BaseModel):
    """A chunk of Python code linked to its original file."""

    name: str
    type: str
    signature: str
    body: str
    file_path: str
    start_line: int
    end_line: int
    parent_context: Optional[str] = None

class ASTCodeChunker:
    def __init__(self, file_path: str, source_code: str, max_chunk_size: int) -> None:
        self.file_path = file_path
        self.source_code = source_code

    def _extract_signature(self, node: ast.AST) -> str:
        x = node.
        pass

    def _creat_function_chunck(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        parent_classe: Optional[str] = None,
        ) -> CodeChunk:
        """ Creat a code chunk """

        type = 'function' if isinstance(node, ast.FunctionDef) else 'async_function'
        return CodeChunk(
            name=node.name,
            type=type,
            signature=_extract_signature(node),
            body=_extract_body(node.lineno, (node.end_lineno or node.lineno)),
            file_path=file_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            parent_context=f'class {parent_classe}' if parent_classe else None
        )

    def _chunks_python_file(self) -> list[CodeChunk]:
        """Parses the codebase using AST and structures it into distinct chunks."""
        try:
            tree = ast.parse(self.source_code)
        except SyntaxError as e:
            print(f"Skipping file {self.file_path} due to SyntaxError: {e}")
            return []

        chunks = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._creat_function_chunck(node))
            elif isinstance(node, ast.ClassDef):
                
                pass


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
    if 1 <= max_chunk_size <= 2000:
        raise ValueError('max_chunk_size must be between 1 and 2000')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_count = 0

    walker = Walk(root_path=str(project_root / corpus_root), store=False)

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for root, _, files in walker:
            for filename in sorted(files):
                if filename.lower().endswith('.py'):
                    file_path = project_root / corpus_root / root / filename

                    chunker = ASTCodeChunker(
                        file_path=str(corpus_root / root / filename),
                        source_code=file_path.read_text(encoding='utf-8'),
                        max_chunk_size=max_chunk_size
                    )
                    chunks = chunker._chunks_python_file()
                    for chunk in chunks:
                        output_file.write(chunk.model_dump_json() + '\n')
                        chunk_count += 1

    return chunk_count
