import ast
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class CodeChunk:
    name: str
    type: str  # "function", "class", "async_function", etc.
    signature: str
    body: str
    file_path: str
    start_line: int
    end_line: int
    parent_context: Optional[str] = None


class ASTCodeChunker:

    def __init__(self, file_path: str):
        self.file_path = file_path
        with open(file_path, "r", encoding="utf-8") as f:
            self.source_lines = f.readlines()
        self.source_code = "".join(self.source_lines)

    def _get_source_segment(self, start_line: int, end_line: int) -> str:
        """Extracts raw source lines given 1-indexed boundaries."""
        return "".join(self.source_lines[start_line - 1 : end_line])

    def _extract_signature(self, node: ast.AST) -> str:
        """Extracts the exact definition/signature line(s) of a function or class."""
        # Finds where the actual body block begins to isolate the signature line(s)
        first_body_node = node.body[0]
        sig_end = first_body_node.lineno - 1
        return "".join(self.source_lines[node.lineno - 1 : sig_end]).strip()

    def chunk_file(self) -> List[CodeChunk]:
        """Parses the codebase using AST and structures it into distinct chunks."""
        try:
            root = ast.parse(self.source_code)
        except SyntaxError as e:
            print(f"Skipping file {self.file_path} due to SyntaxError: {e}")
            return []

        chunks = []

        for node in root.body:
            # Handle Global Functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._create_function_chunk(node))

            # Handle Classes
            elif isinstance(node, ast.ClassDef):
                class_sig = self._extract_signature(node)
                # Option A: Chunk individual methods within the class
                method_found = False
                for sub_node in node.body:
                    if isinstance(
                        sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        method_found = True
                        chunks.append(
                            self._create_function_chunk(
                                sub_node, parent_class=node.name
                            )
                        )

                # Option B: If class has no methods or is tiny, chunk it as one unit
                if not method_found:
                    chunks.append(
                        CodeChunk(
                            name=node.name,
                            type="class",
                            signature=class_sig,
                            body=self._get_source_segment(
                                node.lineno, node.end_lineno
                            ),
                            file_path=self.file_path,
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                        )
                    )
        return chunks

    def _create_function_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_class: Optional[str] = None,
    ) -> CodeChunk:
        chunk_type = (
            "async_function"
            if isinstance(node, ast.AsyncFunctionDef)
            else "function"
        )
        return CodeChunk(
            name=node.name,
            type=chunk_type,
            signature=self._extract_signature(node),
            body=self._get_source_segment(node.lineno, node.end_lineno),
            file_path=self.file_path,
            start_line=node.lineno,
            end_line=node.end_lineno,
            parent_context=f"class {parent_class}" if parent_class else None,
        )


# ==========================================
# Example Usage Validation
# ==========================================
if __name__ == "__main__":
    # Create a dummy target python file to chunk
    dummy_code = """
import os

class AuthManager:
    \"\"\"Manages user authentication tokens.\"\"\"
    
    def __init__(self, secret: str):
        self.secret = secret

    async def verify_token(self, token: str) -> dict:
        if not token:
            raise ValueError("Empty token")
        return {"status": "authenticated"}

def global_helper():
    return os.getpid()
"""

    target_filename = "demo_auth.py"
    with open(target_filename, "w", encoding="utf-8") as f:
        f.write(dummy_code.strip())

    # Run the chunker against the generated script
    chunker = ASTCodeChunker(file_path=target_filename)
    processed_chunks = chunker.chunk_file()

    # Format output for downstream Vector DB ingestion
    print(json.dumps([asdict(c) for c in processed_chunks], indent=2))
