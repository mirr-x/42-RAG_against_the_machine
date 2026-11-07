"""Tests for Python source chunking."""

from pathlib import Path

from src.chunking.python import ASTCodeChunker


def test_module_level_statements_are_chunked(tmp_path: Path) -> None:
    source = (
        '"""Module docs."""\n'
        'import os\n'
        'DATABASE_URL = "sqlite://"\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    print(DATABASE_URL)\n'
    )
    source_path = tmp_path / "example.py"
    source_path.write_text(source, encoding="utf-8")

    chunks = ASTCodeChunker(source_path).chunks_python_file()

    assert len(chunks) == 1
    assert chunks[0].Metadata.name == "<module>"
    assert chunks[0].Metadata.type == "module"
    assert chunks[0].content == source
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 6


def test_module_statements_are_kept_around_definitions(tmp_path: Path) -> None:
    source = (
        "import os\n"
        "\n"
        "def get_path():\n"
        "    return os.getcwd()\n"
        "\n"
        "VALUE = 42\n"
    )
    source_path = tmp_path / "example.py"
    source_path.write_text(source, encoding="utf-8")

    chunks = ASTCodeChunker(source_path).chunks_python_file()

    assert [chunk.Metadata.type for chunk in chunks] == [
        "module",
        "function",
        "module",
    ]
    assert chunks[0].content == "import os\n"
    assert chunks[1].content == "def get_path():\n    return os.getcwd()\n"
    assert chunks[2].content == "VALUE = 42\n"
