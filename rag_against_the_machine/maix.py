from src.chunking.python import chunk_python_corpus_with_rust
from pathlib import Path

def main() -> None:
    """Main function to chunk a Python corpus using Rust."""
    ROOT = Path.cwd()
    CORPUS_ROOT = Path('./tests')
    CHUNKS_FILE = ROOT / 'data/raw/processed/c.jsonl'

    try:
        count = chunk_python_corpus_with_rust(
            project_root=ROOT,
            corpus_root=CORPUS_ROOT,
            output_path=CHUNKS_FILE,
            max_chunk_size=1000
        )
        print(f"Chunks written: {count}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()