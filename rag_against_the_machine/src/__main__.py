""" __main__ entry file """

from pathlib import Path

from src.chunking.markdown import chunk_markdown_corpus_with_rust

ROOT = Path.cwd()
CORPUS_ROOT = Path('data/raw/vllm-0.10.1/docs')
CHUNKS_FILE = ROOT / 'data/raw/processed/chunks.jsonl'


if __name__ == '__main__':
    try:
        count = chunk_markdown_corpus_with_rust(
            project_root=ROOT,
            corpus_root=CORPUS_ROOT,
            output_path=CHUNKS_FILE,
            max_chunk_size=1000
        )
        print(f"Chunks written: {count}")
    except ValueError as e:
        print(f"Error: {e}")
