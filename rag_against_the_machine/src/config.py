""" Configuration for the RAG system """

from pathlib import Path

ROOT = Path.cwd()
CORPUS_DOCS_ROOT = Path('data/raw/vllm-0.10.1/docs')
CORPUS_CODE_ROOT = Path('data/raw/vllm-0.10.1/vllm')
CHUNKS_FILE = ROOT / 'data/processed/chunks.jsonl'
