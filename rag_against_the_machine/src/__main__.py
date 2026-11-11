""" __main__ entry file """

import traceback
from pathlib import Path

from src.chunking.chuncking import Chunking

ROOT = Path.cwd()
CORPUS_DOCS_ROOT = Path('data/raw/vllm-0.10.1/docs')
CORPUS_CODE_ROOT = Path('data/raw/vllm-0.10.1/vllm')
CHUNKS_FILE = ROOT / 'data/raw/processed/chunks.jsonl'


if __name__ == '__main__':
    try:
        chunking = Chunking(
            project_root=ROOT,
            corpus_docs_root=CORPUS_DOCS_ROOT,
            corpus_code_root=CORPUS_CODE_ROOT,
            output_path=CHUNKS_FILE,
            max_chunk_size=1000
        )
        chunking.run()
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Chunking failed: {type(e).__name__}: {e}")
        traceback.print_exc()
