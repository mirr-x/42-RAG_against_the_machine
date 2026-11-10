""" __main__ entry file """

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
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e} from {e.__traceback__.tb_frame.f_code.co_filename} at line {e.__traceback__.tb_lineno}")
