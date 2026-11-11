""" CLI for the RAG system """

from src.chunking.chuncking import Chunking
from src.config import ROOT, CORPUS_DOCS_ROOT, CORPUS_CODE_ROOT, CHUNKS_FILE


class CLI:
    """ CLI for the RAG system """
    def index(self, max_chunk_size: int = 2000):
        """ Index the documentation and code corpus """

        chunking = Chunking(
            project_root=ROOT,
            corpus_docs_root=CORPUS_DOCS_ROOT,
            corpus_code_root=CORPUS_CODE_ROOT,
            output_path=CHUNKS_FILE,
            max_chunk_size=max_chunk_size
        )
        chunking.run()
