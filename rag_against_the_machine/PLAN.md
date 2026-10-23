# RAG Against the Machine - Project Phases

## Project Phases

- [-] 0**Phase 0 - Understand the project:** Understand RAG, its components, and the complete pipeline.
- [-] **Phase 1 - Set up the project:** Configure Python, `uv`, dependencies, linting, typing, and project structure.
- [ ] **Phase 2 - Explore and ingest the codebase:** Read the vLLM repository files and preserve their exact paths.
- [ ] **Phase 3 - Build the chunking system:** Split Python code and documentation into chunks with exact character spans.
- [ ] **Phase 4 - Build the index:** Create and persist a BM25 or TF-IDF index.
- [ ] **Phase 5 - Implement retrieval:** Rank chunks and return the most relevant Top-K results.
- [ ] **Phase 6 - Implement data models and JSON:** Validate data with Pydantic and handle JSON inputs and outputs.
- [ ] **Phase 7 - Connect Qwen and generate answers:** Generate evidence-based answers using retrieved context.
- [ ] **Phase 8 - Build the required CLI:** Implement `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, and `evaluate`.
- [ ] **Phase 9 - Evaluate and improve Recall@K:** Measure retrieval quality and optimize performance against the required thresholds.
- [ ] **Phase 10 - Test, document, and submit:** Test edge cases, complete the README and Makefile, and verify the full pipeline.

