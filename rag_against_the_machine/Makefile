UV := ~/.local/bin/uv
K := 10
MAX_CHUNK_SIZE := 1500
MAX_CONTEXT_LENGTH := 2000
DATASET_DIR := data/datasets_public/public
SEARCH_RESULTS_DIR := data/output/search_results/UnansweredQuestions
ANSWER_RESULTS_DIR := data/output/search_results_and_answer/UnansweredQuestions


help:
	@echo "Available commands:"
	@awk -F: '/^[[:alnum:]_-]+:/ { print "        " $$1 }' $(MAKEFILE_LIST)

uv:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

install: uv
	@UV_LINK_MODE=copy $(UV) sync

index:
	uv run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

search:
	uv run python -m src search --query "what is a multimodal model?" --k $(K)

run-search_dataset_code:
	uv run python -m src search_dataset --dataset_path $(DATASET_DIR)/UnansweredQuestions/dataset_code_public.json --k $(K) --save_directory $(SEARCH_RESULTS_DIR)

run-search_dataset_docs:
	uv run python -m src search_dataset --dataset_path $(DATASET_DIR)/UnansweredQuestions/dataset_docs_public.json --k $(K) --save_directory $(SEARCH_RESULTS_DIR)

run-answer:
	uv run python -m src answer --question "what is a multimodal model?" --k $(K)

run-answer-dataset_code:
	uv run python -m src answer_dataset --student_search_results_path $(SEARCH_RESULTS_DIR)/dataset_code_public.json --save_directory $(ANSWER_RESULTS_DIR) --k $(K)

run-answer-dataset_docs:
	uv run python -m src answer_dataset --student_search_results_path $(SEARCH_RESULTS_DIR)/dataset_docs_public.json --save_directory $(ANSWER_RESULTS_DIR) --k $(K)

run-moulinette_code:
	./moulinette-ubuntu evaluate_student_search_results \
	$(SEARCH_RESULTS_DIR)/dataset_code_public.json \
	$(DATASET_DIR)/AnsweredQuestions/dataset_code_public.json \
	--k $(K) \
	--max_context_length $(MAX_CONTEXT_LENGTH)

run-moulinette_docs:
	./moulinette-ubuntu evaluate_student_search_results \
	$(SEARCH_RESULTS_DIR)/dataset_docs_public.json \
	$(DATASET_DIR)/AnsweredQuestions/dataset_docs_public.json \
	--k $(K) \
	--max_context_length $(MAX_CONTEXT_LENGTH)

run-moulinette_answer_code:
	./moulinette-ubuntu evaluate_student_answers $(ANSWER_RESULTS_DIR)/dataset_code_public.json

run-moulinette_answer_docs:
	./moulinette-ubuntu evaluate_student_answers $(ANSWER_RESULTS_DIR)/dataset_docs_public.json

lint:
	uv run flake8 . --exclude=.venv,data,tmp,tests
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude='(^|/)(\.venv|data|tmp|tests)(/|$$)'

clean:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +
	find . -name '.mypy_cache' -exec rm -rf {} +


.PHONY: help uv install run-index run-search run-search_dataset_code run-search_dataset_docs run-moulinette_code run-moulinette_docs lint clean