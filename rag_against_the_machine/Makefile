uv:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

install: uv
	@UV_LINK_MODE=copy ~/.local/bin/uv sync

run-index:
	uv run python -m src index --max_chunk_size 1500

run-search:
	uv run python -m src search --query "what is a multimodal model?" --k 10

run-search_dataset_code:
	uv run python -m src search_dataset --dataset_path data/datasets_public/public/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

run-search_dataset_docs:
	uv run python -m src search_dataset --dataset_path data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

run-moulinette_code:
	./moulinette-ubuntu evaluate_student_search_results \
	data/output/search_results/UnansweredQuestions/dataset_code_public.json \
	data/datasets_public/public/AnsweredQuestions/dataset_code_public.json \
	--k 10 \
	--max_context_length 2000

run-moulinette_docs:
	./moulinette-ubuntu evaluate_student_search_results \
	data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
	data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
	--k 10 \
	--max_context_length 2000

lint:
	uv run flake8 . --exclude=.venv,data,tmp,tests
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude='(^|/)(\.venv|data|tmp|tests)(/|$$)'

clean:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +


.PHONY: uv install run-index run-search run-search_dataset_code run-search_dataset_docs run-moulinette_code run-moulinette_docs lint clean