
uv:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

install: uv
	@~/.local/bin/uv sync

run-index:
	uv run python -m src index --max_chunk_size 2000

lint:
	uv run flake8 . --exclude=.venv,data,tmp,tests
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude='(^|/)(\.venv|data|tmp|tests)(/|$$)'


clean:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +


.PHONY: uv install run-index lint clean