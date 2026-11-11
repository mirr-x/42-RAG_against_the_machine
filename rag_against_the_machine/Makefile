
uv:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

install: uv
	@~/.local/bin/uv sync

run-index:
	uv run python -m src index --max_chunk_size 2000
	

clean:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +


.PHONY: uv, install, clean