
uv:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

install: uv
	@~/.local/bin/uv sync
	

clean:
	find . -name '__pycache__' -exec rm -rf {} +


.PHONY: uv, install, clean