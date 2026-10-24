.PHONY: format lint type check test build

format:
	uv run ruff format src tests scripts tools
	uv run ruff check --fix src tests scripts tools

lint:
	uv run ruff format --check src tests scripts tools
	uv run ruff check src tests scripts tools

type:
	uv run ty check
	uv run pyright

check: lint type

test:
	CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q

build:
	uv build
