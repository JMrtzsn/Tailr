.PHONY: install lint fix format typecheck test build all venv

install:
	uv sync --all-extras

lint:
	uv run ruff check src/

fix:
	uv run ruff check --fix src/
	uv run ruff format src/

format:
	uv run ruff format --check src/

typecheck:
	uv run mypy src/

test:
	uv run pytest --tb=short -q

build:
	uv build

venv:
	rm -rf .venv
	uv venv
	uv sync --all-extras

all: lint format typecheck test build

