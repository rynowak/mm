.PHONY: check lint format typecheck test

check: lint format typecheck test

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run ty check

test:
	uv run pytest

# Fix variants
fix:
	uv run ruff check --fix .
	uv run ruff format .
