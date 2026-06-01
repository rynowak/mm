# CI Pipeline and PR Quality Gates

## When to Use

No CI pipeline configured yet. When added, it will run on GitHub Actions triggered by PRs to `main`.

## Prerequisites

- Python 3.12+
- uv

## Run Locally Before Pushing

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Gate Breakdown

1. **lint** -- `uv run ruff check .` enforces lint rules.
2. **format-check** -- `uv run ruff format --check .` verifies formatting without writes.
3. **test** -- `uv run pytest` runs the test suite.

## Run Individual Gates

```bash
uv run ruff check .           # lint
uv run ruff format --check .  # format check
uv run ruff format .          # format (auto-fix)
uv run pytest                 # tests
uv run pytest path/to/test.py # single test file
```

## Known Flaky Tests

None known.

## Common Failure Patterns

| Error pattern | Cause | Fix |
|---------------|-------|-----|
| `ruff check` failures | Lint violations | `uv run ruff check --fix .` |
| `ruff format` failures | Formatting drift | `uv run ruff format .` |
| Import errors in tests | Missing dependency | `uv add <package>` |
