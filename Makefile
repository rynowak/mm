.PHONY: check lint format typecheck test verify-cluster

check: lint format typecheck test

# Gate cluster-bound code against the Ray nodes' Python 3.9 before submitting a job.
# Statically catches the silent version breaks (datetime.UTC=3.11, zip(strict=)=3.10,
# int.bit_count()=3.10, ...) that otherwise only surface as a failed remote run.
# Scope = exactly what ships to the cluster: bufo/ + the lib it imports (mm-training),
# tests excluded (they run locally on 3.12).
verify-cluster:
	uv run vermin -t=3.9- --no-tips --violations --eval-annotations \
		--exclude-regex '.*/tests/.*' bufo libs/mm-training/src

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
