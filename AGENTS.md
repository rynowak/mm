# AGENTS.md -- mm

ML training playground for pre-training and reinforcement learning. Sample/reference implementations for learning — e.g. training a model to play Wordle. Python + PyTorch stack, managed with uv. Hosted on github.com (rynowak/mm). No CI yet — details will go in docs/ci-context.md when added.

## Working Style

- **Operate autonomously.** Prefer making reasonable choices over stopping to ask.
- **Fix ALL failures, not just the first.** Tests, lint, type check — fix everything. Iterate until green.
- **Stay focused** on the current component. Reference other components for context only.
- **Cross-component work**: only when explicitly asked.

## Operating Guidelines

| Situation | Approach |
| --------- | -------- |
| Normal | Answer first, explain after |
| Uncertain | Confidence, best guess, what resolves it |
| Stuck | State it, what tried, what needed |
| Disagree | Directly with alternatives |

**Avoid:** apologetic hedging, asking permission for trivial actions, silent failures.

**Autonomy boundaries:** Autonomy applies to local, reversible actions (reading code, running tests, editing files, pushing to your own branch). These actions **always require explicit user instruction**:

- Merging PRs
- Force-pushing or deleting shared branches
- Deploying, releasing, or promoting builds
- Any action that could reach users or break the team's workflow

**Prohibitions:** never commit secrets, bypass tests, force push, silent failures.

**Refactoring bias:** deletion and compaction worth 2x additions (if no data loss). Prefer removing code over adding.

### Source Control

- **Host:** github.com (public GitHub)
- **Remote:** `git@personal.github.com:rynowak/mm.git` (SSH alias)
- **Default branch:** `main`
- **Issues:** GitHub Issues
- **PRs:** squash merge preferred
- **CLI:** `gh` authenticated for github.com (rynowak account)

### Package Management

- **Python:** managed with `uv` (uv sync, uv run, uv add)
- **NOT allowed:** `pip install` directly, `poetry`, `conda`
- **Lock file:** `uv.lock` — keep in sync with `uv lock`
- **Install:** `uv sync --frozen`

### Skill Invocation

- **Proactive routing.** Invoke skills on matching intent — the user doesn't have to type the slash command.

**Research first:** When debugging or implementing features that touch external dependencies, research their documentation BEFORE forming hypotheses.

**Execution:** Run independent ops in parallel; sequential only when output depends on prior.

## Repo Invariants

- All Python code uses type hints.
- Format with `ruff format`, lint with `ruff check`.
- Training scripts must be reproducible: seed all RNGs, log hyperparameters.
- Each sample/exercise lives in its own top-level directory with its own README.
- Extract reusable pieces into library packages — building the abstractions is part of learning.

## Prerequisites

| Need | Why |
|------|-----|
| Python 3.12+ | Runtime |
| uv | Package management |
| PyTorch | ML framework (installed via uv) |
| gh CLI | PRs and issues |

Bootstrap: `uv sync`

## Quality Gates

All code quality checks are run through the Makefile. This is the single way to work with the code.

```bash
make check    # Run all gates: lint, format, typecheck, test
make lint     # ruff check
make format   # ruff format --check
make typecheck # ty check
make test     # pytest
make fix      # Auto-fix lint + format issues
```

**Before committing:** `make check` must pass. No exceptions.

## Testing Guardrails

- Tests required for utility/library code. Training scripts tested via running them.
- Use `pytest` with `tmp_path` for file I/O — no hardcoded paths.
- Training tests should use tiny models/datasets to run fast.
- Set seeds for reproducibility in tests that involve randomness.

## Patterns

| Pattern | Doc |
|---------|-----|
| Review rubrics | `.agents/prompts/review-rubrics.md` |
| CI pipeline | `docs/ci-context.md` |

## Keeping this file small

This file is loaded into every AI agent's context on every task. Every line is a tax.

- Invariants live inline (this file). Patterns live in `docs/*.md`.
- Component-specific context lives in `<component>/AGENTS.md`.
- No enumeration of components, design docs, or subsystem files.
- Reference material (lookup tables, env lists, pipeline details) belongs in `docs/*.md`.
- **Size budget:** root <= 200 lines (warn at 180).
