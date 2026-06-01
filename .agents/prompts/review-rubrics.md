# Review Rubrics

## Hard Rules

### Always Check

- Type hints on all function signatures.
- Training scripts set random seeds (torch, numpy, python random).
- No hardcoded file paths — use `pathlib.Path` and relative paths or CLI args.
- Hyperparameters logged or printed at the start of training.
- Dependencies added via `uv add`, not pip.

### Never Allow

- Committing model checkpoints, datasets, or large binary files.
- `pip install` in any script or documentation.
- Secrets or API keys in code or config files.
- Training code that silently swallows exceptions during data loading.

## Repo-Specific Patterns

- Each exercise/sample is a self-contained directory with its own README.
- Shared utilities go in a common library package, not copy-pasted between exercises.
- RL environments should follow the Gymnasium API when applicable.

## Severity Calibration

| Severity | Criteria | Action |
|----------|----------|--------|
| **Blocking** | Bugs, security issues, data loss, breaks CI | Must fix before merge |
| **High** | Missing error handling, race conditions, perf regression | Request changes |
| **Medium** | Naming, missing tests for edge cases, minor refactor opportunity | Approve with comment |
| **Nit** | Style preference within existing conventions | Comment only if pattern |

## What NOT to Flag

- Formatting issues (ruff handles this).
- Import ordering (ruff handles this).
- Choice of optimizer or hyperparameters in training scripts (that's the experiment).
- Minor docstring style differences.
