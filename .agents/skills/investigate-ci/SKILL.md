---
name: ci-cd-diagnosis[github-actions]
description: >
  Diagnose CI/CD pipeline failures on GitHub Actions. Use when CI checks fail,
  build errors occur, tests are failing, or when asked to investigate why builds fail.
  Triggers on: 'CI failing', 'build broken', 'pipeline failed', 'diagnose CI',
  'why is CI red'.
file_dependencies:
  - path: docs/ci-context.md
    description: "Pipeline structure, required checks, known flaky tests, common failure patterns"
    template: templates/ci-context.md.tmpl
---

# CI/CD Diagnosis — GitHub Actions

Systematic approach to diagnosing CI/CD pipeline failures on GitHub Actions.

## Prerequisites

- `gh` CLI authenticated
- Access to the repository's Actions tab

## When to Use

- PR CI checks are failing
- User asks "why is CI failing?" or "diagnose CI"
- Need to determine if failure is code-related or infrastructure-related
- Want to check if the default branch is also broken

## Repo-Specific Context

Read `docs/ci-context.md` for pipeline structure, known flaky tests, and common failure patterns. Also check `AGENTS.md` CI/CD section for gate commands and required checks.

## Diagnosis Workflow

### Step 1: Identify the Failing Run

```bash
# Get failed check runs for the current PR/branch
gh run list --branch $(git branch --show-current) --status failure --limit 5
```

If a PR number is known:
```bash
gh pr checks <number>
```

### Step 2: Fetch Failure Logs

```bash
gh run view <run-id> --log-failed
```

If the output is too large, target a specific job:
```bash
gh run view <run-id> --job <job-id> --log-failed
```

### Step 3: Classify the Failure

| Category | Indicators | Action |
|----------|-----------|--------|
| **Code error** | Compiler/lint/test failure referencing changed files | Fix the code |
| **Infrastructure** | Timeout, runner error, network failure, rate limit | Re-run: `gh run rerun <run-id> --failed` |
| **Flaky test** | Test passes locally, intermittent history | Re-run, or mark as known-flaky per repo conventions |
| **Dependency** | Package resolution failure, version conflict | Check lockfile, update deps |
| **Configuration** | Workflow syntax error, missing secret/variable | Fix workflow file |
| **Base branch broken** | Same failure on default branch | Not your problem — check if there's a revert in progress |

### Step 4: Check if Base Branch is Broken

```bash
# Check if main/default branch has the same failure
gh run list --branch main --status failure --limit 3
```

If main is also broken, inform the user — their PR may not be the cause.

### Step 5: Determine if Failure is Related to Changes

```bash
# Get the files changed in this PR/branch
git diff --name-only origin/main...HEAD
```

Cross-reference changed files with the failure message. If the failure references files not in the diff, it's likely a pre-existing issue or infrastructure problem.

### Step 6: Re-run if Infrastructure

For infrastructure/flaky failures:
```bash
gh run rerun <run-id> --failed
```

Monitor the re-run:
```bash
gh run watch <run-id>
```

### Step 7: Local Reproduction

If the failure appears code-related, attempt local reproduction using the commands from `AGENTS.md` or the workflow file:

```bash
# Read the failing step's command from the workflow
cat .github/workflows/<workflow>.yml
```

Run the equivalent command locally to get better error output.

## Output

Present findings as:

```
## CI Diagnosis

**Run**: <run-id> (<workflow name>)
**Status**: <failure category>
**Job**: <failing job name>

### Root Cause
<one sentence explanation>

### Evidence
<relevant log lines>

### Recommendation
<what to do — fix code, re-run, wait for main fix, etc.>
```

## Common Patterns

| Pattern | Diagnosis |
|---------|-----------|
| `Error: Process completed with exit code 1` with no other context | Check the step's `run:` command — often a script that swallows errors |
| Timeout after 6h | Runner hung — re-run |
| `Error: No space left on device` | Runner disk full — re-run or reduce artifacts |
| `Error: The operation was canceled` | Workflow was superseded by a newer push — expected |
| Secret/token errors | Secret expired or not available to forks |
| `annotations` showing lint/type errors | Code issue — fix locally |
