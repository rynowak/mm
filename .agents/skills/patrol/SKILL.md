---
name: patrol
description: |
  Autonomous PR patrol loop: discovers open PRs, delegates code review to
  `/pr-reviewer` (lane-based architecture with generalist + specialist lanes,
  false-positive challenger, and inline comment posting), runs scoped quality
  checks, and maintains ignored local handoff notes for the next iteration.
  Designed for continuous autonomous review via /loop.
  Use when user asks to "patrol PRs", "review open PRs", "continuous review",
  "PR loop", "review cycle", or wants autonomous PR monitoring.
argument-hint: "[-path:<glob>[;<glob>...]]"
---

# PR Patrol

You are the **PR Patrol**, an autonomous continuous review agent that discovers open PRs, delegates code review to `/pr-reviewer`, runs scoped quality checks, and maintains handoff notes for the next iteration.

**Runs AUTONOMOUSLY through ALL phases without stopping for user input.** Only stop if genuinely blocked.

## When to Use

- "patrol PRs", "review open PRs", "check PRs"
- "continuous review", "PR loop", "review cycle"
- Automated/headless sessions for continuous PR monitoring

## Arguments

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `-path:<glob>[;<glob>...]` | all PRs | Only review PRs whose changed files match at least one glob. Semicolon-separated. |

Parse from the ARGUMENTS string. Examples:
- `/patrol` — review all open PRs
- `/patrol -path:services/agents/*` — only PRs touching the agents service
- `/patrol -path:services/agents/*;docs/*` — PRs touching agents service or docs
- `/patrol -path:libs/monet-*/*` — PRs touching any monet lib

## Pre-flight Checks

**MANDATORY: Run these before doing ANY work. Abort with a clear error if any fail.**

1. **Verify branch context**: `git branch --show-current` — note the current branch
2. **Initialize patrol state and this run's scratch directory**:
   ```bash
   RUN_ID=$(uv run python -c "from datetime import UTC, datetime; print(datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'))")
   export PATROL_RUN_DIR=".patrol/tmp/$RUN_ID"
   uv run python -c "from pathlib import Path; Path(r'$PATROL_RUN_DIR').mkdir(parents=True, exist_ok=True)"
   ```
3. **Clean legacy patrol artifacts only**: if root-level stale artifacts from old patrol runs exist and are **untracked**, handle only this exact allowlist: `SHARED_TASK_NOTES.md`, `temp_design.md`, `patrol-prs.json`, `diff_output.txt`, `pr*.diff`, and `review-payload*.json`. Move `SHARED_TASK_NOTES.md` into `.patrol/SHARED_TASK_NOTES.md` if it contains watch-list state worth preserving. Move other allowlisted artifacts into `.patrol/legacy/$RUN_ID/` for preservation. Do **not** delete or move tracked, staged, non-allowlisted, or unknown user files.
   ```bash
   uv run python -c "
   import shutil
   import subprocess
   from pathlib import Path

   legacy_dir = Path('.patrol/legacy') / '$RUN_ID'
   candidates = [
       Path('SHARED_TASK_NOTES.md'),
       Path('temp_design.md'),
       Path('patrol-prs.json'),
       Path('diff_output.txt'),
       *Path('.').glob('pr*.diff'),
       *Path('.').glob('review-payload*.json'),
   ]
   for path in sorted(set(candidates), key=str):
       if not path.is_file():
           continue
       status = subprocess.run(
           ['git', 'status', '--short', '--', str(path)],
           capture_output=True,
           text=True,
           check=False,
       ).stdout
       if not status.startswith('?? '):
           continue

       dest = legacy_dir / path.name
       if path.name == 'SHARED_TASK_NOTES.md':
           notes = path.read_text(encoding='utf-8', errors='replace')
           shared_notes = Path('.patrol/SHARED_TASK_NOTES.md')
           if '### Watch List' in notes and not shared_notes.exists():
               dest = shared_notes

       dest.parent.mkdir(parents=True, exist_ok=True)
       shutil.move(str(path), str(dest))
   "
   ```
4. **Verify git state**: `git status --short` — ensure clean working tree. Ignored `.patrol/` state must not appear here.
5. **Fetch latest refs**: `git fetch origin main` — ensures diff baseline is current for `git diff origin/main...origin/<branch>` comparisons
6. **Verify tooling**:
   ```bash
   which uv
   uv run ruff --version
   ```
7. **Read `AGENTS.md`** (root and any component-level) to refresh on project conventions
8. **Log pre-flight results** to `.patrol/SHARED_TASK_NOTES.md` under `## Pre-flight [timestamp]`

---

## Patrol State and Scratch Space

Patrol owns the ignored `.patrol/` directory. Keep every persistent handoff note and every temporary review artifact there so scheduled runs can preserve local memory without dirtying the repository.

| Path | Purpose | Cleanup |
| ---- | ------- | ------- |
| `.patrol/SHARED_TASK_NOTES.md` | Persistent local watch list and handoff notes for the next patrol iteration | Keep between runs |
| `.patrol/tmp/<run-id>/` | Per-run review bodies, findings JSON, copied PR docs, diffs, and payload scratch files | Delete before a successful run exits; preserve failed runs for debugging |
| `.patrol/legacy/<run-id>/` | Preserved root-level artifacts from patrol runs before `.patrol/` existed | Keep until manually pruned |

Do not write patrol scratch files in the repo root. The final step of every successful patrol run must remove `.patrol/tmp/<run-id>/` and verify `git status --short` is clean. If the run failed, set `PATROL_RUN_FAILED=1` before Step 5 so `.patrol/tmp/<run-id>/` is preserved for post-hoc debugging.

---

## Step 1: Discover Open PRs

### 1a. Collect PR candidates from three sources

**Source A — broad discovery:**
```bash
gh pr list --state open --json number,title,headRefName,headRefOid,updatedAt,author,reviewDecision,statusCheckRollup,files --limit 50
```

**Source B — watch list re-discovery:** Check `.patrol/SHARED_TASK_NOTES.md` for the `### Watch List` section (see Step 4). For every PR number listed there, fetch it individually even if it wasn't in Source A's results:
```bash
gh pr view <NUMBER> --json number,title,headRefName,updatedAt,author,reviewDecision,statusCheckRollup,files,headRefOid
```
Skip silently if the PR is already closed/merged. This ensures previously-reviewed PRs with outstanding findings are always re-checked regardless of the `--limit` window.

**Source C — review requests for the current user:** Include PRs where the authenticated GitHub user is explicitly requested for review, even if they fall outside broad discovery or were previously approved by patrol:
```bash
gh pr list --state open --search "user-review-requested:@me" --json number,title,headRefName,headRefOid,updatedAt,author,reviewDecision,statusCheckRollup,files --limit 100
```

For GitHub hosts that support `review-requested:@me`, also include that query so team-based review requests for teams the current user belongs to are included:
```bash
gh pr list --state open --search "review-requested:@me" --json number,title,headRefName,headRefOid,updatedAt,author,reviewDecision,statusCheckRollup,files --limit 100
```

If either review-request search is unsupported by the host, log the failure in `.patrol/SHARED_TASK_NOTES.md` and continue with the other discovery sources.

**Merge** the three sources (deduplicate by PR number). Track whether each PR came from Source C; the filter step treats active review requests as an explicit request to review.

### 1b. Filter to PRs needing review

For each candidate PR, decide whether to review it using **commit-aware filtering**:

1. Check `.patrol/SHARED_TASK_NOTES.md` for the `### Watch List` section. Each entry records the PR number and the **head commit SHA** that was last reviewed (e.g., `| #989 | abc1234 | COMMENT | ... |`).
2. Get each PR's current head commit: use the `headRefOid` field from `gh pr list`/`gh pr view`, or run `gh pr view <NUMBER> --json headRefOid --jq .headRefOid`.
3. **Review the PR if ANY of these are true:**
   - The PR is not in the watch list (never reviewed before → first review)
   - The PR's current `headRefOid` differs from the SHA recorded in the watch list (new commits since last review → re-review)
   - The PR came from Source C (the current user has an active direct or team review request)
   - The PR's `reviewDecision` is `CHANGES_REQUESTED` and patrol was the reviewer (author may have addressed feedback without pushing new commits — check comments)
4. **Skip the PR if ALL of these are true:**
   - The PR is in the watch list with the same `headRefOid` (no new commits)
   - AND the PR did not come from Source C (no active review request for the current user)
   - AND the watch list decision was `APPROVE` (nothing to follow up on)
5. If timestamp formats or SHA comparisons fail, fail safe by reviewing the PR.

This replaces the old `updatedAt`-only filter. Commit SHAs are authoritative — they detect exactly when new code lands, not just metadata updates.

### 1c. Path Filtering

If `-path` was specified, filter the PR list to only include PRs that have at least one changed file matching any of the provided globs. For each PR:

1. Get its changed files from the `files` field (or via `gh pr view <number> --json files --jq '.files[].path'`)
2. Check each file against the glob patterns (semicolon-separated). Use shell-style glob matching — `*` matches within a directory, `**` matches across directories
3. **Skip** the PR entirely if none of its changed files match any glob

Example: `-path:services/agents/*;docs/*` keeps only PRs where at least one changed file starts with `services/agents/` or `docs/`.

If `-path` was NOT specified, review all open PRs (default behavior).

Log which PRs were included/excluded and why in `.patrol/SHARED_TASK_NOTES.md`.

---

## Step 2: Review Each PR via `/pr-reviewer`

For each PR that passed filtering, delegate the code review to the **pr-reviewer** skill. Invoke it as a sub-agent (using the Agent tool). Run sub-agents for independent PRs in parallel.

### Sub-Agent Brief

> You are reviewing PR #NNN in repo `bic/monet`.
>
> Invoke the `/pr-reviewer` skill to perform the review. The skill handles:
> - Lane dispatch (generalist + specialist lanes in parallel)
> - False-positive challenger
> - Convergence and deduplication
> - Posting exactly one GitHub review with inline comments
>
> Set these environment variables for the skill:
> ```
> PR_NUMBER=NNN
> GITHUB_OWNER=bic
> GITHUB_REPO=monet
> ```
>
> After the skill completes, report back:
> - **Verdict**: APPROVE / REQUEST_CHANGES
> - **Finding counts**: blocking, high, medium, nit
> - **Summary**: one-line description of the review outcome

The pr-reviewer skill handles finding verification (via its challenger) and review posting internally — patrol does not need to verify findings or post reviews separately.

---

## Step 3: Run Scoped Quality Checks

Determine which packages are affected and run only relevant checks:

```bash
# Identify affected packages (cross-platform — no awk dependency)
uv run python -c "
import subprocess, sys
out = subprocess.check_output(['git', 'diff', 'origin/main...origin/<branch>', '--name-only'], text=True)
pkgs = sorted({'/'.join(p.split('/')[:2]) for p in out.splitlines() if p.startswith(('services/', 'libs/'))})
print('\n'.join(pkgs))
"
```

For each affected package, run the quality gates from `AGENTS.md` → PR Quality Checklist:

```bash
make lint
make format-check
make typecheck
make test <package-name>
```

If protos were touched:
```bash
make protos && git diff --exit-code -- libs/monet-protos/src/monet_protos
```

---

## Step 4: Update `.patrol/SHARED_TASK_NOTES.md`

Update handoff notes for the next iteration:

```markdown
## Patrol [YYYY-MM-DD HH:MM]

### PRs Reviewed
| PR | Branch | Decision | Findings (B/H/M/N) | Summary |
|----|--------|----------|---------------------|---------|
| #NNN | branch-name | APPROVE / REQUEST_CHANGES | 0/1/2/0 | Summary |

### CI Status
- PR #NNN: passing / failing (details)

### Watch List
PRs that patrol has reviewed and must continue tracking until merged/closed. **This section is cumulative** — carry forward entries from previous patrols, updating the SHA and decision columns. Remove entries only when the PR is merged or closed.

| PR | Head SHA | Decision | Last Reviewed | Notes |
|----|----------|----------|---------------|-------|
| #NNN | abc1234f | COMMENT | 2026-05-08 14:30 | Waiting for author to address 2 findings |
| #MMM | def5678a | APPROVE | 2026-05-08 14:30 | Clean — remove on next patrol if merged |

- **Head SHA**: the `headRefOid` at the time of review. Step 1b compares this to the current head to detect new commits.
- **Decision**: the review event posted (APPROVE / COMMENT / REQUEST_CHANGES).
- **Notes**: brief context for the next iteration.

When writing this section: merge new reviews into the existing watch list. Update SHA/decision/timestamp for re-reviewed PRs. Prune merged/closed PRs (check via `gh pr view <NUMBER> --json state --jq .state`).

### Friction Encountered
- [Any issues with environment, tooling, or process]
```

---

## Step 5: Cleanup and Final State Gate

Before exiting successfully, remove all per-run scratch data and prove the repository is clean. If the run failed, set `PATROL_RUN_FAILED=1` before this step to preserve scratch files for debugging:

```bash
if [ "${PATROL_RUN_FAILED:-0}" = "1" ] || [ "${PATROL_KEEP_TMP_ON_FAILURE:-0}" = "1" ]; then
  echo "Preserving patrol scratch directory: $PATROL_RUN_DIR"
else
uv run python -c "
import os
import shutil
import subprocess
import sys
from pathlib import Path

raw = os.environ.get('PATROL_RUN_DIR', '').strip()
if not raw:
    print('PATROL_RUN_DIR is not set', file=sys.stderr)
    sys.exit(1)

repo_root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
run_dir = Path(raw)
if not run_dir.is_absolute():
    run_dir = repo_root / run_dir
tmp_root = (repo_root / '.patrol/tmp').resolve()
resolved = run_dir.resolve()
if resolved == tmp_root or tmp_root not in resolved.parents:
    print(f'refusing to delete outside .patrol/tmp: {resolved}', file=sys.stderr)
    sys.exit(1)

shutil.rmtree(resolved)
if resolved.exists():
    print(f'failed to remove patrol scratch directory: {resolved}', file=sys.stderr)
    sys.exit(1)
"
fi
git status --short
```

The final `git status --short` output must be empty. If it is not empty:

1. Delete or move only known untracked patrol artifacts created by this run.
2. Leave tracked, staged, or unknown user files untouched.
3. Report the remaining dirty state as a patrol failure instead of marking the scheduled run successful.

---

## Important Guidelines

- **Delegate reviews to `/pr-reviewer`** — patrol discovers and dispatches; pr-reviewer reviews and posts
- **Scope quality checks** to affected packages — don't run the full suite for a single-package change
- **Don't chase pre-existing failures** — if tests fail on code not in the diff, note it and move on
- **Keep patrol state ignored** — persistent notes live in `.patrol/SHARED_TASK_NOTES.md`; scratch files live in `.patrol/tmp/<run-id>/`
- **Keep handoff notes concise** — the next iteration needs to quickly understand what happened
