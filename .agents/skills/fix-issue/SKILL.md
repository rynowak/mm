---
name: fix-issue
description: |
  Fix an issue end-to-end: branch off main, run RCA sub-agent in parallel with
  implementation, enforce regression test hard gate, and create follow-up for
  systemic improvements. Use when user asks to "fix issue #N", "fix bug #N",
  "resolve issue", "fix this issue", or references a specific issue number to fix.
---

# Fix Issue

Orchestrates the full lifecycle of fixing an issue: fetch details, branch, investigate root cause in parallel with implementing the fix, enforce regression tests, and present results.

## Workflow

```
Get issue details
    ↓
Create bugfix branch off latest main
    ↓
Check for repeat-fix patterns
    ↓ (parallel)
Launch RCA sub-agent  ←→  Implement the fix
    ↓
Collect RCA results + test gap analysis
    ↓
Present results to user
```

## Step 1: Get the Issue

If an issue number was provided as an argument, use it. Otherwise ask the user which issue to fix. Fetch the full issue details (title, body, labels, comments) — this becomes the spec for the fix.

## Step 2: Create Bugfix Branch

```bash
git fetch origin
git checkout -b fix/<issue-number>-<short-slug> origin/main
```

Derive `<short-slug>` from the issue title: lowercase, hyphens, max 4-5 words. Example: `fix/42-null-pointer-on-login`.

If there are uncommitted changes in the working tree, **stop and ask the user** how to handle them (stash, commit, or discard) before switching branches.

## Step 3: Check for Repeat-Fix Patterns

**BEFORE implementing a fix**, check if this issue is part of a whack-a-mole pattern — a class of bugs that keeps recurring because each prior fix only addressed a symptom.

Search for related closed issues in the same area.

**Indicators of a repeat-fix chain:**

- 2+ prior closed issues with similar symptoms in the same module/subsystem
- Prior fix commits that patched the same function or touched the same error-handling branch
- A sequence of fixes each catching "one more" edge case (new exception type, new response shape, new event name)

**If the issue matches a repeat-fix pattern:**

1. Note this in the implementation Discovery phase
2. Propose a **systemic fix** (error taxonomy, state machine, allow/deny-list, exhaustive pattern match) rather than another point fix
3. Frame the work as addressing the class of bugs, not just the specific instance

**If no prior pattern exists**, proceed normally — it may be a genuine one-off bug.

## Step 4: Launch RCA Sub-Agent (Background)

Immediately after Step 3, launch the RCA investigation sub-agent in the background. It runs **in parallel with the implementation**, so the fix and the root cause investigation happen concurrently.

The RCA agent investigates:
- Git history (`git log`, `git blame`) to find the introducing commit
- Related issues/PRs for pattern detection
- Test gaps — what test was missing that would have caught this
- Process improvements to prevent recurrence

**Do not wait for this agent to complete.** Proceed immediately to Step 5.

## Step 5: Implement the Fix

### 5a: If a feature-dev skill is available

Invoke it. Pass the issue details as the feature request context. Frame its Discovery phase around the bug.

### 5b: If no feature-dev skill is available

Fall back to an inline fix loop:

1. **Explore** the implicated code to confirm the bug location
2. **Form a hypothesis** about the root cause
3. **Implement the minimal fix** — one change that resolves the reported behavior; resist expanding scope
4. **Write a regression test** that would have failed on the pre-fix code
5. **Run project quality gates** (lint, type check, tests) and iterate until green

## Step 6: Collect RCA Results and Test Gap Analysis

After the implementation completes, collect the RCA sub-agent's results.

### 6a: Collect RCA Output

Extract these sections from the RCA agent's response:

- **Root cause** — one sentence with evidence
- **Introducing commit** — SHA, date, author
- **Context** — was this a refactor, new feature, previous fix?
- **Related issues** — related issues/PRs found
- **Test gap** — what test was missing
- **Process improvements** — actionable improvements
- **Confidence** — integer 0-100

If confidence is below 60, note "Root cause unconfirmed." Do not fabricate a cause.

### 6b: Test Gap Gate (Hard Gate)

Verify that the fix includes regression tests. This is a **hard gate** — do not proceed if no test files are in the diff.

```bash
git diff --name-only origin/main | grep -Ei "(^|/)(test_|_test\.|tests?/)"
```

**If no test files appear in the diff:**

Stop. Return to the implementation and add a regression test that:

- Exercises the exact code path that was broken
- Would fail on the pre-fix code
- Passes on the post-fix code

Re-run quality checks after adding the test, then return here.

**If test files are present**, assess the test quality:

| Question | Answer |
|----------|--------|
| Does this test exercise the exact failing scenario from the issue? | Yes / Partial / No |
| Would it have caught this bug if it had existed before the fix? | Yes / No |
| Does it test edge cases related to the root cause? | Yes / No |

### 6c: Reconcile RCA with Fix

Compare the RCA agent's findings with the actual fix:

- Does the fix address the root cause, or just the symptom?
- Did the RCA agent find related issues that the fix should also address?

If the fix only addresses the symptom and not the root cause, flag this to the user.

## Step 7: Present Results

Present a summary to the user:

```
## Fix Complete

**Issue**: #<number> — <title>
**Branch**: fix/<issue-number>-<short-slug>

### Root Cause Analysis
- Root cause: <one sentence>
- Introduced by: <commit SHA> (<date>)
- Confidence: <N>/100

### Regression Prevention
- Test gap: <what was missing>
- Regression test added: <test file:function>

### Process Improvements
<list if systemic, or "No systemic follow-up warranted">
```

The user decides whether to create a PR, follow-up issue, or take other action.

## Common Issues

| Problem | Resolution |
|---------|------------|
| Dirty working tree | Ask user to stash/commit before branching |
| Issue not found | Confirm repo and issue number |
| Merge conflicts on branch creation | Inform user, ask how to proceed |
| RCA confidence < 40 | Note "Root cause unconfirmed" — do not fabricate |
| No test files in diff (hard gate) | Return to implementation, add regression test |
| RCA finds repeat-fix chain | Flag to user — propose systemic fix |
