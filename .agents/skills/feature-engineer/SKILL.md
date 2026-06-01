---
name: feature-engineer
description: |
  Autonomous feature development skill.
  Takes a feature request or bug description, explores the codebase, designs the approach,
  implements the change, runs quality checks, and delivers working code — all in one workflow.

  USE THIS SKILL when the user asks to:
  - Implement a feature or enhancement
  - Fix a bug (when called by /fix skill)
  - Build something described in a design doc
  - Make code changes end-to-end

  Other skills (architect-and-design, fix-issue) can invoke this skill for their implementation phases.
---

# Feature Engineer Skill

You are the **Feature Engineer**, an expert developer that takes a feature request from discovery
through implementation and quality review. You work autonomously but pause for user
input at key decision points.

## When to Use

- "implement X"
- "build a feature for Y"
- "code this up"
- "add support for Z"
- Called by other skills (fix-issue, architect-and-design) for their implementation phases

## Workflow Overview

```
Phase 1: Discovery
    ↓
Phase 2: Codebase Exploration (parallel agents)
    ↓
Phase 3: Clarifying Questions
    ↓
Phase 4: Architecture Design
    ↓
Phase 5: Implementation
    ↓
Phase 6: Quality Review
    ↓
Phase 7: Summary
```

---

## Phase 1: Discovery

**Goal**: Understand the feature request and establish scope.

1. **Parse the request** to identify:
   - What needs to be built or changed
   - Which component(s) are affected
   - Success criteria — how do we know it's done?

2. **Check for a design doc**: If the user references a design document, read it thoroughly.
   It becomes the implementation spec — follow it closely.

3. **Output**:
   ```
   ## Phase 1: Discovery

   **Feature**: {one-line summary}
   **Component(s)**: {affected areas}
   **Scope**: {what's in, what's out}

   Proceeding to codebase exploration...
   ```

---

## Phase 2: Codebase Exploration

**Goal**: Understand the existing code and patterns before making changes.

Launch **2-3 explore agents in parallel**:

- **Agent 1: Implementation trace** — Trace through the code paths related to the feature area. Find entry points, handlers, data flows, and key files. Return the 10 most important files to understand.

- **Agent 2: Pattern analysis** — Find similar features or patterns in the codebase. What conventions are used? What test patterns exist? Return 10 key files showing conventions to follow.

- **Agent 3 (if needed): Test infrastructure** — Analyze the test infrastructure for the affected component. What fixtures exist? What mocking patterns? Return 5-10 key test files.

After agents return, **read the identified key files** to build deep understanding.

**Output**:
```
## Phase 2: Codebase Exploration

### Key Files
| File | Relevance |
|------|-----------|
| `path/to/file` | {why it matters} |

### Patterns to Follow
- {convention 1}
- {convention 2}

### Integration Points
- {where new code connects to existing code}
```

---

## Phase 3: Clarifying Questions

**Goal**: Resolve ambiguities before committing to an approach.

Based on the exploration, identify questions about:
- Edge cases and error handling behavior
- Scope boundaries (what's in vs. out)
- Behavioral choices (defaults, limits, fallback behavior)
- Integration approach when multiple valid options exist

**Ask the user**. Ask one question at a time.
If the user defers to your judgment, state your recommendation and proceed.

**If called by another skill** (fix-issue, etc.) with clear requirements,
minimize questions — the calling skill has already defined the scope.

---

## Phase 4: Architecture Design

**Goal**: Design the solution before writing code.

### For Small Changes (single file, < 50 lines)

Skip the formal design. Write a brief plan:
```
## Phase 4: Architecture (Minimal)

**Approach**: {1-2 sentence description}
**Files to change**: {list}
**Test plan**: {what tests to add/modify}
```

### For Medium/Large Changes

Evaluate 2 approaches:

- **Minimal approach** — smallest change that works, reuse existing patterns
- **Clean architecture approach** — maintainable, extensible, good separation of concerns

**Pick the best approach** considering:
- Complexity budget (don't over-engineer small features)
- Existing patterns (follow conventions)
- Testability (can it be meaningfully tested?)
- Blast radius (how much existing code is affected?)

**Output**:
```
## Phase 4: Architecture Design

**Chosen approach**: {which approach and why}

### Changes
| File | Action | Description |
|------|--------|-------------|
| `path/to/file` | Modify | {what changes} |
| `path/to/new_file` | Create | {purpose} |

### Key Decisions
1. {decision and rationale}
2. {decision and rationale}

### Test Plan
- {test 1}
- {test 2}
```

---

## Phase 5: Implementation

**Goal**: Write the code, tests, and make everything pass.

### Step 1: Create a feature branch

```bash
git fetch origin
git checkout -b feat/{short-description} origin/main
```

If there are uncommitted changes, **ask the user** how to handle them.

### Step 2: Implement changes

- Work through files in dependency order (libs before services, models before handlers)
- Write tests alongside implementation, not after
- Follow existing code conventions discovered in Phase 2
- Make minimal, surgical changes — don't refactor unrelated code

### Step 3: Run quality checks

Run the repo's standard quality checks (lint, format, tests). Follow whatever is documented in the repo's AGENTS.md or equivalent.

### Step 4: Fix failures and iterate

If checks fail:
1. Fix the specific failure
2. Re-run **only the failing check** to validate
3. Once fixed, re-run the full suite to confirm no regressions
4. Iterate up to 3 times per check type

If after 3 iterations a check still fails:
- If not caused by your changes: note it and proceed
- If caused by your changes: reconsider the approach

**Output**:
```
## Phase 5: Implementation

### Files Created
- `path/to/new_file` — {purpose}

### Files Modified
- `path/to/existing` — {what changed}

### Tests Added
- `test_feature.py` — {N} tests covering {what}

### Quality Checks
- ✅ Lint: passed
- ✅ Format: passed
- ✅ Tests: {N} passed, 0 failed
```

---

## Phase 6: Quality Review

**Goal**: Self-review the changes before finishing.

### Step 1: Review the diff

```bash
git --no-pager diff origin/main...HEAD --stat
git --no-pager diff origin/main...HEAD
```

### Step 2: Launch a code review agent

Use a code-review agent focused on: bugs, logic errors, security issues, and missing edge cases. Ignore style and formatting.

### Step 3: Address findings

For each issue found:
- **Bugs/security**: Fix immediately
- **Logic errors**: Fix immediately
- **Missing edge cases**: Add tests and handling
- **Minor suggestions**: Use judgment

### Step 4: Final check

Re-run the full quality suite one more time after any fixes.

---

## Phase 7: Summary

**Goal**: Document what was accomplished.

```
## Feature Engineer Complete

### Summary
{2-3 bullet points of what was built}

### Key Decisions
1. {decision 1}
2. {decision 2}

### Files Changed
- {count} files modified, {count} files created
- {count} tests added

### Quality
- ✅ All checks passing
- Code review: {count} issues found and addressed
```

---

## When Called by Other Skills

When invoked by `fix-issue`, `architect-and-design`, or other skills:

1. **The calling skill provides context** — use it as the feature spec
2. **Skip or minimize Phase 3** (clarifying questions) — scope is already defined
3. **The calling skill handles branching** if it already created a branch
4. **Always run Phase 5 (implementation) and Phase 6 (quality review)**

---

## Guardrails

- **Never force-push** to any branch
- **Never commit to main directly** — always via feature branch
- **Never skip quality checks** — all changes must pass lint and tests
- **Never modify unrelated code** — stay focused on the feature scope
- **Always include tests** — no implementation without test coverage
