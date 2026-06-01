# AGENTS.md Rubric

## Content Taxonomy

Every section of an AGENTS.md must map to exactly one class. This is the primary tool for evaluating placement.

| Class | Definition | Placement |
|-------|------------|-----------|
| **IDENTITY** | What this thing is in one paragraph. Tech stack, deploy target, CI system — with inline pointers to docs for details. | Root (repo) / Local (component) |
| **INVARIANT** | A hard rule that must be followed on every task. Breaks something material if violated. | Root (repo-wide) / Local (component-wide) — **always inline, never linked out** |
| **ROUTING** | "When you touch X, go read Y." A pointer. | Root Patterns TOC / Local (routes to design docs) |
| **PATTERN** | How to do a specific thing the right way. Concrete steps or code. | `docs/<topic>.md` — **not AGENTS.md** |
| **KNOWLEDGE** | A fact about the system. | `docs/*.md` — **not AGENTS.md** |
| **REFERENCE** | A lookup table: commands, versions, env vars, cluster lists. | `docs/*.md` — **not AGENTS.md** |

### Placement decision procedure

Ask in order:

1. **Cross-cutting PATTERN?** → `docs/<topic>.md`. Add one Patterns TOC entry if new.
2. **KNOWLEDGE or REFERENCE?** → `docs/*.md`. No root edit.
3. **Component-specific?** → `<component>/AGENTS.md`. Create if it doesn't exist.
4. **Cross-cutting INVARIANT?** → root AGENTS.md inline. Verify it's not better as a pattern doc.
5. **ROUTING?** → almost never a new root edit. New TOC rows only for genuinely new repeatable patterns.

## Root Structure

| Check | Pass |
|-------|------|
| Root exists | AGENTS.md at repo root |
| Size budget | ≤ 200 lines (warn at 180) |
| Identity paragraph | 2-3 sentences: what, stack, deploy target, CI. Inline pointers to low-frequency docs (deployment, pipelines). |
| Invariants inline | All INVARIANT content is in the root file — not linked out |
| Patterns TOC | Bounded table of `docs/*.md` pointers — O(patterns), not O(files). No component enumeration. No design-doc enumeration. |
| Maintenance note | Present at bottom — size budget rule + pointer to standard |

### Root skeleton

```
Identity paragraph (2-3 sentences + inline doc pointers)

## Working Style              [INVARIANT]
## Operating Guidelines       [INVARIANT]
## Prerequisites              [INVARIANT]
## PR Quality Checklist       [INVARIANT]
## Testing Guardrails         [INVARIANT — hard rules only]

## Patterns                   [ROUTING — bounded TOC]

  | Pattern | Doc |
  |---------|-----|
  | ...     | docs/... |

## Keeping this file small    [MAINTENANCE]
```

Section names are examples — repos may use different headings. What matters is taxonomy compliance: every section is IDENTITY, INVARIANT, or ROUTING. Nothing else.

## Invariant Content Guidelines

Invariants stay inline because they are loaded on every task. Examples:

- Working style (autonomous vs ask-first, scope limits)
- Git operations (GitHub instance, branch strategy, merge strategy, PR recipe)
- Package management rules (what tools, what's NOT ALLOWED)
- Quality gate commands (bash block of exact commands)
- Testing hard rules (parallelism safety, isolation, framework rules)
- Repo-specific hard rules (zero suppressions, naming conventions, etc.)

### What makes a good invariant

- Imperative voice ("Run tests" not "you should run tests")
- Explicit thresholds ("Happy path + key guard conditions" not "enough tests")
- Explicit conditions ("When X" not "when appropriate")
- Hard rules only — no advisory prose, no "try to", no "be thorough"

## Red Flags

| Pattern | Problem | Fix |
|---------|---------|-----|
| REFERENCE table inline | Wrong tier — loaded on every task, stale-risk | Move to `docs/*.md`, add Patterns TOC entry or inline pointer |
| Component enumeration | Grows unboundedly, stale | Replace with discovery pointer (`docs/repo-layout.md`) |
| Duplication with docs/ | Two sources of truth | Link to doc, remove inline — never restate |
| Patterns TOC > 15 rows | TOC is growing unboundedly | Some entries are not repeatable patterns — demote |
| "Be thorough/careful" | Uncalibrated — models ignore it | Enumerate what to check |
| Content that doesn't change agent behavior | Tax, not resource | Delete |
| Historical context / rationale | Not every-task content | Belongs in design doc |
| Vague qualifiers | "important", "thorough", "careful" | Replace with specific checks or thresholds |

## Patterns TOC Rules

The Patterns TOC earns a row only when the content is:
1. **Repo-wide** in scope (not component-specific)
2. **Hit with reasonable frequency** during typical tasks
3. **A repeatable procedure** (not architecture or rationale)

Disqualification:
- Component-specific → local AGENTS.md
- Low-frequency ops reference → identity-paragraph inline pointer
- Architectural detail / rationale → design doc, linked from local AGENTS.md
- Single-file-scope → inline comment or subpackage AGENTS.md
