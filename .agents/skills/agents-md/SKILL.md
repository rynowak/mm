---
name: agents-md
description: |
  Reviews and improves AGENTS.md files against architecture standards.
  Use when user asks to "review AGENTS.md", "improve AGENTS.md", "check AGENTS.md",
  "audit agent config", "fix AGENTS.md", "AGENTS.md quality", "agents-md review",
  "is my AGENTS.md good", or needs guidance on AGENTS.md structure and content.
---

# AGENTS.md Review & Improvement

Review, assess, and improve AGENTS.md files against the tiered architecture standard.

## Navigation

| Topic | Location |
|-------|----------|
| Review rubric | [references/rubric.md](references/rubric.md) |
| Calibration patterns | [references/calibration.md](references/calibration.md) |
| Reasoning techniques | [references/reasoning-techniques.md](references/reasoning-techniques.md) |

---

## Overview

AGENTS.md is the single source of repo-specific context for AI agents. It must be tight, actionable, and maintainable. This skill evaluates existing AGENTS.md files and proposes concrete improvements.

## When to Use

- Reviewing an existing AGENTS.md for quality and completeness
- Improving a bloated or vague AGENTS.md
- Checking compliance with the tiered architecture
- After major repo changes (new CI, new deploy target, new toolchain)

## When NOT to Use

- Creating a brand-new AGENTS.md from scratch — use the template from `skills-sync --init` instead
- Writing skills — use **skill**

---

## Core Workflow

### 1. Read the AGENTS.md

Read the root AGENTS.md. If component-level AGENTS.md files exist, note them but focus on root first.

**Success criteria:** You have the full text and line count.

### 2. Assess against rubric

Evaluate each area. Be specific — cite line numbers and quote text.

| Area | Pass criteria |
|------|---------------|
| **Size** | Root ≤ 200 lines (warn at 180) |
| **Identity** | 2-3 sentences: what, stack, deploy, CI. Has doc pointers. |
| **Working Style** | Present. Behavioral, not aspirational. |
| **Source Control** | Present. Enough for `gh`/`git` to work first try. |
| **Issues & PRs** | Present. Agent can create a PR correctly on first attempt. |
| **CI/CD** | Present. Maps commands to CI gates. |
| **Package Management** | Present. NOT ALLOWED items bolded. Auth commands included. |
| **Repo Invariants** | Hard rules only. No advisory prose. Imperative voice. |
| **Prerequisites** | Table format. Includes auth requirements. |
| **PR Quality Checklist** | Bash code block. Exact CI gate commands. |
| **Testing Guardrails** | Hard rules only. Not general advice. |
| **Patterns TOC** | Bounded table. Links resolve. No component enumeration. |
| **Taxonomy** | Root has only IDENTITY + INVARIANT + ROUTING content. No KNOWLEDGE or REFERENCE. |
| **Calibration** | No vague qualifiers ("thorough", "careful", "important"). All thresholds explicit. |
| **Autonomy boundaries** | Clear. Distinguishes local/reversible from team-impacting. |

### 3. Produce findings

Format each finding as:

```
[AREA] LINE N: "quoted text"
  Problem: what's wrong
  Fix: specific replacement text or action
```

Severity levels:
- **FAIL** — violates a hard rule (size budget, taxonomy, missing required section)
- **WARN** — weakens effectiveness (vague language, stale links, advisory prose in invariants)
- **OK** — passes

### 4. Propose edits

After findings, produce the concrete edits. Prefer:
- **Deletion** over rewording (compaction bias: removal worth 2x addition)
- **Moving** content to docs/ over keeping it inline
- **Specificity** over generality (replace "be careful" with enumerated checks)

### 5. Verify post-edit

After applying changes, re-check:
- [ ] Line count ≤ 200
- [ ] All links resolve
- [ ] No taxonomy violations
- [ ] No vague qualifiers remain

---

## Anti-patterns to flag

| Pattern | Problem | Fix |
|---------|---------|-----|
| Component enumeration | Grows unboundedly, stale | Replace with discovery pointer |
| "Be thorough/careful" | Uncalibrated — models ignore it | Enumerate what to check |
| Prose paragraphs in invariants | Skimmed and forgotten | Convert to table or bullet rules |
| Duplicating docs/ content | Bloats root, goes stale | Link to doc, remove inline |
| Missing auth/feed setup | Agent gets stuck on first `install` | Add exact commands |
| "When appropriate" | Ambiguous trigger | Replace with explicit condition |

---

## Additional Resources

- [references/rubric.md](references/rubric.md) — Structure, sections, content rules, taxonomy
- [references/calibration.md](references/calibration.md) — Objective instruction patterns
- [references/reasoning-techniques.md](references/reasoning-techniques.md) — Verified reasoning patterns
