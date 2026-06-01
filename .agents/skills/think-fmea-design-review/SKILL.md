---
name: fmea-design-review
description: |
  Challenges design thinking with ephemeral Failure Mode and Effects Analysis, then folds reliability, rollout, recovery, and agent-control gaps back into the design. Use when user asks to "run FMEA", "failure mode analysis", "challenge this design", "production failure modes", "reliability design review", "rollback risk", or "operational risk review".
---

# FMEA Design Challenge

## Overview

Use Failure Mode and Effects Analysis (FMEA) as an internal reasoning tool to challenge a design before implementation. The FMEA itself is not the deliverable; the deliverable is a stronger design with clearer requirements, controls, verification gates, observability, rollout safety, and open questions.

**IMPORTANT:** Do not create or preserve a standalone FMEA document unless the user explicitly asks for one. Use FMEA to find design gaps, then fold the outcomes into the design's Requirements, Recommended Approach, Migration Plan, Test Strategy, Risk Assessment, Observability, or Decision Records sections.

## When to Use

- Writing or reviewing a design document, RFC, ADR, migration plan, or architecture proposal before implementation.
- Challenging a design for reliability, operability, rollout safety, rollback safety, dependency failure, data integrity, or recovery gaps.
- Adding an adversarial production-readiness lens during design creation or after architecture, security, privacy, performance, or testability review.

## When NOT to Use

- Security attacker modeling - use `threat-modeling` or `attack-reasoning`.
- Privacy risk analysis - use `privacy-reasoning`.
- Code-only bug hunting without a design or implementation plan - use `code-review` or `test-engineer`.
- Production incident root cause analysis - use `adversarial-investigation`, `kusto-query`, `mdm-metrics`, or `local-dev` depending on the environment.

## Quick Start

```text
/fmea-design-review docs/design/new-cache-layer.md
```

If no file path is provided, use the design currently being created or reviewed. If needed, find the relevant design document from the user request or search `docs/design/`, `docs/refactor/`, and component-local `docs/design/` directories.

## Core Instructions

### Step 1: Establish Design Inputs

Identify:
1. The design artifact under review.
2. The proposed components, APIs, data flows, dependencies, rollout phases, and operational controls.
3. The stated requirements, constraints, non-goals, and success metrics.
4. Any claims about reliability, latency, durability, consistency, observability, rollback, migration safety, graceful degradation, or agent-managed operation.

**Success criteria:** The working analysis names the design artifact and lists the main design elements that were challenged.

### Step 2: Build the Challenge Scope

Create a compact inventory of review targets:

| Target Type | Examples |
|-------------|----------|
| Component | Service, worker, database, cache, queue, scheduler |
| Interface | API, gRPC method, event contract, schema, CLI command |
| Data flow | Read path, write path, sync flow, async processing flow |
| Dependency | External service, package, identity provider, storage account |
| Operation | Deployment, migration, backfill, rollback, failover, recovery |
| Control | Validation, timeout, retry, circuit breaker, metric, alert, audit log |

Do not review every line of the design. Pick the targets whose failure would most affect correctness, availability, user trust, operational safety, or reversibility.

**Success criteria:** The challenge covers at least one target from each target type that is present in the design.

### Step 3: Identify Failure Modes

For each high-value target, ask these prompts:
1. What happens if this component is unavailable, slow, inconsistent, overloaded, or returns malformed data?
2. What happens if this dependency violates the design's assumptions?
3. What happens if retries, timeouts, cancellation, idempotency, or ordering behave differently than expected?
4. What data can be lost, duplicated, corrupted, exposed to the wrong tenant, or made unrecoverable?
5. What can go wrong during deployment, migration, backfill, rollback, partial rollout, or mixed-version operation?
6. What user-visible failure appears first, and which telemetry detects it?
7. What failure would operators miss until customers report it?

**Success criteria:** Each failure mode is concrete, tied to a design element, and worded as a falsifiable production scenario.

### Step 4: Score with Calibrated Labels

Use qualitative labels instead of fake-precision numeric RPN scores.

| Dimension | Low | Medium | High |
|-----------|-----|--------|------|
| Severity | Minor degradation, easy retry, no data risk | Partial outage, incorrect behavior, delayed recovery, limited data risk | Broad outage, data loss/corruption, tenant boundary risk, irreversible migration, or no safe rollback |
| Likelihood | Requires rare preconditions or multiple independent failures | Plausible under normal growth, dependency instability, or operator error | Expected under realistic load, common dependency behavior, or normal rollout conditions |
| Detectability | Existing metric, alert, test, or invariant catches it quickly | Some signals exist but require manual correlation or delayed investigation | No clear signal, silent data issue, or customers likely detect it first |

Assign **Priority**:
- **Blocking**: High severity plus Medium/High likelihood or High detectability risk.
- **High**: High severity with plausible controls, or Medium severity with High likelihood.
- **Medium**: Meaningful failure with contained blast radius or straightforward mitigation.
- **Low**: Minor failure or already well-controlled risk.

**Success criteria:** Every failure mode has Severity, Likelihood, Detectability, Priority, and a one-sentence rationale for the priority.

### Step 5: Challenge Existing Controls and Agent Loops

For each failure mode, evaluate controls in this order:
1. **Prevention:** validation, schema checks, quota, idempotency, isolation, bounded concurrency.
2. **Containment:** timeouts, circuit breakers, bulkheads, feature flags, partial degradation, tenant isolation.
3. **Detection:** metrics, logs, traces, audits, SLOs, alerts, invariant checks.
4. **Recovery:** retry policy, replay, rollback, backfill repair, data restore, manual runbook.
5. **Verification:** unit test, integration test, load test, chaos/fault injection test, migration dry run, canary gate.
6. **Agent control loop:** responsible agent or skill, signal source, allowed action, evidence requirement, authority boundary, and fallback if the agent or telemetry is unavailable.

Do not credit a control unless the design explicitly includes it or the codebase evidence shows it already exists. Label unverified controls as `ASSUMED`.

**Success criteria:** Blocking and High priority rows include either a concrete mitigation or a clearly stated open question.

### Step 6: Convert Findings into Design Inputs

Translate the working FMEA into design amendments. Do not require the design to retain the FMEA scoring table. Make the design stronger by adding or updating:

| Design Section | What to Fold In |
|----------------|-----------------|
| Requirements | New reliability, reversibility, idempotency, compatibility, or operational requirements |
| Recommended Approach | Control placement, fallback behavior, degradation strategy, bounded retry/cancellation behavior |
| Migration Plan | Canary gates, mixed-version safety, rollback strategy, data repair or replay plan |
| Test Strategy | Failure injection, mixed-version tests, migration dry runs, invariant checks |
| Risk Assessment | Only the residual risks and mitigation rationale that matter after design changes |
| Observability | Metrics, logs, traces, SLOs, alerts, agent-readable signals |
| Decision Records | Significant trade-offs discovered by the challenge |
| Open Questions | Decisions that require human or service-owner input |

If you are editing the design document, apply the amendments directly to the appropriate sections. If you are only reviewing, output the specific amendments for the author to fold in.

**Success criteria:** Every Blocking or High priority challenge becomes a design amendment, verification gate, or open question.

### Step 7: Produce the Challenge Summary

Use this output format for the transient review summary:

```markdown
## FMEA Design Challenge

**Reviewed artifact:** {path or title}
**Scope challenged:** {components / flows / operations reviewed}
**Outcome:** {Ready after amendments | Conditional | Needs design changes}

### Design Amendments to Fold In

| Priority | Design Area | Challenge | Design Input |
|----------|-------------|-----------|--------------|
| {Blocking/High/Medium/Low} | {section or element} | {failure scenario} | {requirement/control/test/metric/open question to add} |

### Agent-Managed Controls to Add or Clarify

| Control | Signal Source | Responsible Agent/Skill | Allowed Action | Evidence Required | Human Boundary |
|---------|---------------|-------------------------|----------------|-------------------|----------------|
| {control} | {metric/log/test/gate} | {agent or skill} | {comment/block/open issue/escalate/recommend rollback} | {proof} | {requires approval for destructive/release action} |

### Open Questions

- {question that must be answered before implementation}

### Positive Design Controls Observed

- {Control}: {failure mode it helps mitigate}
```

## Review Rules

- Do not invent requirements or controls. If the design does not say how a failure is handled, mark the control as `None Found` or `ASSUMED`.
- Do not report generic risks. Tie every failure mode to a specific design element, flow, dependency, or operation.
- Do not duplicate security or privacy findings unless the failure mode is primarily reliability or operability related.
- Do not require zero risk. Recommend proportionate controls that match the priority and blast radius.
- Prefer design changes that improve reversibility, containment, and observability over broad rewrites.
- Do not preserve FMEA scoring in the design unless the user explicitly asks. The design should retain the decision, control, test, signal, or open question, not the analysis worksheet.

## Examples

### Example: Cache Introduction

**Challenge:** Cache returns stale authorization data after permission revocation.

| Field | Value |
|-------|-------|
| Priority | Blocking |
| Design Input | Add a requirement for revocation-triggered invalidation or authoritative checks for sensitive actions. |
| Verification Gate | Integration test for permission revocation plus metric for stale-auth fallback count. |
| Agent Control | PR/design reviewer blocks implementation until the test and metric are present; production telemetry agent opens an issue if stale-auth fallback count exceeds threshold. |

### Example: Migration Plan

**Challenge:** Mixed-version services write incompatible records during phased rollout.

| Field | Value |
|-------|-------|
| Priority | Blocking |
| Design Input | Add an expand-migrate-contract plan with dual-read or dual-write compatibility. |
| Verification Gate | Mixed-version integration test and canary metric for parse failures. |
| Agent Control | CI agent blocks the PR on missing mixed-version tests; release agent treats parse-failure canary regression as a stop signal and requires human approval before continuing. |
