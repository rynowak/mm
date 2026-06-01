---
name: add-telemetry
description: |
  Reviews and adds query-friendly telemetry: structured logs, correlation fields,
  spans, and metrics. Enforces OpenTelemetry signal definitions and the principle
  of instrumenting boundaries and decisions, not internals.
  Use when user asks to "add telemetry", "instrument this code", "review telemetry",
  "add observability", "correlate logs and traces", or "improve observability".
file_dependencies:
  - path: docs/telemetry-context.md
    description: "Telemetry libraries, conventions, exporters, correlation patterns"
    template: templates/telemetry-context.md.tmpl
---

# Add Telemetry

## Overview

Add telemetry that lets operators and agents answer "what happened and why?"
without text search. Use OpenTelemetry primitives (spans, logs, metrics) at
boundaries and decisions. Do not introduce custom abstractions, schema registries,
or telemetry frameworks unless the repo already uses them.

## When to Use

- Adding or reviewing telemetry for a feature, handler, job, tool, stream, or
  background workflow.
- Code changes a boundary, decision, dependency call, state transition, retry,
  or streaming path.
- Investigation would otherwise require text search instead of stable fields.

## When NOT to Use

- Security-only audit completeness review — use `sec-logging`.
- Production log querying or dashboards — out of scope.

## Dependencies

Read repo-specific telemetry context before instrumenting:
- `AGENTS.md` (telemetry section if present)
- Repo telemetry docs (e.g. `docs/telemetry.md`, `docs/observability.md`)
- Existing instrumentation in the code being changed

If a `telemetry-context.md` template has been filled in for this repo, read it
first — it defines the repo's telemetry patterns, libraries, and conventions.

## Workflow

### 1. Match existing patterns first

Before adding anything, read the existing telemetry in the codebase:
- What logging library/framework is used?
- How are structured fields attached (extra=, attributes, tags)?
- Are there existing span/activity decorators or context managers?
- What metrics already exist?
- How is correlation (trace_id, request_id, session_id) propagated?

Match those patterns. Do not introduce a parallel style.

### 2. Add telemetry at boundaries and decisions

Add telemetry where an operator or agent later needs to know what happened and
why. **Do not log every helper or internal function.**

| Category | Examples |
|----------|----------|
| **Entry points** | HTTP/gRPC handlers, MCP tools, scheduled jobs, dispatcher loops, CLI commands, WebSockets, message consumers. |
| **Decisions** | Auth accepted/denied, validation failed, rate limited, dedup hit/miss, trigger skipped, lease acquired/denied, retry, timeout, capacity, status change. |
| **External work** | Database calls, HTTP/gRPC outbound, file/blob persistence, message queue publish, third-party API calls. |
| **Lifecycle** | Startup/readiness/shutdown, session started/completed, dispatch started/completed/failed, reconnect/cancel. |
| **Streams** | Direction and event type only; never raw payloads or deltas. |

Each changed workflow should log its boundary plus operator-relevant branch
outcomes and reasons.

### 3. Choose the right OTEL signal

| Signal | Use for | Rules |
|--------|---------|-------|
| **Structured log** | Stable facts and query joins. | Keep the message stable. Put IDs/counts/status in structured fields. High-cardinality IDs (request_id, session_id) belong here. |
| **Span** | Causality, duration, and error attribution. | Use stable names. One span per logical operation. Set status on error. Add safe attributes. |
| **Metric** | Counts, durations, queue depth, retries, drops, health aggregates. | Labels MUST be low-cardinality. Never use request/session/task IDs as metric dimensions. |

**When to use each:**

- Need to know *if* something happened → structured log
- Need to know *how long* and *what caused failure* → span
- Need to aggregate across many requests for dashboards/alerts → metric
- Most operations need a span AND a log (the span gives duration/causality, the log gives queryable fields)

### 4. Use useful fields, not a grand schema

Prefer field names already used in the component. Common useful fields:

- **Correlation**: `request_id`, `session_id`, `trace_id`, `span_id`, `task_id`, `correlation_id`
- **Actor/scope**: `user_id`, `tenant_id`, `service_name`, `agent_name`, `tool_name`, `action`
- **Outcome**: `status`, `outcome` (success/failure/skipped/denied/timeout), `error_code`, `reason`, `count`, `duration_ms`, `attempt`
- **Dependency**: `rpc.method`, `rpc.system`, `http.route`, `http.response.status_code`, `db.operation`

Do not add schema-version fields, global event taxonomies, or deny-by-default
registries unless the repo already uses them. Extend existing models rather than
inventing parallel ones.

### 5. Preserve correlation without framework churn

1. Carry existing request/session/task IDs through the code path.
2. For spawned tasks (async, threadpool), propagate context explicitly or copy it at task creation.
3. Add narrower IDs inside tool calls, stream events, retries, and leases.
4. Logs from spawned work must be joinable back to the parent workflow.

### 6. Redact and control volume

**Never log:** tokens, secrets, signatures, keys, authorization headers, passwords,
raw prompts/messages, model output, tool args/results, webhook payloads, or
content-part payloads.

**Use safe derivatives:** type, status, count, size, duration, schema name, reason
code, exception class, bounded sanitized error message, or event type.

**Volume rules:**
- INFO for low-volume boundaries/decisions
- WARNING/ERROR for failures, drops, denials, timeouts, or data-loss risk
- DEBUG for high-volume detail
- Streams: log type and direction only, never content

### 7. Test semantics

Assert on fields, not message strings. Cover:

| Case | Assert |
|------|--------|
| Boundary log | Stable message or event name, correlation IDs, safe dimensions |
| Branch decision | Skipped/denied/retry/timeout path includes `reason` or `error_code` |
| Spawned work | Child logs retain join fields |
| Stream telemetry | Type/direction present; payload absent |
| Redaction | Unsafe content cannot be emitted |
| Span | Stable name, safe attrs, error/duration behavior |
| Metrics | Labels exclude high-cardinality IDs |

## Output Format

When reviewing telemetry without editing:

```markdown
## Telemetry Review

| Workflow | Current coverage | Missing telemetry | Recommended signal |
|----------|------------------|-------------------|--------------------|

## Required changes
1. <file/function>: <specific telemetry to add>

## Safety checks
- Redaction: <any sensitive data risk?>
- Cardinality: <any high-cardinality metric labels?>
- Correlation: <can logs be joined across boundaries?>
- Tests: <are telemetry assertions present?>
```

When editing code, report only the behavior added and files changed.
