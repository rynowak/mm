---
name: query-telemetry
description: >
  Query and analyze production telemetry (logs, metrics, traces) to investigate
  issues, analyze performance, and understand system behavior. Use when the user
  needs to query logs, check metrics, investigate traces, debug production issues,
  or analyze telemetry data. Triggers on: 'query logs', 'check metrics', 'traces',
  'latency', 'error rate', 'investigate production', 'telemetry'.
file_dependencies:
  - path: docs/telemetry-context.md
    description: "Telemetry libraries, conventions, exporters, correlation patterns"
    template: templates/telemetry-context.md.tmpl
---

# Query Telemetry

Query and analyze production telemetry to investigate issues, understand system behavior, and diagnose problems. Covers logs, metrics, and traces.

**Act, don't theorize.** When the user wants to know what's happening in production, write and execute queries. Don't explain what you *would* query — query it.

## When to Use

- Investigating production errors or failures
- Analyzing latency, error rates, or request volumes
- Tracing requests across services
- Checking service health or resource usage
- Answering questions about system behavior with data

## Prerequisites

Read `docs/telemetry-context.md` before doing anything. It contains:

- Backend type and connection details (how to authenticate and connect)
- Available data sources (tables, metric namespaces, trace backends)
- Query language and syntax for this environment
- Common schemas and field names
- Team-specific conventions (namespaces, filters, metric names)

Without this context, you cannot write correct queries.

## Workflow

### 1. Understand the Question

Determine what signal type answers the question:

| Question type | Signal |
|---------------|--------|
| "Why is X failing?" | Logs (errors), then traces for context |
| "Is X slow?" | Metrics (latency percentiles), then traces for breakdown |
| "What happened at time T?" | Logs + traces correlated by time window |
| "How much traffic?" | Metrics (counters/rates) |
| "Show me the request flow" | Traces (distributed trace by ID) |

### 2. Write and Execute Queries

Follow the query patterns documented in `docs/telemetry-context.md`. General principles:

- **Start narrow, widen if needed.** Begin with a short time window (1h), expand only if results are empty.
- **Filter aggressively.** Apply all known filters (service name, environment, namespace) before scanning.
- **Use fast filters before expensive ones.** String-contains before regex. Indexed fields before free-text.
- **Limit results during exploration.** Cap at 50-100 rows until you know what you're looking for.
- **Iterate.** First query finds the shape of the problem. Follow-up queries drill into specifics.

### 3. Correlate Across Signals

Production issues rarely live in one signal:

1. **Logs → Traces**: Find error logs, extract trace ID, look up the full trace to see what failed and where.
2. **Metrics → Logs**: Spot an anomaly in metrics (error spike, latency jump), query logs in that time window for details.
3. **Traces → Metrics**: Find slow traces, check if the latency is systemic (metrics) or isolated (one bad request).

### 4. Analyze and Report

When presenting findings:

- Lead with the answer, not the query
- Include relevant data points (timestamps, error messages, latency values)
- Show the query you ran (so the user can re-run or modify)
- Suggest next steps if the root cause isn't yet clear

## Query Best Practices

- **Time ranges**: Always specify. Default to last 1 hour, expand to 6h or 24h only if needed.
- **Aggregations**: Use percentiles (p50/p90/p99) for latency, rates for counters, counts for errors.
- **Grouping**: Group by service, endpoint, or status code to find which component is responsible.
- **Sampling**: If result sets are huge, sample or aggregate rather than pulling raw rows.

## Growing the Telemetry Context

After every investigation, consider whether `docs/telemetry-context.md` should be updated:

- **New schemas discovered**: If you queried a table or metric not yet documented, add its schema.
- **Useful queries**: If you wrote a query that answered a common question, add it to the common patterns section.
- **Gotchas learned**: If you hit a quirk (unexpected field name, required filter, naming convention), document it.

Proactively suggest additions to the user: *"This query was useful — want me to add it to telemetry-context.md for next time?"*

The goal is to make `docs/telemetry-context.md` a living reference that gets richer over time, so future investigations start faster.

## Anti-Patterns

- Querying without time bounds (scans everything, slow and expensive)
- Pulling thousands of raw log lines without filtering (find the signal first)
- Ignoring the query patterns in `docs/telemetry-context.md` (they exist for a reason — correct field names, required filters)
- Theorizing about what the data might show instead of querying it
