---
name: local-dev
description: >
  Troubleshoot and investigate services running in the local development environment.
  Use when the user asks to debug a service, check why something is failing, investigate
  an error, verify changes work, look at logs or traces, check health, or diagnose
  local dev issues. Triggers on: 'debug', 'check logs', 'why is it failing', 'test locally',
  'verify my changes', 'service won't start', 'check health', 'is it working'.
dependencies:
  - docs/local-dev-*.md
---

# Local Dev Troubleshooting

Investigate and verify services running in the local development environment. Covers container/process state, logs, traces, metrics, and health checks.

**Act, don't theorize.** When the user wants to know if something works locally, test it. Don't explain what *would* happen — find out.

## When to Use

- A service is returning errors, crashing, or not starting
- A request is slow and you need to find the bottleneck
- Verifying a code change works end-to-end locally
- Diagnosing inter-service communication issues
- User asks "does this work?", "test this", or any local verification

## When NOT to Use

- Production incidents (use `query-telemetry` or `investigate-production`)
- CI/CD failures (use `ci-cd-diagnosis`)

## Prerequisites

Read the relevant `docs/local-dev-*.md` file(s) before doing anything. These documents describe:

- How to start the service(s) (commands, prerequisites)
- Available observability tools (log backends, trace UIs, metrics endpoints)
- Service ports, health check endpoints, and how to verify liveness
- Common failure modes and their resolutions

If the repo has multiple services, there will be multiple docs following the naming convention `local-dev-<service-name>.md`. Read the one relevant to what the user is asking about. If unclear which service is involved, check all of them.

## Workflow

### 1. Determine Service State

Before investigating deeper, establish baseline:

- Is the service running? Check process/container status.
- Is it healthy? Hit health endpoints.
- Is it reachable? Verify ports are bound and accepting connections.

If the service isn't running, check startup logs first — many issues are import errors, missing env vars, or config problems that show up immediately.

### 2. Investigate with Observability Tools

Use whatever the local environment provides (documented in `docs/local-dev-*.md`):

| Signal | Use for |
|--------|---------|
| Logs | Errors, warnings, request details, stack traces |
| Traces | Request flow, latency breakdown, inter-service calls |
| Metrics | Throughput, error rates, resource usage |

**Prefer structured observability tools over raw log tailing.** If the environment has Loki, Jaeger, Prometheus (or equivalents), use their APIs rather than scrolling through stdout.

Exception: if a service crashes at startup before telemetry is initialized, fall back to raw container/process logs.

### 3. Reproduce and Test

When verifying a fix or testing a change:

1. Make the request or trigger the scenario
2. Check the result (response, side effects)
3. Verify in observability tools (no errors in logs, trace looks correct, metrics updated)

### 4. Correlate Across Signals

Same principle as production investigation:

- **Logs → Traces**: Find error, extract trace ID, see full request flow
- **Traces → Logs**: Find slow span, check logs in that time window
- **Metrics → Logs**: Spot anomaly in metrics, drill into logs for details

## Common Scenarios

### Service won't start

1. Check process/container status
2. Read startup logs (may not reach observability backends if crash is early)
3. Common causes: missing env var, auth expired, port conflict, syntax error

### Service returning errors

1. Query error logs
2. Look at error traces for stack context
3. Use trace ID to follow the request across services

### Request is slow

1. Find slow traces (filter by duration)
2. Inspect span breakdown — which step is the bottleneck?
3. Check if it's systemic (metrics) or isolated (one request)

### Inter-service call failing

1. Confirm both services are running and healthy
2. Check caller logs for the error
3. Check callee logs for corresponding failure
4. Use distributed tracing to see both sides

### Telemetry not showing up

1. Verify observability backends are running
2. Confirm at least one request has been made (telemetry needs traffic)
3. Check for export delays (metrics often batch on intervals)
4. Look at collector/agent logs if available

## Growing the Local Dev Docs

After resolving issues, consider updating `docs/local-dev-*.md`:

- **New debugging techniques** that worked well
- **Common failure modes** and their resolutions
- **Useful commands or queries** for this specific service
- **Prerequisites** that weren't previously documented (auth steps, dependencies)

Suggest additions to the user: *"This was a tricky one — want me to add it to the local-dev doc so it's faster next time?"*
