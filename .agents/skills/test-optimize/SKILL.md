---
name: optimize-tests
description: |
  Profiles the Python test suite, identifies the slowest tests, and applies
  targeted performance fixes without reducing logical test coverage. Spawns
  reviewer agents after edits to verify no behavior went untested. Manual
  invocation only — runs when the user types `/optimize-tests`.
disable-model-invocation: true
---

# Optimize-Tests

Profile the workspace test suite, fix the slowest cases, and verify that
logical coverage is preserved. The goal is **wall-clock reduction without
losing behavior verification.**

## Flow

1. **Baseline.** Run the full suite with durations enabled and capture the
   timing distribution and total wall time.
2. **Pick targets.** Decide which tests are worth optimizing based on the
   distribution you just measured. No fixed N — the model picks based on
   shape (long tail vs few outliers).
3. **For each target, in series:**
   - Read the test and the code under test.
   - Diagnose the slowness using the playbook below.
   - Apply the fix.
   - Re-run just that test (or its file) to confirm it still passes and is
     actually faster.
4. **Coverage review.** Spawn parallel reviewer agents (one per modified
   test file) to check that **logical coverage** is preserved — no behavior
   that used to be asserted is now unasserted. Coverage by *another* test is
   fine; coverage by *no* test is not.
5. **Final verification.** Run the full suite. Compare new total wall time
   to baseline. If anything regressed, fix before returning.
6. **Report.** Emit the report described at the bottom.

## Step 1: Baseline

The first run is cold (import caches, bytecode, fixture warmup) and will
overstate wall time. Run twice and use the **second** run as the baseline
so you're comparing against a hot baseline.

```bash
uv run pytest -q > /dev/null 2>&1  # warmup, discard
uv run pytest --durations=0 -q 2>&1 | tee /tmp/optimize-tests-baseline.txt
```

Record:
- Total wall time
- Top ~30 durations (you'll cherry-pick from these)
- Whether the long tail is "few outliers" or "many medium tests"

If the suite doesn't pass cleanly at baseline, **stop** — fix the failure
first or surface it to the user. Optimizing on a red suite hides regressions.

## Step 2: Pick Targets

Use judgment. Some heuristics:
- A single test taking >5% of total wall time is always a target
- The top 10 tests by duration are usually worth at least a look
- A cluster of similarly-slow tests in one file often shares a root cause
  (slow fixture, real I/O, missing scope) — fix the cause once

Skip:
- Tests already under ~50ms (the overhead is pytest itself)
- Tests whose slowness is **inherent to what they verify** (e.g. an explicit
  timeout test asserting a 1s deadline) — note in the report, don't touch

## Step 3: Diagnose & Fix — Playbook

For each target, walk the playbook in roughly this order. Stop at the
first applicable fix; don't pile on.

### Pytest collection / fixtures
- **Fixture scope too narrow.** Expensive setup (DB connections, gRPC servers,
  parsed protos, JWKS caches) at `function` scope when the state isn't mutated
  by the test → promote to `module` or `session`.
- **Universal `autouse=True` fixtures** that aren't actually universal → make
  opt-in so unrelated tests don't pay the cost.
- **Parametrize redundancy.** Multiple parametrize cases that exercise the
  same code path with different surface data → keep one representative,
  drop the rest. (Real edge cases — boundaries, error paths — stay.)

### `time.sleep` and polling
- `time.sleep(N)` in a test → replace with an `asyncio.Event`, condition var,
  awaited future, or mock the clock (`pytest.MonkeyPatch.setattr` on
  `time.time` / `datetime.now`).
- `asyncio.sleep(N)` polling loops waiting for state → use an `Event` or
  `wait_for` on a future.
- Tests asserting "after T seconds, X happens" → freeze time and advance it
  manually, don't actually wait.

### I/O and external services
- Real network calls when the test isn't an integration test → fake the
  client (`aiohttp` test client, gRPC in-process server, in-memory store).
- Real disk I/O on shared paths → use `tmp_path` (fast, ephemeral). Avoid
  `tmp_path` when the test is genuinely about the filesystem semantics.
- Subprocess / Docker spin-ups whose surface is already covered by direct
  unit calls → replace with the direct call. (Keep one end-to-end if it's
  the *only* coverage of the integration.)
- Repeated expensive setup across a test class → session-scoped fixture
  with explicit reset between tests if needed.

### Async
- Manual event-loop construction per test → let `pytest-asyncio` (we use
  `asyncio_mode = "auto"`) handle it.
- `asyncio.gather` of N independent coroutines done sequentially in a loop
  → gather them.
- **Do NOT promote async fixtures that own a running task (uvicorn / aiohttp
  server, background `asyncio.create_task`, gRPC `aio.server`) to module or
  session scope with `loop_scope="module"`.** The server's accept/handler
  task is bound to the loop the fixture started on, but each `async def`
  test runs via its own `asyncio.Runner.run()` on a *different* loop. The
  server's loop only ticks during fixture setup/teardown — between tests it
  is dormant, so HTTP/gRPC requests from the test connect but the response
  coroutine never runs. Symptom: **flaky, indefinite hangs** with uvicorn
  logging `ASGI callable returned without completing response`; the first
  test in the module may pass, later ones hang. Module-scoping is only safe
  for fixtures that yield **pure values** (parsed config, JWKS bytes,
  pre-built clients without background tasks). For running servers, keep
  function scope, or run the server in a separate thread with its own loop.

### Heavy machinery in scope of unit tests
- `subprocess.run` invoking the CLI under test → call the function directly
  with the same args.
- Loading and re-parsing large fixtures (JSON files, protos, YAML) per test
  → load once at session scope, treat as immutable.
- Importing the world inside the test function → move imports to module
  scope (also helps collection time).

### Redundancy
- Tests fully subsumed by a stronger test (same path, weaker assertions)
  → delete the weaker one. Note in the coverage-review prompt that this
  was intentional.
- Two tests differing only in unrelated setup that produces the same
  branch coverage → merge or delete one.

### Anti-patterns — flag, do NOT auto-fix
Surface to the user if the only path to a faster test is one of these:
- Removing the **only** end-to-end test of a flow
- Mocking the boundary the test was specifically written to verify
- Dropping parametrize axes that cover real edge cases (boundary values,
  error paths, sovereign-cloud branches)
- Replacing an integration test with a mock-heavy unit test that no longer
  catches the integration drift it was written to catch

## Step 4: Coverage Review (parallel agents)

After all fixes are applied, for each modified test file, spawn a
**code-reviewer agent in parallel** with a prompt of this shape:

> Compare the test file `<path>` between `git show main:<path>` and the
> current working tree. The intent of the change is to make tests faster
> without reducing logical coverage. For each removal, scope change, or
> mock substitution: is the behavior still verified somewhere — either by
> a remaining assertion in this file, or by another test in the suite? If
> a behavior is no longer covered by any test, flag it. Coverage by another
> test is fine; coverage by nothing is not. Report findings only — do not
> edit.

Consolidate findings. For each "no longer covered" claim:
- Verify it (grep for the behavior elsewhere).
- If genuine → restore the assertion (in the cheapest form available) or
  revert the specific change.
- If false positive → note in the report's `coverage-review` section.

## Step 5: Final Verification

```bash
uv run pytest -q 2>&1 | tee /tmp/optimize-tests-final.txt
uv run pytest --durations=20 -q  # for the report
make typecheck  # if any non-test code was touched
make lint
```

All must pass. Compare total wall time to baseline.

`make test` already enforces `--timeout=10 --timeout-method=thread`, so a
hang introduced by the optimization (most likely from a fixture-scope
promotion — see the async playbook above) will fail loudly with the test
ID instead of appearing to "still be running." If you ran pytest directly
instead of `make test`, add the same flags. Re-run **at least twice** if
any fixture-scope change was applied, since order-dependent bugs may only
trip on certain runs.

## Guardrails

- **Logical coverage is load-bearing.** "It still passes" is not enough —
  a test that passes because its assertion was deleted is worse than a
  slow test.
- **Behavior preserved in production code.** If you touch non-test code
  (e.g. introducing a clock seam), it's a behavior-neutral refactor only.
  Surface bugs, don't silently fix.
- **One concern per commit.** Fixture-scope changes, mock substitutions,
  and dead-test deletion are separable — keep them in distinct commits if
  you're committing along the way.
- **Don't optimize what you can't measure.** If a "fix" doesn't move the
  per-test duration measurably, revert it. Cleverness without speedup is
  noise.
- **Skip flakes.** If a target test is timing-flaky, surface it and move
  on — fixing flakes is a different job.
- **Wall-clock budget.** Cap the pass at ~30 minutes of model work. If the
  remaining targets won't fit, stop and report what's left.

## Report Format

```
optimize-tests report
baseline total: <Xs>
final total:    <Ys>  (-Zs, -P%)

targets fixed:
  services/echo/tests/test_handler.py::test_retry_backoff
    before: 4.2s   after: 0.05s   fix: replaced asyncio.sleep polling with Event
  libs/monet-grpc/tests/test_outbound.py::test_channel_setup  (file-level)
    before: 6.1s avg/test  after: 0.3s  fix: promoted JWKS fixture to session scope

deletions:
  libs/monet-config/tests/test_env.py::test_kv_uri_parses_basic
    rationale: subsumed by test_kv_uri_parses_with_version (stronger assertions)

flagged, not fixed:
  services/memory/tests/test_grpc_e2e.py::test_full_remember_recall (3.8s)
    rationale: only end-to-end test of the Remember→Recall round trip

coverage-review:
  3 reviewer agents, 0 genuine coverage gaps
  1 false positive: test_kv_uri_parses_basic — `kv://` parsing still covered by test_kv_uri_parses_with_version

verification:
  pytest         PASS   (<Ys>)
  typecheck      PASS
  lint           PASS
```
