---
name: autoresearch
description: |
  Runs one iteration of the autoresearch loop (Karpathy, 2026): reads the training
  script and research program, proposes a targeted change, trains, evaluates the
  metric, and commits improvements or reverts regressions. Experiment history lives
  in git. Designed for /loop continuous autonomous iteration.

  Use when user asks to "run autoresearch", "iterate on training", "autoresearch loop",
  "improve training", "optimize the model", "experiment on training", or wants
  autonomous ML experiment iteration.
---

# Autoresearch

One iteration: **read → propose → train → evaluate → keep or revert.**

Stateless — reads all history from git. Use with `/loop` for continuous iteration.

`/autoresearch [project_dir]` — default: `tetris/`

## Project Convention

| File | Role |
|------|------|
| `program.md` | Research direction, metric name, constraints |
| `train.py` | Training script — the ONLY file to edit |

`train.py` prints the metric as `<metric_name>: <value>` on its last matching line.

## Status & Visibility

Overwrite `<dir>/autoresearch-status.json` at each phase transition (dashboard
polls this). Write compact JSON with ALL fields shown for that phase.

Append to `<dir>/autoresearch-log.jsonl` after each iteration (one JSON object
per line, never pretty-printed).

## Workflow

### 1. Read State

Status: `{"phase": "reading", "iteration": <N>}`

Read in parallel:
1. `<dir>/program.md`
2. `<dir>/train.py`
3. `git log --oneline --grep="\[autoresearch\]" -20`

Parse best metric from most recent `[autoresearch]` commit. If none exist →
**baseline run** (skip step 2).

### 2. Propose One Change

Status: `{"phase": "proposing", "iteration": <N>, "metric_before": <best>}`

From program.md, current code, and git history:
1. Identify what has been tried vs. not tried
2. Pick ONE targeted change with a 1-2 sentence rationale
3. Edit `train.py`

**Priority:** structural changes (architecture, algorithm, reward) over
micro-optimizations. New directions over variations of failed attempts.

**Baseline run:** Skip — train the code as-is.

### 3. Train

Status: `{"phase": "training", "iteration": <N>, "change": "<desc>", "rationale": "<why>", "metric_before": <best>}`

Run `uv run python <dir>/train.py` and wait (~5 min).

If training crashes → `git checkout <dir>/train.py`, write idle status, end.

### 4. Evaluate

Parse `<metric_name>: <value>` from the last matching stdout line.

Status: `{"phase": "evaluating", "iteration": <N>, "change": "<desc>", "metric": <new>, "metric_before": <best>}`

### 5. Keep or Revert

| Condition | Action |
|-----------|--------|
| Baseline (no prior commits) | `git add` + commit `"[autoresearch] <metric>: <val> (baseline)"` |
| Improved (new > best) | `git add` + commit `"[autoresearch] <metric>: <val> (+<delta>) — <desc>"` |
| Not improved | `git checkout <dir>/train.py` |

### 6. Log + Report

Append to `<dir>/autoresearch-log.jsonl`:
```json
{"iteration": <N>, "timestamp": "<ISO>", "change": "<desc>", "rationale": "<why>", "metric_name": "<name>", "metric": <val>, "metric_before": <prev>, "delta": <diff>, "result": "BASELINE|KEPT|REVERTED"}
```

Status: `{"phase": "idle"}`

Print summary:
```
## Autoresearch Iteration <N>
**Change:** <desc>  |  **Result:** KEPT/REVERTED/BASELINE  |  **Metric:** <val> (best: <best>)
**Rationale:** <why>  |  **Next idea:** <what to try>
```

## Constraints

- ONLY edit `train.py` — never environment, program.md, or eval protocol
- NEVER change metric output format or training time budget
- ONE change per iteration — atomic and attributable
- Do not repeat reverted changes (check git log)
- Commits MUST start with `[autoresearch]`
- Crashes → revert and report, do not debug
