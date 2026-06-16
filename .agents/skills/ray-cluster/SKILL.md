---
name: ray-cluster
description: >
  Submit and monitor GPU jobs on the team's "picasso" Ray clusters (A100 in
  swedencentral, H100 in westus3) via the Ray Jobs API, with MLflow + Grafana.
  Use when the user wants to run training/eval/long jobs on remote GPUs, move a
  slow local run to the cluster, or check a cluster job. Triggers on: 'ray
  cluster', 'ray job submit', 'submit a job', 'run on A100', 'run on H100', 'use
  the cluster', 'picasso', 'remote GPU', 'cloud GPU', 'run this on the cluster'.
---

# Ray GPU clusters (picasso)

Two autoscaling Ray clusters. Jobs go to the Ray **Jobs API** (`:8265`); metrics →
**MLflow**; GPU/node monitoring → **Grafana**. **VPN / corp network is required** —
off-VPN every URL times out (HTTP 000).

## Clusters

| Cluster | GPU | Use | Ray Jobs API (`:8265`) |
|---|---|---|---|
| **swedencentral** | A100 80GB | **experiments (default)** | http://ray-picasso-dash-swc.swedencentral.cloudapp.azure.com:8265 |
| **westus3** | H100 | production — coordinate first | http://ray-picasso-dash.westus3.cloudapp.azure.com:8265 |

- MLflow: `http://ray-picasso-mlflow-swc.swedencentral.cloudapp.azure.com` (westus3: drop `-swc`)
- Grafana: `http://ray-picasso-grafana-swc.swedencentral.cloudapp.azure.com`

Default to **swedencentral (A100)** for experiments; leave **westus3 (production)** alone unless told.

## Node environment (verified by probe)

Head + GPU workers: **conda Python 3.9**, `pip` (no `uv`), **`torch 2.4.1+cu121`**
preinstalled, **A100 80GB / CUDA 12.1**. Project deps (`diffusers`, `transformers`,
`peft`, `accelerate`, `safetensors`, …) are **NOT** present — add them per job via
`runtime_env` pip. GPU nodes **autoscale from zero**, so the first GPU job sits
`PENDING` for a few minutes while Azure provisions the node.

Implications:
- **Keep the base torch** (`+cu121`) — only pip-add the missing libs (never list `torch`).
- **Leave deps unpinned.** The repo's py3.12 pins **fail to install on py3.9** (e.g. `transformers==5.11`
  needs ≥3.10). Unpinned, pip resolves working py3.9 wheels — verified: `diffusers 0.36`,
  `transformers 4.57`, `peft 0.17`, `accelerate 1.10`. For exact-version reproduction, ship a py3.12
  **container image** instead. (Write code that's robust across these versions.)
- Base is **Python 3.9** (repo targets 3.12) → avoid 3.10+ runtime syntax, or ship a container.
- **No `uv` on nodes** → don't `uv sync`; pip-install third-party deps and put the uv-workspace
  libs on `PYTHONPATH`.

## Submitting a job

Drive it with the Ray client via `uvx` (no local install), matching the cluster's Ray **2.40.0**:

```bash
ADDR=http://ray-picasso-dash-swc.swedencentral.cloudapp.azure.com:8265
uvx --from "ray[default]==2.40.0" ray job submit \
  --address "$ADDR" \
  --working-dir . \
  --entrypoint-num-gpus 1 \
  --runtime-env-json '{"pip":["diffusers","transformers","peft","accelerate","safetensors","pyyaml"],
                       "excludes":["runs","**/data",".venv",".git"],
                       "env_vars":{"HF_HUB_DISABLE_TELEMETRY":"1"}}' \
  -- python your_script.py
```

- **`--entrypoint-num-gpus 1`** requests a GPU (triggers A100 autoscale).
- **`--working-dir .`** uploads the cwd. **Always `excludes`** big/regenerable dirs
  (`runs`, data, `.venv`, `.git`) or the upload exceeds Ray's size cap and fails.
- **uv-workspace local libs:** don't pip-install them — add their `src` dirs to `PYTHONPATH`
  via `env_vars` and run as a module.
- **Data:** the node has none — fetch/prepare it in the entrypoint or read from shared storage.

### bufo example (full SDXL training on the A100)

```bash
ADDR=http://ray-picasso-dash-swc.swedencentral.cloudapp.azure.com:8265
uvx --from "ray[default]==2.40.0" ray job submit --address "$ADDR" \
  --working-dir . --entrypoint-num-gpus 1 \
  --runtime-env-json '{
    "pip":["diffusers","transformers","peft","accelerate","safetensors","pyyaml"],
    "excludes":["runs","bufo/data",".venv",".git"],
    "env_vars":{"HF_HUB_DISABLE_TELEMETRY":"1",
      "PYTHONPATH":"libs/mm-tokenizers/src:libs/mm-model/src:libs/mm-training/src:libs/mm-grpo/src:libs/mm-wordle/src:libs/mm-viz/src"}}' \
  -- bash -lc "python -m bufo.prepare && python -m bufo.train_lora --config bufo/configs/lora-sdxl.yaml"
```

A100 turns the ~3 h MPS SDXL run into ~15–30 min, and the ~40 min eval into ~2–3 min.

## Monitoring

- **CLI:** `uvx --from "ray[default]==2.40.0" ray job logs <id> --address "$ADDR" --follow`;
  also `ray job list` / `ray job status <id>` / `ray job stop <id>`.
- **REST (no client):** `GET $ADDR/api/jobs/` (list+status), `GET $ADDR/api/jobs/{id}/logs`,
  `GET $ADDR/api/cluster_status` (nodes + GPU autoscale state).
- **Dashboard:** open `$ADDR` in a browser. **MLflow:** run metrics. **Grafana:** GPU/node health.

## Gotchas

- **VPN required** — off-VPN: HTTP 000 / timeouts.
- **First GPU job is slow to *start*** — `PENDING` while the A100 autoscales (minutes); later jobs reuse the warm node.
- **working_dir size cap** — always `excludes` `runs/`, data, `.venv`, `.git`.
- **Python 3.9 base** — pin deps; avoid 3.10+ runtime syntax (or supply a container image).
- **Incremental progress (repo rule)** — submit jobs that checkpoint/resume (`--resume`) and stream
  flushed progress, so a preemption/timeout never throws the run away.
