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

## VPN access (required)

The cluster URLs are reachable **only over the Azure VPN `MSFT-AzVPN-Manual`**
(`com.microsoft.AzureVpnMac`) — *not* Tailscale. When it drops, every cluster URL
times out (**HTTP 000**); jobs keep running server-side (outputs still land on
`/mnt/ray`), only your ability to *query* them is lost.

- **Liveness probe:** `curl -s -m 10 -o /dev/null -w '%{http_code}' "$ADDR/api/version"`
  → `200` = up, `000`/timeout = VPN down.
- **Check state:** `scutil --nc list` (look for `"MSFT-AzVPN-Manual"  (Connected)`) or
  `scutil --nc status "MSFT-AzVPN-Manual"`.
- **Reconnect (fixable):** `scutil --nc start "MSFT-AzVPN-Manual"`, then re-probe until
  `200`. It may require interactive auth/MFA via the Azure VPN app — if `scutil` start
  doesn't restore it, ask the user to reconnect.
- **A VPN drop never loses work** — re-run the status poll against the same job IDs once `200` returns.

## Node environment (the baked image)

Pods run a **prebuilt image** `ray-app:2.40.0-gpu` — owned by Cameron's **raygun** repo
(KubeRay-on-AKS infra; built via Terraform `az acr build` → ACR, deployed as the worker
pod image). We don't maintain it; treat it as fixed. Baked in: **Python 3.9**,
**`torch 2.4.1+cu121`** (CUDA 12.1),
**`numpy<2`** (pinned), `mlflow-skinny`, Ray 2.40.0. `pip` is present; **`uv` is NOT**.
Pods have PyPI egress, so *per-job* deps go through Ray `runtime_env` pip. GPU nodes
**autoscale from zero**, so the first GPU job sits `PENDING` a few minutes.

Implications:
- **Don't list `torch` or `numpy`** in `runtime_env` pip — they're baked, and numpy is
  pinned `<2` (pulling numpy≥2 risks an ABI break with the baked torch).
- **Leave the rest unpinned.** The repo's py3.12 pins fail on py3.9 (e.g.
  `transformers==5.11` needs ≥3.10); unpinned, pip resolves py3.9 wheels — verified
  `diffusers 0.36 / transformers 4.57 / peft 0.17`. Write code robust across versions.
- **Verify py3.9 compat locally before you submit** — `make verify-cluster` runs `vermin`
  against the cluster-bound code (bufo/ + mm-training) and fails on any 3.10+ feature, so
  the silent version breaks surface in seconds locally instead of on a failed remote run.
- Base is **Python 3.9** (repo targets 3.12) → avoid 3.10+ runtime syntax. Write
  version-robust code; `make verify-cluster` is the gate.
- **No `uv`** → don't `uv sync`; pip the third-party deps + put workspace libs on `PYTHONPATH`.
- **MLflow:** if you use it, keep `mlflow-skinny==2.22.0` and `protobuf<4` (Ray-2.40 compat).
- **One GPU only.** Quota is 1×A100 (sweden) / 1×H100 (westus3); request **exactly**
  `--entrypoint-num-gpus 1`. You can't get >1 GPU, and parallel jobs **queue** on the
  single GPU — for two concurrent runs, use both clusters.

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

- **`--entrypoint-num-gpus 1`** — required, and **must be exactly 1** (single-GPU
  quota). Without it the job silently runs CPU-only.
- **`--working-dir .`** uploads the cwd (~100 MiB cap). **Always `excludes`** big/
  regenerable dirs (`runs`, data, `.venv`, `.git`) — or use a `.rayignore` file.
- **uv-workspace local libs:** don't pip-install them — add their `src` dirs to `PYTHONPATH`
  via `env_vars` and run as a module.
- **Data:** the node has none — fetch/prepare it in the entrypoint or read from shared storage.

### bufo example (full SDXL training on the A100)

```bash
ADDR=http://ray-picasso-dash-swc.swedencentral.cloudapp.azure.com:8265
uvx --from "ray[default]==2.40.0" ray job submit --address "$ADDR" \
  --working-dir . --entrypoint-num-gpus 1 \
  --runtime-env-json '{
    "pip":["diffusers","transformers","peft","accelerate","safetensors","pyyaml","tensorboard"],
    "excludes":["runs","bufo/data",".venv",".git","wordle","wordle2","wordle3","docs"],
    "env_vars":{"HF_HUB_DISABLE_TELEMETRY":"1","HF_HOME":"/mnt/ray/hf",
      "PYTHONPATH":"libs/mm-tokenizers/src:libs/mm-model/src:libs/mm-training/src:libs/mm-grpo/src:libs/mm-wordle/src:libs/mm-viz/src"}}' \
  -- bash -lc "mkdir -p /mnt/ray/bufo-runs /mnt/ray/hf /mnt/ray/bufo-data; \
       ln -sfn /mnt/ray/bufo-runs runs; ln -sfn /mnt/ray/bufo-data bufo/data; \
       python -m bufo.prepare --config bufo/configs/lora-sdxl.yaml && \
       python -m bufo.train_lora --config bufo/configs/lora-sdxl.yaml"
```

Notes from real runs: `tensorboard` **must** be in the pip list (importing
`mm_training` loads `SummaryWriter`). Symlink `runs/` and `HF_HOME` onto **`/mnt/ray`**
(100 GB Azure Files NFS) so checkpoints + the ~7 GB SDXL cache survive node teardown
and are reused. Verified: a 600-step SDXL+TE-LoRA run is **~27 min** and a 24-prompt
eval **~3-5 min** on the A100 (vs hours / ~40 min on MPS).

## Custom image (rarely worth it — it's raygun's, not ours)

The cluster image is **owned by Cameron's `raygun` repo** (KubeRay-on-AKS Terraform), not
this repo: `docker/requirements.txt` + `docker/Dockerfile` → `az acr build` → ACR
(`rayacrswed110c2c4` for the swedencentral A100 site, RG `rg-ray-aks-swedencentral`),
consumed as the worker **pod image**. There is **no per-job container `runtime_env`**.
ACR auth is Entra/RBAC with admin disabled; the only standing grant is **`AcrPull`** → the
AKS kubelet identity, and **pushing** needs **`AcrPush`** on your Entra identity
(`az role assignment create --assignee <upn> --role AcrPush --scope <acr-id>`). Base is
**py3.9 by design** (its MLflow-2.x / click-8.1.7 pins); bumping to py3.12 ripples through
shared infra. So baking our deps = a PR to raygun (`requirements.txt`) + an AcrPush grant +
Cameron's `terraform apply`.

**We don't do this.** The py3.9 image + per-job `runtime_env` pip already runs our SDXL
train + eval, and `make verify-cluster` keeps our code py3.9-clean — so the drift is gone
without a custom image. Reach for one only if per-job pip latency ever justifies the
cross-repo coordination.

## Monitoring

- **CLI:** `uvx --from "ray[default]==2.40.0" ray job logs <id> --address "$ADDR" --follow`;
  also `ray job list` / `ray job status <id>` / `ray job stop <id>`.
- **REST (no client):** `GET $ADDR/api/jobs/` (list+status), `GET $ADDR/api/jobs/{id}/logs`,
  `GET $ADDR/api/cluster_status` (nodes + GPU autoscale state).
- **Dashboard:** open `$ADDR` in a browser. **MLflow:** run metrics. **Grafana:** GPU/node health.

## Gotchas

- **VPN required** — off-VPN: HTTP 000 / timeouts.
- **First GPU job is slow to *start*** — `PENDING` while the A100 autoscales (minutes); later jobs reuse the warm node.
- **working_dir size cap (~100 MiB)** — `excludes` `runs/`, data, `.venv`, `.git` (or `.rayignore`).
- **Single-GPU quota** — one GPU per cluster; parallel sweeps **queue**. For two
  concurrent runs, use both clusters (A100 sweden + H100 westus3).
- **Don't override baked `torch`/`numpy`** (numpy pinned `<2`); needing different pins → custom image.
- **`tensorboard` in the pip list** if you import `mm_training`.
- **Persist to `/mnt/ray`** — node-local `runs/` vanishes on scale-down; symlink it (+ `HF_HOME`) to the NFS.
- **Python 3.9 base** — avoid 3.10+ runtime syntax: no `zip(..., strict=)` (3.10),
  `datetime.UTC` (3.11 → use `timezone.utc`), or `int.bit_count()` (3.10 → `bin(x).count("1")`).
  Or supply a custom py3.12 image.
- **Don't rely on symlinks for NFS data** — point `prepare`/`eval`/`train`/`recaption` at
  the absolute path via `--data-dir /mnt/ray/bufo-data` (a `bufo/data` symlink silently
  wrote ephemeral on the node).
- **Incremental progress (repo rule)** — submit jobs that checkpoint/resume (`--resume`) and stream
  flushed progress, so a preemption/timeout never throws the run away.
