#!/usr/bin/env bash
# Phase 2 on the curated 257 set: find the non-overfit SDXL checkpoint + train
# Flux (which phase 1 couldn't, due to a diffusers/torch enable_gqa mismatch —
# now pinned to diffusers 0.31). Fewer Flux steps (500=~8 epochs) to avoid the
# overfit/tiling seen in SDXL@2000. Resumable from /mnt/ray.
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

mkdir -p /mnt/ray/bufo-runs /mnt/ray/hf
DATA=/mnt/ray/bufo-data-canon
mkdir -p "$DATA"
ln -sfn /mnt/ray/bufo-data/images "$DATA/images"
cp bufo/data_canon/metadata.jsonl "$DATA/metadata.jsonl"
ln -sfn /mnt/ray/bufo-runs runs
export HF_HOME=/mnt/ray/hf
[ -f /mnt/ray/hf/token ] && export HF_TOKEN="$(cat /mnt/ray/hf/token)" && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
python -c "import diffusers,torch; print('diffusers',diffusers.__version__,'torch',torch.__version__)" || true

latest_n() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }
resume_arg() { local n; n=$(latest_n "$1"); [ -n "$n" ] && echo "--resume $1/checkpoint-$n"; }

# --- find the non-overfit SDXL checkpoint (300->2000 already trained) ---
SDX=/mnt/ray/bufo-runs/sdxl-canon
for CK in 600 900 1200; do
  if [ -d "$SDX/checkpoint-$CK" ] && [ ! -d "$SDX/eval-$CK" ]; then
    log "=== SDXL eval ckpt $CK ==="
    python -m bufo.eval --lora "$SDX/checkpoint-$CK" --base-kind sdxl \
      --base-model stabilityai/stable-diffusion-xl-base-1.0 \
      --eval-config bufo/configs/eval-bufo-v2.yaml --data-dir "$DATA" \
      --images-per-prompt 3 --out "$SDX/eval-$CK" \
      && log "SDXL eval $CK OK" || log "SDXL eval $CK FAILED rc=$?"
  fi
done

# --- Flux train (500 steps ~ 8 epochs) + eval 250 & 500 ---
FLX=/mnt/ray/bufo-runs/flux-canon
log "=== Flux train (500 steps) ==="
python -m bufo.train_lora --config bufo/configs/lora-flux.yaml \
  --data-dir "$DATA" --run-dir "$FLX" --max-steps 500 $(resume_arg "$FLX") \
  && log "Flux train OK" || log "Flux train FAILED rc=$?"
for CK in 500 250; do
  if [ -d "$FLX/checkpoint-$CK" ] && [ ! -d "$FLX/eval-$CK" ]; then
    log "=== Flux eval ckpt $CK (ipp=2) ==="
    python -m bufo.eval --lora "$FLX/checkpoint-$CK" --base-kind flux \
      --base-model black-forest-labs/FLUX.1-dev \
      --eval-config bufo/configs/eval-bufo-v2.yaml --data-dir "$DATA" \
      --images-per-prompt 2 --out "$FLX/eval-$CK" \
      && log "Flux eval $CK OK" || log "Flux eval $CK FAILED rc=$?"
  fi
done
log "PHASE2 DONE. SDXL evals in $SDX/eval-*  Flux in $FLX/eval-*"
