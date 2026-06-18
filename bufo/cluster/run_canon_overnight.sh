#!/usr/bin/env bash
# Overnight curated-bufo run on one A100: SDXL + Flux LoRA on the 257-image
# canonical set (Claude-vision curated), each followed by a CLIP-metric eval.
# Resumable: re-running picks up from the latest checkpoint on /mnt/ray.
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- stage curated data on this cluster's NFS (images already live there) ---
mkdir -p /mnt/ray/bufo-runs /mnt/ray/hf
DATA=/mnt/ray/bufo-data-canon
mkdir -p "$DATA"
ln -sfn /mnt/ray/bufo-data/images "$DATA/images"
cp bufo/data_canon/metadata.jsonl "$DATA/metadata.jsonl"
NIMG=$(ls "$DATA/images" 2>/dev/null | wc -l | tr -d ' ')
NREC=$(wc -l < "$DATA/metadata.jsonl" | tr -d ' ')
log "curated: $NREC records over $NIMG source images at $DATA"
if [ "$NIMG" -lt 100 ]; then log "FATAL: source images missing on NFS ($NIMG); aborting"; exit 1; fi
ln -sfn /mnt/ray/bufo-runs runs

export HF_HOME=/mnt/ray/hf
if [ -f /mnt/ray/hf/token ]; then
  export HF_TOKEN="$(cat /mnt/ray/hf/token)"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  log "HF token loaded for gated Flux"
else
  log "WARN: /mnt/ray/hf/token missing — Flux (gated) will fail; SDXL still runs"
fi

latest_n() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }
resume_arg() { local n; n=$(latest_n "$1"); [ -n "$n" ] && echo "--resume $1/checkpoint-$n"; }

# ============================ SDXL ============================
SDX=/mnt/ray/bufo-runs/sdxl-canon
log "=== SDXL train (target 2000 steps, ~8 epochs on 257) ==="
python -m bufo.train_lora --config bufo/configs/lora-sdxl.yaml \
  --data-dir "$DATA" --run-dir "$SDX" --max-steps 2000 $(resume_arg "$SDX") \
  && log "SDXL train OK" || log "SDXL train FAILED rc=$?"
SN=$(latest_n "$SDX")
if [ -n "$SN" ]; then
  log "=== SDXL eval (checkpoint-$SN) ==="
  python -m bufo.eval --lora "$SDX/checkpoint-$SN" --base-kind sdxl \
    --base-model stabilityai/stable-diffusion-xl-base-1.0 \
    --eval-config bufo/configs/eval-bufo-v2.yaml --data-dir "$DATA" \
    --images-per-prompt 3 --out "$SDX/eval-$SN" \
    && log "SDXL eval OK -> $SDX/eval-$SN" || log "SDXL eval FAILED rc=$?"
fi

# ============================ Flux ============================
FLX=/mnt/ray/bufo-runs/flux-canon
log "=== Flux train (1000 steps, ~16 epochs on 257) ==="
python -m bufo.train_lora --config bufo/configs/lora-flux.yaml \
  --data-dir "$DATA" --run-dir "$FLX" $(resume_arg "$FLX") \
  && log "Flux train OK" || log "Flux train FAILED rc=$?"
FN=$(latest_n "$FLX")
if [ -n "$FN" ]; then
  log "=== Flux eval (checkpoint-$FN, ipp=2 for speed) ==="
  python -m bufo.eval --lora "$FLX/checkpoint-$FN" --base-kind flux \
    --base-model black-forest-labs/FLUX.1-dev \
    --eval-config bufo/configs/eval-bufo-v2.yaml --data-dir "$DATA" \
    --images-per-prompt 2 --out "$FLX/eval-$FN" \
    && log "Flux eval OK -> $FLX/eval-$FN" || log "Flux eval FAILED rc=$?"
fi

log "ALL DONE. SDXL=$SDX (ckpt $SN)  Flux=$FLX (ckpt $FN)"
