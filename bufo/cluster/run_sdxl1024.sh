#!/usr/bin/env bash
# Aggressive SDXL @ native 1024 on the curated 257 — fix the duplication/artifacts
# (train+sample at 1024, not 768) while keeping the bufo identity SDXL already has.
# Harder LoRA (rank 64, attn+FF). Eval at 1024. Resumable from /mnt/ray.
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

mkdir -p /mnt/ray/bufo-runs /mnt/ray/hf
DATA=/mnt/ray/bufo-data-canon
mkdir -p "$DATA"; ln -sfn /mnt/ray/bufo-data/images "$DATA/images"
cp bufo/data_canon/metadata.jsonl "$DATA/metadata.jsonl"
ln -sfn /mnt/ray/bufo-runs runs
export HF_HOME=/mnt/ray/hf
python -c "import diffusers,torch; print('diffusers',diffusers.__version__,'torch',torch.__version__)" || true

latest_n() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }
resume_arg() { local n; n=$(latest_n "$1"); [ -n "$n" ] && echo "--resume $1/checkpoint-$n"; }

SDX=/mnt/ray/bufo-runs/sdxl-1024
log "=== SDXL @1024 train (rank64 attn+FF, 1500 steps) ==="
python -m bufo.train_lora --config bufo/configs/lora-sdxl-1024.yaml \
  --data-dir "$DATA" --run-dir "$SDX" $(resume_arg "$SDX") \
  && log "train OK" || log "train FAILED rc=$?"

for CK in 1500 1000; do
  if [ -d "$SDX/checkpoint-$CK" ] && [ ! -d "$SDX/eval-$CK" ]; then
    log "=== eval ckpt $CK @1024 ==="
    python -m bufo.eval --lora "$SDX/checkpoint-$CK" --base-kind sdxl \
      --base-model stabilityai/stable-diffusion-xl-base-1.0 \
      --eval-config bufo/configs/eval-bufo-v2.yaml --data-dir "$DATA" \
      --images-per-prompt 3 --resolution 1024 --out "$SDX/eval-$CK" \
      && log "eval $CK OK -> $SDX/eval-$CK" || log "eval $CK FAILED rc=$?"
  fi
done
log "SDXL1024 DONE -> $SDX/eval-*"
