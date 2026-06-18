#!/usr/bin/env bash
# Full-res eval of the SDXL@1024 LoRA with the GOOD denoiser (DPM++ 2M Karras),
# replacing SDXL's default Euler. Generates checkpoint-1500 + 1000 at 1024, ipp 4.
# Run with the diffusers-0.31 env (same as training).
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
DATA=/mnt/ray/bufo-data-canon
mkdir -p "$DATA"; ln -sfn /mnt/ray/bufo-data/images "$DATA/images"
cp bufo/data_canon/metadata.jsonl "$DATA/metadata.jsonl"
ln -sfn /mnt/ray/bufo-runs runs
export HF_HOME=/mnt/ray/hf
python -c "import diffusers,torch;print('diffusers',diffusers.__version__,'torch',torch.__version__)" || true

SDX=/mnt/ray/bufo-runs/sdxl-1024
for CK in 1500 1000; do
  if [ -d "$SDX/checkpoint-$CK" ] && [ ! -d "$SDX/eval-$CK-dpmpp" ]; then
    log "=== DPM++ 2M Karras eval ckpt $CK @1024 ipp4 ==="
    python -m bufo.eval --lora "$SDX/checkpoint-$CK" --base-kind sdxl \
      --base-model stabilityai/stable-diffusion-xl-base-1.0 \
      --eval-config bufo/configs/eval-bufo-v2.yaml --data-dir "$DATA" \
      --resolution 1024 --images-per-prompt 4 --sampler dpmpp_2m_karras \
      --out "$SDX/eval-$CK-dpmpp" \
      && log "eval $CK dpmpp OK -> $SDX/eval-$CK-dpmpp" || log "eval $CK dpmpp FAILED rc=$?"
  fi
done
log "DPMPP EVAL DONE"
