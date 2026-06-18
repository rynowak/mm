#!/usr/bin/env bash
# VLM-judge bufo identity on the DPM++ eval images (the real metric, not CLIP).
# Needs a NEWER transformers than the diffusers-0.31 eval env (Qwen2.5-VL), so this
# runs as its own job with its own runtime-env.
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
export HF_HOME=/mnt/ray/hf
python -c "import transformers,torch;print('transformers',transformers.__version__,'torch',torch.__version__)" || true

SDX=/mnt/ray/bufo-runs/sdxl-1024
for CK in 1500 1000; do
  D="$SDX/eval-$CK-dpmpp/images"
  if [ -d "$D" ]; then
    log "=== VLM identity judge ckpt $CK ==="
    python -m bufo.eval_vlm --images-dir "$D" --out "$SDX/eval-$CK-dpmpp/vlm.json" \
      && log "vlm $CK OK" || log "vlm $CK FAILED rc=$?"
  fi
done
log "VLM JUDGE DONE"
