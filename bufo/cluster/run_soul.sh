#!/usr/bin/env bash
# Soul retrain: SDXL @1024 rank-96 LoRA on the soul-consistent 242 with structured
# captions. Then eval at 1024 with DPM++ 2M Karras on structured prompts. Resumable.
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

mkdir -p /mnt/ray/bufo-runs /mnt/ray/hf
DATA=/mnt/ray/bufo-data-canon-v2
mkdir -p "$DATA"
ln -sfn /mnt/ray/bufo-data/images "$DATA/images"
cp bufo/data_canon_v2/metadata.jsonl "$DATA/metadata.jsonl"
N=$(wc -l < "$DATA/metadata.jsonl" | tr -d ' '); log "structured-caption records: $N"
ln -sfn /mnt/ray/bufo-runs runs
export HF_HOME=/mnt/ray/hf
python -c "import diffusers,torch; print('diffusers',diffusers.__version__,'torch',torch.__version__)" || true

latest_n() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }
resume_arg() { local n; n=$(latest_n "$1"); [ -n "$n" ] && echo "--resume $1/checkpoint-$n"; }

RUN=/mnt/ray/bufo-runs/sdxl-soul
log "=== train (rank96 attn+FF+TE, 2000 steps, structured captions) ==="
python -m bufo.train_lora --config bufo/configs/lora-sdxl-soul.yaml \
  --data-dir "$DATA" --run-dir "$RUN" $(resume_arg "$RUN") \
  && log "train OK" || log "train FAILED rc=$?"

for CK in 2000 1500 1000; do
  if [ -d "$RUN/checkpoint-$CK" ] && [ ! -d "$RUN/eval-$CK" ]; then
    log "=== eval ckpt $CK @1024 DPM++ Karras ==="
    python -m bufo.eval --lora "$RUN/checkpoint-$CK" --base-kind sdxl \
      --base-model stabilityai/stable-diffusion-xl-base-1.0 \
      --eval-config bufo/configs/eval-bufo-soul.yaml --data-dir "$DATA" \
      --resolution 1024 --sampler dpmpp_2m_karras --images-per-prompt 3 \
      --out "$RUN/eval-$CK" \
      && log "eval $CK OK -> $RUN/eval-$CK" || log "eval $CK FAILED rc=$?"
  fi
done
log "SOUL RUN DONE -> $RUN/eval-*"
