#!/usr/bin/env bash
# FLUX + tonight's lessons. SDXL hit a coherence cap (cursed anatomy); FLUX fixed
# coherence before but used the OLD terse captions on the looser 257 set. This run
# gives FLUX the two things that worked on SDXL tonight: STRUCTURED captions
# (bufo, {expr}, {prop}, {pose}) for prop control + the tight soul-consistent /
# neutral-downsampled v3 data for identity. Resumable.
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

mkdir -p /mnt/ray/bufo-runs /mnt/ray/hf
META_SRC="${META_SRC:-bufo/data_canon_v3/metadata.jsonl}"
DATA="${DATA:-/mnt/ray/bufo-data-canon-v3}"
RUN="${RUN:-/mnt/ray/bufo-runs/flux-soul}"
TRAIN_CONFIG="${TRAIN_CONFIG:-bufo/configs/lora-flux.yaml}"
MAX_STEPS="${MAX_STEPS:-600}"
mkdir -p "$DATA"
ln -sfn /mnt/ray/bufo-data/images "$DATA/images"
cp "$META_SRC" "$DATA/metadata.jsonl"
N=$(wc -l < "$DATA/metadata.jsonl" | tr -d ' '); log "structured-caption records: $N (from $META_SRC)"
export HF_HOME=/mnt/ray/hf
[ -f /mnt/ray/hf/token ] && export HF_TOKEN="$(cat /mnt/ray/hf/token)" && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
python -c "import diffusers,torch; print('diffusers',diffusers.__version__,'torch',torch.__version__)" || true
df -h /mnt/ray | tail -1 | awk '{print "[disk] /mnt/ray free="$4}'

latest_n() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }
resume_arg() { local n; n=$(latest_n "$1"); [ -n "$n" ] && echo "--resume $1/checkpoint-$n"; }

if [ -z "${SKIP_TRAIN:-}" ]; then
  log "=== FLUX train ($TRAIN_CONFIG, $MAX_STEPS steps, structured captions, v3 data) ==="
  python -m bufo.train_lora --config "$TRAIN_CONFIG" \
    --data-dir "$DATA" --run-dir "$RUN" --max-steps "$MAX_STEPS" $(resume_arg "$RUN") \
    && log "train OK" || log "train FAILED rc=$?"
fi

# FLUX is guidance-distilled (negative prompts ~no-op) and doesn't tile, so eval with
# the plain structured prompts at native 1024, full LoRA. ipp=2 keeps it ~under an hour.
# EVAL_CONFIG must match the training caption schema (v4 = green/soft anchored).
EVAL_CONFIG="${EVAL_CONFIG:-bufo/configs/eval-bufo-soul.yaml}"
CKPTS="${CKPTS:-600}"
for CK in $CKPTS; do
  [ -d "$RUN/checkpoint-$CK" ] || continue
  [ -d "$RUN/eval-$CK" ] && continue
  log "=== FLUX eval ckpt $CK @1024 (ipp=2) config=$EVAL_CONFIG ==="
  python -m bufo.eval --lora "$RUN/checkpoint-$CK" --base-kind flux \
    --base-model black-forest-labs/FLUX.1-dev \
    --eval-config "$EVAL_CONFIG" --data-dir "$DATA" \
    --lora-config "$TRAIN_CONFIG" \
    --images-per-prompt 2 --resolution 1024 --out "$RUN/eval-$CK" \
    && log "eval $CK OK -> $RUN/eval-$CK" || log "eval $CK FAILED rc=$?"
done
log "FLUX SOUL DONE -> $RUN/eval-*"
