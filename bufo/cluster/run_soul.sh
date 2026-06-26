#!/usr/bin/env bash
# Soul retrain: SDXL @1024 rank-96 LoRA on the soul-consistent 242 with structured
# captions. Then eval at 1024 with DPM++ 2M Karras on structured prompts. Resumable.
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

mkdir -p /mnt/ray/bufo-runs /mnt/ray/hf
# Overridable so one script serves both the base soul run and variant experiments
# (e.g. the neutral-downsampled v3 rebalance): set META_SRC/DATA/RUN/TRAIN_CONFIG.
META_SRC="${META_SRC:-bufo/data_canon_v2/metadata.jsonl}"
DATA="${DATA:-/mnt/ray/bufo-data-canon-v2}"
RUN="${RUN:-/mnt/ray/bufo-runs/sdxl-soul}"
TRAIN_CONFIG="${TRAIN_CONFIG:-bufo/configs/lora-sdxl-soul.yaml}"
mkdir -p "$DATA"
ln -sfn /mnt/ray/bufo-data/images "$DATA/images"
cp "$META_SRC" "$DATA/metadata.jsonl"
N=$(wc -l < "$DATA/metadata.jsonl" | tr -d ' '); log "structured-caption records: $N (from $META_SRC)"
export HF_HOME=/mnt/ray/hf
python -c "import diffusers,torch; print('diffusers',diffusers.__version__,'torch',torch.__version__)" || true

latest_n() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }
resume_arg() { local n; n=$(latest_n "$1"); [ -n "$n" ] && echo "--resume $1/checkpoint-$n"; }

df -h /mnt/ray | tail -1 | awk '{print "[disk] /mnt/ray free="$4" used="$3"/"$2}'

# SKIP_TRAIN=1 → eval-only (judge an existing checkpoint at multiple scales without
# spending GPU on training). Default (unset) → prune superseded ckpts, then resume+train.
if [ -z "${SKIP_TRAIN:-}" ]; then
  # Disk hygiene: drop superseded early checkpoints so the resume can't run the NFS
  # (107GB) dry writing checkpoint-1500/2000 (~3.5GB each). Keep >=1000.
  for OLD in 500; do
    [ -d "$RUN/checkpoint-$OLD" ] && { rm -rf "$RUN/checkpoint-$OLD"; log "pruned superseded checkpoint-$OLD"; }
  done
  MS_ARG=""; [ -n "${MAX_STEPS:-}" ] && MS_ARG="--max-steps $MAX_STEPS"
  log "=== train ($TRAIN_CONFIG, structured captions) $MS_ARG ==="
  python -m bufo.train_lora --config "$TRAIN_CONFIG" \
    --data-dir "$DATA" --run-dir "$RUN" $MS_ARG $(resume_arg "$RUN") \
    && log "train OK" || log "train FAILED rc=$?"
else
  log "=== SKIP_TRAIN set: eval-only ==="
fi

# Eval at the strength the LoRA will actually be used at: 0.70 = clean read, 0.85 =
# max identity. EVAL_CONFIG/EVAL_TAG let one script test alternate eval recipes (e.g.
# an anti-tiling negative prompt) into a distinct eval dir without clobbering.
EVAL_CONFIG="${EVAL_CONFIG:-bufo/configs/eval-bufo-soul.yaml}"
EVAL_TAG="${EVAL_TAG:-}"
SCALES="${SCALES:-0.70 0.85}"
# CKPTS limits which checkpoints to eval (default all the usual milestones). Set it to
# avoid wasting GPU re-evaling intermediate checkpoints you don't care about.
CKPTS="${CKPTS:-2000 1500 1000}"
log "eval config=$EVAL_CONFIG tag='${EVAL_TAG}' scales='$SCALES' ckpts='$CKPTS'"
for CK in $CKPTS; do
  [ -d "$RUN/checkpoint-$CK" ] || continue
  for S in $SCALES; do
    TAG=$(echo "$S" | tr -d '.')
    OUT="$RUN/eval-$CK-s$TAG$EVAL_TAG"
    [ -d "$OUT" ] && continue
    log "=== eval ckpt $CK @1024 DPM++ Karras lora-scale $S -> $OUT ==="
    python -m bufo.eval --lora "$RUN/checkpoint-$CK" --base-kind sdxl \
      --base-model stabilityai/stable-diffusion-xl-base-1.0 \
      --eval-config "$EVAL_CONFIG" --data-dir "$DATA" \
      --resolution 1024 --sampler dpmpp_2m_karras --images-per-prompt 3 \
      --lora-scale "$S" --out "$OUT" \
      && log "eval $CK s$TAG$EVAL_TAG OK -> $OUT" || log "eval $CK s$TAG$EVAL_TAG FAILED rc=$?"
  done
done
log "SOUL RUN DONE -> $RUN/eval-*"
