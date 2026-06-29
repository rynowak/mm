#!/usr/bin/env bash
# Step-sweep for the SD3.5-Medium full fine-tune: train 0->2000 in 500-step segments,
# generating the eval grid after each so we can compare 500/1000/1500/2000 BY EYE.
#
# Segmented + resume (not 4 fresh runs) = ~2000 steps total, not 5000. checkpoints_total_limit
# 1 keeps only the latest full checkpoint (~14G) so disk stays bounded; the final pipeline
# save is patched out (we eval straight from checkpoint-N/transformer). Fully resumable:
# a segment whose grid already exists is skipped.
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
export HF_HOME=/mnt/ray/hf
[ -f /mnt/ray/hf/token ] && export HF_TOKEN="$(cat /mnt/ray/hf/token)" && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_SRC="${DATA_SRC:-/mnt/ray/bufo-data-sd35full}"
RUN="${RUN:-/mnt/ray/bufo-runs/sd35-sweep}"
MODEL="${MODEL:-stabilityai/stable-diffusion-3.5-medium}"
RES="${RES:-512}"
DIFFUSERS_VER="${DIFFUSERS_VER:-0.32.2}"
STEPS_LIST="${STEPS_LIST:-500 1000 1500 2000}"
mkdir -p "$RUN"
df -h /mnt/ray | tail -1 | awk '{print "[disk] free="$4}'

# Stage imagefolder (images + metadata.jsonl with basename file_name).
STAGE="/mnt/ray/sd35-stage-$(basename "$RUN")"; rm -rf "$STAGE" "$HF_HOME/datasets"; mkdir -p "$STAGE"
cp "$DATA_SRC"/images/* "$STAGE"/ 2>/dev/null
python - "$DATA_SRC/metadata.jsonl" "$STAGE/metadata.jsonl" <<'PY'
import json, os, sys
src, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src)]
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps({"file_name": os.path.basename(r["file_name"]), "caption": r["caption"]}) + "\n")
print("staged", len(rows), "rows")
PY

# Fetch + patch the official SD3 dreambooth example: (1) load our local imagefolder with
# captions, (2) skip the final full-pipeline save (it would dump ~15G per segment).
SCRIPT=/tmp/train_dreambooth_sd3.py
python - "$DIFFUSERS_VER" "$SCRIPT" <<'PY'
import sys, urllib.request
ver, out = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/huggingface/diffusers/v{ver}/examples/dreambooth/train_dreambooth_sd3.py", out)
print("fetched")
PY
python - "$SCRIPT" <<'PY'
import re, sys
p = sys.argv[1]; src = open(p).read()
helper = '''
def _bufo_load_local(d):
    import os, json
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
    rows = [json.loads(l) for l in open(os.path.join(d, "metadata.jsonl"))]
    ds = Dataset.from_dict(
        {"image": [os.path.join(d, r["file_name"]) for r in rows],
         "caption": [r["caption"] for r in rows]},
        features=Features({"image": HFImage(), "caption": Value("string")}))
    return DatasetDict({"train": ds})
'''
assert "import torch\n" in src
src = src.replace("import torch\n", "import torch\n" + helper, 1)
src2 = re.sub(r"dataset = load_dataset\([^)]*\)", "dataset = _bufo_load_local(args.dataset_name)", src, count=1)
assert src2 != src, "load_dataset patch missed"
# Exit right before the post-training block (save pipeline -> reload from output_dir ->
# validation -> push). We don't need any of it: checkpoints are saved during the loop and we
# eval from checkpoint-N/transformer. Skipping just the save broke the reload (no model_index
# .json), so cut the whole block. Also keeps disk minimal (no ~15G pipeline dump per segment).
anchor = "    # Save the lora layers\n    accelerator.wait_for_everyone()\n"
assert anchor in src2, "final-block anchor missing"
src3 = src2.replace(anchor, anchor + "    import sys as _bufo_sys; _bufo_sys.exit(0)  # bufo: skip post-train\n", 1)
open(p, "w").write(src3)
print("patched (load_dataset + early-exit:", src3 != src2, ")")
PY

PREV=""
for N in $STEPS_LIST; do
  EVAL_DIR="$RUN/sweep-eval-$N"
  if [ -f "$EVAL_DIR/contact_sheet.png" ]; then log "segment $N already evaluated — skip"; PREV="$N"; continue; fi
  RESUME=""
  [ -d "$RUN/checkpoint-$PREV" ] && RESUME="--resume_from_checkpoint latest"
  if [ ! -d "$RUN/checkpoint-$N" ]; then
    log "=== train -> $N steps @ $RES ${RESUME:+(resume from $PREV)} ==="
    accelerate launch --num_processes 1 --mixed_precision bf16 "$SCRIPT" \
      --pretrained_model_name_or_path "$MODEL" \
      --dataset_name "$STAGE" --image_column image --caption_column caption \
      --instance_prompt "olive green bufo, adult, soft-shaded cartoon sticker" \
      --output_dir "$RUN" \
      --resolution "$RES" --train_batch_size 1 --gradient_accumulation_steps 4 \
      --use_8bit_adam --gradient_checkpointing --mixed_precision bf16 \
      --learning_rate 1e-5 --lr_scheduler constant --lr_warmup_steps 0 \
      --max_train_steps "$N" --checkpointing_steps 500 --checkpoints_total_limit 1 \
      --seed 42 $RESUME \
      && log "train $N OK" || { log "train $N FAILED rc=$?"; exit 1; }
  fi
  log "=== eval grid for checkpoint-$N ==="
  FT="$RUN/checkpoint-$N/transformer" OUT="$EVAL_DIR" RES=1024 IPP=2 \
    python bufo/cluster/sd35_gen_grid.py && log "eval $N OK" || log "eval $N FAILED rc=$?"
  df -h /mnt/ray | tail -1 | awk '{print "[disk] free="$4}'
  PREV="$N"
done
log "SWEEP DONE — grids in $RUN/sweep-eval-{$STEPS_LIST}"
