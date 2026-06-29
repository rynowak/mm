#!/usr/bin/env bash
# FALLBACK for P3: high-rank LoRA on SD3.5-LARGE (8B), if the full-FT (sd35_large_ft.sh)
# OOMs. A LoRA keeps the 8B base frozen (no optimizer state for 8B params) so it fits the
# A100 comfortably — this is the way to still get a "bigger model" bufo to compare, even
# though our thesis is that full-FT binds identity harder than LoRA (FLUX LoRA gave "baby
# bufo"). Uses diffusers' official SD3 LoRA dreambooth example.
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
export HF_HOME=/mnt/ray/hf
[ -f /mnt/ray/hf/token ] && export HF_TOKEN="$(cat /mnt/ray/hf/token)" && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_SRC="${DATA_SRC:-/mnt/ray/bufo-data-sd35full}"
RUN="${RUN:-/mnt/ray/bufo-runs/sd35-large-lora}"
MODEL="${MODEL:-stabilityai/stable-diffusion-3.5-large}"
RES="${RES:-1024}"
RANK="${RANK:-64}"
MAX_STEPS="${MAX_STEPS:-1000}"
LR="${LR:-1e-4}"
DIFFUSERS_VER="${DIFFUSERS_VER:-0.32.2}"
mkdir -p "$RUN"
df -h /mnt/ray | tail -1 | awk '{print "[disk] free="$4}'

STAGE="/mnt/ray/sd35-stage-$(basename "$RUN")"; rm -rf "$STAGE" "$HF_HOME/datasets"; mkdir -p "$STAGE"
cp "$DATA_SRC"/images/* "$STAGE"/ 2>/dev/null
python - "$DATA_SRC/metadata.jsonl" "$STAGE/metadata.jsonl" <<'PY'
import json, os, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
with open(sys.argv[2], "w") as f:
    for r in rows:
        f.write(json.dumps({"file_name": os.path.basename(r["file_name"]), "caption": r["caption"]}) + "\n")
print("staged", len(rows))
PY

SCRIPT=/tmp/train_dreambooth_lora_sd3.py
python - "$DIFFUSERS_VER" "$SCRIPT" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/huggingface/diffusers/v{sys.argv[1]}/examples/dreambooth/train_dreambooth_lora_sd3.py", sys.argv[2])
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
s2 = re.sub(r"dataset = load_dataset\([^)]*\)", "dataset = _bufo_load_local(args.dataset_name)", src, count=1)
assert s2 != src, "load_dataset patch missed"
open(p, "w").write(s2)
print("patched load_dataset")
PY

log "=== SD3.5-LARGE LoRA | rank $RANK | $MAX_STEPS steps @ $RES | lr $LR ==="
accelerate launch --num_processes 1 --mixed_precision bf16 "$SCRIPT" \
  --pretrained_model_name_or_path "$MODEL" \
  --dataset_name "$STAGE" --image_column image --caption_column caption \
  --instance_prompt "olive green bufo, adult, soft-shaded cartoon sticker" \
  --output_dir "$RUN" \
  --rank "$RANK" --resolution "$RES" --train_batch_size 1 --gradient_accumulation_steps 4 \
  --use_8bit_adam --gradient_checkpointing --mixed_precision bf16 \
  --learning_rate "$LR" --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps "$MAX_STEPS" --checkpointing_steps 100000 \
  --seed 42 \
  && log "LARGE LoRA OK" || log "LARGE LoRA FAILED rc=$?"
