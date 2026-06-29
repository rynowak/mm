#!/usr/bin/env bash
# Full fine-tune SD3.5-Large on bufo, via diffusers' official SD3 DreamBooth (full-FT)
# example. Memory fit on one 85GB A100: 8-bit Adam + gradient checkpointing + bf16 +
# frozen text encoders (embeddings precomputed) => ~60GB. MAX_STEPS=20 = stack/memory
# validation; set higher for the real run.
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
export HF_HOME=/mnt/ray/hf
[ -f /mnt/ray/hf/token ] && export HF_TOKEN="$(cat /mnt/ray/hf/token)" && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# 8B full-FT is right at the 80GB edge — reduce fragmentation (the OOM was by ~46MB).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stage from the cluster-resident copy (the working-dir upload drops images via .gitignore).
DATA_SRC="${DATA_SRC:-/mnt/ray/bufo-data-teacher-v6}"
RUN="${RUN:-/mnt/ray/bufo-runs/sd35-ft}"
MAX_STEPS="${MAX_STEPS:-20}"
RES="${RES:-512}"
DIFFUSERS_VER="${DIFFUSERS_VER:-0.32.2}"
# Large (8B) full-FT needs DeepSpeed CPU offload on one A100; Medium (2.24B) fits on-GPU.
MODEL="${MODEL:-stabilityai/stable-diffusion-3.5-medium}"
mkdir -p "$RUN"
python -c "import diffusers,torch,bitsandbytes; print('diffusers',diffusers.__version__,'torch',torch.__version__,'bnb',bitsandbytes.__version__)" || true
df -h /mnt/ray | tail -1 | awk '{print "[disk] free="$4}'

# Build an imagefolder-compatible dataset dir: images + metadata.jsonl (file_name basename).
# Fresh stage dir per run + clear datasets cache (imagefolder caches by path and can
# serve a stale/corrupt build if the same path held different content before).
STAGE="/mnt/ray/sd35-stage-$(basename "$RUN")"; rm -rf "$STAGE" "$HF_HOME/datasets"; mkdir -p "$STAGE"
cp "$DATA_SRC"/images/* "$STAGE"/ 2>/dev/null
python - "$DATA_SRC/metadata.jsonl" "$STAGE/metadata.jsonl" <<'PY'
import json, os, sys
src, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src)]
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps({"file_name": os.path.basename(r["file_name"]), "caption": r["caption"]}) + "\n")
print("staged", len(rows), "rows ->", out)
PY

SCRIPT=/tmp/train_dreambooth_sd3.py
python - "$DIFFUSERS_VER" "$SCRIPT" <<'PY'
import sys, urllib.request
ver, out = sys.argv[1], sys.argv[2]
url = f"https://raw.githubusercontent.com/huggingface/diffusers/v{ver}/examples/dreambooth/train_dreambooth_sd3.py"
urllib.request.urlretrieve(url, out)
print("fetched", url)
PY
wc -l "$SCRIPT"
# The script calls load_dataset(dataset_name) which loads our metadata.jsonl as plain
# JSON (no 'image' column). Force the imagefolder builder so images load as 'image'.
python - "$SCRIPT" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()
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
new = re.sub(r"dataset = load_dataset\([^)]*\)",
             "dataset = _bufo_load_local(args.dataset_name)", src, count=1)
assert new != src, "patch did not match load_dataset call"
open(p, "w").write(new)
print("patched load_dataset -> explicit local builder")
PY

log "=== full-FT SD3.5-large | $MAX_STEPS steps @ ${RES} ==="
accelerate launch --num_processes 1 --mixed_precision bf16 "$SCRIPT" \
  --pretrained_model_name_or_path "$MODEL" \
  --dataset_name "$STAGE" --image_column image --caption_column caption \
  --instance_prompt "olive green bufo, adult, soft-shaded cartoon sticker" \
  --output_dir "$RUN" \
  --resolution "$RES" --train_batch_size 1 --gradient_accumulation_steps 4 \
  --use_8bit_adam --gradient_checkpointing --mixed_precision bf16 \
  --learning_rate 1e-5 --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps "$MAX_STEPS" --checkpointing_steps "${CKPT_STEPS:-100000}" \
  --checkpoints_total_limit 2 --seed 42 \
  && log "FT run OK" || log "FT run FAILED rc=$?"
