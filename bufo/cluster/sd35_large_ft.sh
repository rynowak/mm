#!/usr/bin/env bash
# ATTEMPT: SD3.5-LARGE (8B) full fine-tune on one A100-80GB.
# The 8B full-FT OOMed earlier (~79GB static) with on-GPU 8-bit Adam. The lever tried here:
# patch the optimizer to bitsandbytes PagedAdamW8bit, which keeps optimizer states in pinned
# CPU memory and pages them to the GPU per step — removing ~16GB of GPU optimizer state.
# Combined with gradient checkpointing + bf16 + res 512 this MIGHT fit. If it still OOMs,
# the fallback (sd35_large_lora.sh) trains a high-rank LoRA on the same 8B base instead.
set -u
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
export HF_HOME=/mnt/ray/hf
[ -f /mnt/ray/hf/token ] && export HF_TOKEN="$(cat /mnt/ray/hf/token)" && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_SRC="${DATA_SRC:-/mnt/ray/bufo-data-sd35full}"
RUN="${RUN:-/mnt/ray/bufo-runs/sd35-large-ft}"
MODEL="${MODEL:-stabilityai/stable-diffusion-3.5-large}"
RES="${RES:-512}"
MAX_STEPS="${MAX_STEPS:-1000}"
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

SCRIPT=/tmp/train_dreambooth_sd3_large.py
python - "$DIFFUSERS_VER" "$SCRIPT" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/huggingface/diffusers/v{sys.argv[1]}/examples/dreambooth/train_dreambooth_sd3.py", sys.argv[2])
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
# CPU-paged 8-bit optimizer to shave GPU optimizer state for the 8B model.
s3 = s2.replace("optimizer_class = bnb.optim.AdamW8bit", "optimizer_class = bnb.optim.PagedAdamW8bit")
assert s3 != s2, "paged-optimizer patch missed"
# Exit before the post-training save/reload/validation/push block (we eval from
# checkpoint-N/transformer); the reload does from_pretrained(output_dir) and the ~30G save
# would not fit disk anyway.
anchor = "    # Save the lora layers\n    accelerator.wait_for_everyone()\n"
assert anchor in s3, "final-block anchor missing"
save_exit = (
    anchor
    + "    _bt = unwrap_model(transformer).to(torch.bfloat16)\n"
    + "    _bt.save_pretrained(os.path.join(args.output_dir, 'transformer'))\n"
    + "    import sys as _bufo_sys; _bufo_sys.exit(0)  # bufo: save bf16 transformer only (no fp32 opt checkpoint)\n"
)
s4 = s3.replace(anchor, save_exit, 1)
open(p, "w").write(s4)
print("patched (paged-opt + bf16-save-exit:", s4 != s3, ")")
PY

log "=== SD3.5-LARGE full-FT ATTEMPT | $MAX_STEPS steps @ $RES | PagedAdamW8bit ==="
accelerate launch --num_processes 1 --mixed_precision bf16 "$SCRIPT" \
  --pretrained_model_name_or_path "$MODEL" \
  --dataset_name "$STAGE" --image_column image --caption_column caption \
  --instance_prompt "olive green bufo, adult, soft-shaded cartoon sticker" \
  --output_dir "$RUN" \
  --resolution "$RES" --train_batch_size 1 --gradient_accumulation_steps 4 \
  --use_8bit_adam --gradient_checkpointing --mixed_precision bf16 \
  --learning_rate 1e-5 --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps "$MAX_STEPS" --checkpointing_steps 999999 --checkpoints_total_limit 1 \
  --seed 42 \
  && log "LARGE full-FT OK (bf16 transformer at $RUN/transformer)" || log "LARGE full-FT FAILED rc=$? (likely OOM — fall back to LoRA-on-Large)"
