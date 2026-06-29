# bufo — SD3.5-Medium full fine-tune (the "that's him" model)

The approach that finally produced a bufo the user recognized as the character
("whoa!" / "that's him!"), after SD1.5/SDXL/FLUX LoRA paths plateaued at
"recognizable but not him" or "baby bufo". History of those attempts:
[`bufo-soul-retrain.md`](bufo-soul-retrain.md).

## What worked: full fine-tune, not LoRA

Every prior attempt was a **LoRA** (adapter on a frozen base). The step-change was a
**full fine-tune** — all transformer weights trainable — of **SD3.5-Medium** (2.5B), which
binds character identity harder than a low-rank adapter can. Base-model scale matters less
than full-FT-vs-LoRA here: Medium full-FT > FLUX-12B LoRA for *this* character.

## Recipe

- **Base:** `stabilityai/stable-diffusion-3.5-medium`
- **Data:** 332 images = 242 curated canon (Claude-vision set) + 90 teacher cells (Gemini
  2.5 Flash image-edits), assembled by `bufo/cluster/build_sd35_dataset.py` →
  `/mnt/ray/bufo-data-sd35full`.
- **Captions:** adult-anchored schema `olive green adult bufo, {desc}, soft-shaded cartoon
  sticker` (the "adult" anchor fights the FLUX "baby bufo" drift).
- **Trainer:** diffusers v0.32.2 `examples/dreambooth/train_dreambooth_sd3.py`, patched to
  load our local imagefolder *with captions* (the stock imagefolder path silently drops
  them). Script: `bufo/cluster/run_sd35_ft.sh`.
- **Hyperparams:** 1000 steps, lr 1e-5 constant, bf16, 8-bit Adam, gradient checkpointing,
  grad-accum 4, resolution 512, seed 42. ~36 min on one A100-80GB (~2.0 s/step steady).
- **Inference/eval:** `bufo/cluster/sd35_eval_judge.py` / `sd35_gen_grid.py` — load base
  SD3.5-medium, swap in the fine-tuned `transformer/`, generate at 1024, 28 steps, guidance
  4.5, negative prompt `deformed, blurry, low quality, extra limbs, teeth, fangs, ...`.

## Durable artifact

- **Private HF repo `rynowak/bufo-sd35-medium-ft`** — the full-precision transformer
  (9.9 GB) + dataset metadata + a recipe card. This is the canonical copy.
- Inference = base pipeline with this `transformer/` swapped in (see the card).

## On judging: the user's eye, not a metric

A DINOv2 embedding-similarity judge (`bufo/cluster/dino_baseline.py`) was built to score
outputs objectively against the approved bufos. **It does not track the user's eye:** it
scored the rejected FLUX-v6 (canon-sim 0.796) tied with the accepted SD3.5 win (0.801) —
within noise — and only cleanly flagged the gross SDXL failure (0.63). Embedding similarity
catches "is it broken," not "is it *him*." DINOv2 is retired as a gate; the user judges.

## SD3.5-Large (8B): full-FT infeasible on 1×A100, LoRA delivered

**Empirically confirmed (2026-06-29):** full-FT of Large (8B) OOMs one A100-80GB even with a
CPU-paged 8-bit optimizer (`PagedAdamW8bit`) + gradient checkpointing + res 512. It loads the
model and computes the first forward (loss printed), then OOMs on the backward at ~79.2/80 GB
— the 8B params + grads + frozen T5-XXL/CLIP encoders fill the card. Making it fit needs
multi-GPU, DeepSpeed ZeRO-3, or offloading the text encoders. `bufo/cluster/sd35_large_ft.sh`
is the attempt (patched to save only a bf16 transformer, no fp32 optimizer checkpoint).

**Fallback delivered:** `bufo/cluster/sd35_large_lora.sh` — rank-64 LoRA on the frozen 8B
base (1000 steps @ 1024, lr 1e-4, ~1h22m), eval via `sd35_gen_grid_lora.py`. This is the 8B
result to compare against Medium full-FT by eye. Thesis: full-FT binds identity harder than
LoRA (FLUX-12B LoRA gave "baby bufo"), so Medium-full-FT may still win — the user judges.
Grid: `/mnt/ray/bufo-runs/sd35-large-lora/eval/`.

## Deliverables

- **Sticker set** (`bufo/cluster/sd35_make_stickers.py`): 55 emoji across emotions, hand
  gestures, props, costumes × 4 candidates, background cut out (rembg), auto-trimmed and
  centered, exported as 1024 masters + 128px Slack PNGs (RGBA) →
  `/mnt/ray/bufo-runs/sd35-medium-ft/stickers/`. The generation framing is bottom-cropped
  "peek" style (matches the approved look); a full-body variant is a prompt change away.
- **Step-sweep** (`bufo/cluster/sd35_sweep.sh`): completed — eval grids at 500/1000/1500/2000
  in `/mnt/ray/bufo-runs/sd35-sweep/sweep-eval-*` to pick the sweet spot by eye. Segmented
  resume keeps disk bounded (~19 GB free throughout); the trainer's post-run save/validate
  block is cut (early `sys.exit`) since we eval from `checkpoint-N/transformer`.
- **Large swing** — see the SD3.5-Large section above: full-FT OOMs 1×A100; LoRA-on-Large
  delivered as the 8B comparison.

## Cluster env gotchas (Ray/picasso)

- GPU jobs **must** pass `--entrypoint-num-gpus 1`; the cluster autoscales GPU nodes on
  demand and the CPU workers' base torch is CPU-only (a no-GPU job silently runs on CPU).
- Do **not** put `torch` in the runtime-env pip list — the GPU node's base image already has
  CUDA torch; pip would pull a CPU build. Use the proven list: diffusers 0.32.2 /
  transformers 4.46.3 / accelerate 1.1.1 / bitsandbytes / peft / datasets / ...
- `rembg` pulls numpy 2.x which breaks Ray's pre-built pyarrow (`numpy.core.multiarray
  failed to import`) and kills the job at supervisor init — pin **`numpy==1.26.4`** in any
  env that includes rembg.
- In Ray `bash -lc` entrypoints, **shell variables can come back empty** — use literal paths
  or drive from `python -c` instead.
