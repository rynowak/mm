# bufo — Stable Diffusion LoRA fine-tuning on the bufo emoji set

A self-contained sample that **fine-tunes a pretrained text-to-image diffusion
model** (Stable Diffusion 1.5) with **LoRA** to generate new [bufo](https://github.com/knobiknows/all-the-bufo)
frog emoji. It complements the `wordle*` samples (small models trained from
scratch) by showing the other common regime: cheaply adapting a large frozen
model with a few million trainable parameters.

## What you learn

- The **latent-diffusion training objective**: encode an image to latents with a
  frozen VAE, add sampled noise at a random timestep, and train the UNet to
  predict that noise (DDPM `epsilon`/`v_prediction`) conditioned on a CLIP text
  embedding.
- **LoRA**: freeze the whole base model, inject low-rank adapters into the UNet
  attention projections (`to_q/to_k/to_v/to_out.0`), and train only those —
  ~1% of the parameters.
- The data plumbing: transparency compositing, square-pad + resize, and
  filename-derived captions with a consistent trigger word (`bufo`).

## Pipeline

```
all-the-bufo PNGs ──prepare──▶ images/ + metadata.jsonl ──train_lora──▶ LoRA ckpt ──sample──▶ new bufos
```

### 1. Prepare the dataset (once)

Downloads ~1.4k PNGs, composites transparency onto white, pads to square, resizes
to 512px, and derives a caption per file (e.g. `cowboy-bufo.png` →
`"a bufo of cowboy, frog emoji sticker, white background"`). The `bigbufo_*`
tiles (slices of one giant bufo) are excluded.

```bash
uv run python -m bufo.prepare                 # full corpus, 512px
uv run python -m bufo.prepare --limit 32      # quick subset
```

Output lands in `bufo/data/` (git-ignored — regenerable).

### 2. Fine-tune

```bash
uv run python -m bufo.train_lora --config bufo/configs/lora-sd15.yaml
# quick run:   add --max-steps 300
# watch loss:  uv run tensorboard --logdir runs/bufo-lora
```

Frozen VAE/text-encoder/UNet; only LoRA adapters train. Periodic preview grids
are written to `runs/bufo-lora/<ts>/snapshot-<step>/grid.png`, LoRA checkpoints to
`checkpoint-<step>/pytorch_lora_weights.safetensors`.

### 3. Sample

```bash
uv run python -m bufo.sample \
    --lora runs/bufo-lora/<ts>/checkpoint-1500 \
    --prompt "a bufo of astronaut" --prompt "a bufo of pizza" --num 4
```

Prompts get the `", frog emoji sticker, white background"` suffix automatically
(matching training captions); pass `--raw-prompt` to opt out. Images + a grid are
written under `runs/bufo-samples/`.

## Hardware notes

Built for a single Apple-silicon GPU (MPS, fp32). On CUDA, `amp: true` enables
bf16 autocast for the UNet forward. The base model (~4GB) downloads once to the
Hugging Face cache. SD 2.1-base is gated; `stable-diffusion-v1-5/stable-diffusion-v1-5`
and `stabilityai/sd-turbo` are open and work as `training.base_model`.

## Layout

| File | Role |
|------|------|
| `config.py` | Pydantic `DataConfig` / `LoRAConfig` / `TrainingConfig` (+ YAML) |
| `data.py` | Download, caption, preprocess, `BufoDataset` |
| `prepare.py` | CLI to build the dataset |
| `pipeline.py` | SD component loading + LoRA attach / save / load |
| `train_lora.py` | The fine-tuning loop |
| `sample.py` | Generate bufos from a trained LoRA |

## Tests

Fast offline tests (captions, preprocessing, dataset, config) run in `make check`.
The end-to-end training smoke downloads the base model, so it is gated:

```bash
BUFO_SMOKE=1 uv run pytest bufo/tests/test_train_smoke.py -s
```
