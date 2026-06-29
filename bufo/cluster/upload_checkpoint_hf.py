"""Upload the SD3.5 full-FT bufo checkpoint to a PRIVATE HuggingFace repo (durable).

Run on the cluster. Reads the write token from env HF_WRITE_TOKEN (never printed),
resolves the account namespace via whoami, creates a private model repo, and uploads
the kept transformer + dataset metadata + a small recipe card. Prints only the repo URL.
"""

from __future__ import annotations

import os

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

KEEP = "/mnt/ray/bufo-keep/sd35-medium-ft-1000"
REPO_NAME = os.environ.get("HF_REPO_NAME", "bufo-sd35-medium-ft")

CARD = """---
license: other
license_name: stabilityai-ai-community
base_model: stabilityai/stable-diffusion-3.5-medium
tags: [text-to-image, diffusers, sd3, bufo, full-finetune]
library_name: diffusers
---

# bufo — SD3.5-Medium full fine-tune

Full fine-tune (all transformer weights) of `stabilityai/stable-diffusion-3.5-medium`
on a curated bufo emoji set (242 curated canon + 90 teacher cells, 332 total),
adult-anchored captions: `"olive green adult bufo, {desc}, soft-shaded cartoon sticker"`.

- 1000 steps, lr 1e-5 constant, bf16, 8-bit Adam, grad-checkpointing, grad-accum 4, res 512.
- ~36 min on one A100-80GB. The `transformer/` here replaces the base model's transformer.

## Use
```python
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
t = SD3Transformer2DModel.from_pretrained("REPO_ID", subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", transformer=t, torch_dtype=torch.bfloat16).to("cuda")
img = pipe("olive green adult bufo, happy, soft-shaded cartoon sticker",
           negative_prompt="deformed, blurry, extra limbs, teeth, text, watermark",
           num_inference_steps=28, guidance_scale=4.5, height=1024, width=1024).images[0]
```
"""


def main() -> None:
    token = os.environ["HF_WRITE_TOKEN"]
    api = HfApi(token=token)
    who = api.whoami()
    ns = who["name"]
    repo_id = f"{ns}/{REPO_NAME}"
    create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=token)
    upload_folder(
        repo_id=repo_id,
        folder_path=KEEP,
        path_in_repo=".",
        token=token,
        commit_message="bufo SD3.5-medium full-FT checkpoint-1000",
        ignore_patterns=["*.lock"],
    )
    upload_file(
        path_or_fileobj=CARD.replace("REPO_ID", repo_id).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
        commit_message="recipe card",
    )
    print("UPLOADED https://huggingface.co/" + repo_id + " (private)")


if __name__ == "__main__":
    main()
