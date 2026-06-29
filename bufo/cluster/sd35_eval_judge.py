"""Generate eval images from the SD3.5 full-FT model and score them objectively.

- Loads SD3.5 base + the fine-tuned transformer (checkpoint-*/transformer).
- Generates the structured eval prompts (adult-anchored schema).
- Scores EVERY output by DINOv2 cosine similarity to the user-approved reference set
  (max over references) — an objective number, no Claude judgment.
- Writes a contact sheet + individual images + scores.json.

Env: BASE, FT (transformer dir), OUT, REF (approved-image dir), RES, IPP.
"""

from __future__ import annotations

import json
import os

import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from PIL import Image

BASE = os.environ.get("BASE", "stabilityai/stable-diffusion-3.5-medium")
FT = os.environ["FT"]
OUT = os.environ["OUT"]
REF = os.environ.get("REF", "/mnt/ray/bufo-data-teacher-v6/images")
RES = int(os.environ.get("RES", "1024"))
IPP = int(os.environ.get("IPP", "2"))

SUBJECTS = [
    "neutral",
    "happy",
    "sad",
    "crying",
    "angry",
    "smug",
    "surprised",
    "content",
    "happy, holding coffee",
    "neutral, wearing a wizard hat",
    "angry, holding a torch",
    "content, holding flowers",
    "happy, eating pizza",
    "neutral, holding a book",
    "concerned, sitting",
    "happy, arms raised",
]
PROMPT = "olive green adult bufo, {s}, soft-shaded cartoon sticker"
NEG = "deformed, blurry, low quality, extra limbs, teeth, fangs, text, watermark"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    img_out = os.path.join(OUT, "images")
    os.makedirs(img_out, exist_ok=True)
    dev = torch.device("cuda")

    transformer = SD3Transformer2DModel.from_pretrained(FT, torch_dtype=torch.bfloat16)
    pipe = StableDiffusion3Pipeline.from_pretrained(BASE, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)

    gen_paths: list[str] = []
    for i, s in enumerate(SUBJECTS):
        for j in range(IPP):
            g = torch.Generator(device=dev).manual_seed(1000 + j)
            img = pipe(
                prompt=PROMPT.format(s=s),
                negative_prompt=NEG,
                num_inference_steps=28,
                guidance_scale=4.5,
                height=RES,
                width=RES,
                generator=g,
            ).images[0]
            p = os.path.join(img_out, f"{i:02d}_{j}.png")
            img.save(p)
            gen_paths.append(p)
        print(f"gen {i + 1}/{len(SUBJECTS)}: {s}", flush=True)

    # contact sheet
    cols = IPP
    cell = 360
    rows = len(SUBJECTS)
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for idx, p in enumerate(gen_paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell - 8, cell - 8))
        x = (idx % cols) * cell + 4
        y = (idx // cols) * cell + 4
        sheet.paste(im, (x, y))
    sheet.save(os.path.join(OUT, "contact_sheet.png"))

    # DINOv2 judge: cosine sim of each generated image to the nearest approved reference
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()

    def embed(path: str) -> torch.Tensor:
        im = Image.open(path).convert("RGB")
        inp = proc(images=im, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = dino(**inp).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(out, dim=-1)[0]

    refs = [os.path.join(REF, f) for f in os.listdir(REF) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    ref_emb = torch.stack([embed(r) for r in refs])  # [R, D]
    scores = []
    for p in gen_paths:
        sim = embed(p) @ ref_emb.T  # [R]
        scores.append({"image": os.path.basename(p), "max_sim": float(sim.max()), "mean_sim": float(sim.mean())})
    import statistics as st

    maxs = [float(s["max_sim"]) for s in scores]
    summary = {
        "n_generated": len(scores),
        "n_refs": len(refs),
        "dino_max_sim_mean": round(st.mean(maxs), 4),
        "dino_max_sim_min": round(min(maxs), 4),
        "dino_max_sim_max": round(max(maxs), 4),
    }
    with open(os.path.join(OUT, "scores.json"), "w") as f:
        json.dump({"summary": summary, "per_image": scores}, f, indent=2)
    print("JUDGE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
