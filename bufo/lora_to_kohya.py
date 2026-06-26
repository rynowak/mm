"""Convert a diffusers/PEFT SDXL LoRA checkpoint to kohya format for ComfyUI / A1111.

``train_lora`` saves ``pytorch_lora_weights.safetensors`` in the diffusers convention
(``unet.<path>.lora.down.weight``, ``text_encoder.<path>.lora_linear_layer.down.weight``,
no alpha tensors). ComfyUI's LoRA loader wants the kohya convention
(``lora_unet_<path>.lora_down.weight`` / ``lora_te1_…`` / ``lora_te2_…`` plus ``.alpha``).

This is a pure key-renaming + alpha-synthesis pass — no math on the weights — so it is
lossless apart from the optional fp16 cast. Usage:

    python -m bufo.lora_to_kohya --checkpoint RUN/checkpoint-1000 --out out.safetensors --fp16
    # optionally also publish:
    #   --hf-repo rynowak/bufo-soul-lora --hf-path bufo-soul-v3-kohya.safetensors
"""

from __future__ import annotations

import argparse
import os

import torch
from safetensors.torch import load_file, save_file

# diffusers suffix -> kohya suffix
_TAILS = [
    (".lora.down.weight", "lora_down.weight"),
    (".lora.up.weight", "lora_up.weight"),
    (".lora_linear_layer.down.weight", "lora_down.weight"),
    (".lora_linear_layer.up.weight", "lora_up.weight"),
]
# diffusers top-level prefix -> kohya prefix
_PREFIXES = [
    ("unet.", "lora_unet_"),
    ("text_encoder_2.", "lora_te2_"),  # check the _2 form first
    ("text_encoder.", "lora_te1_"),
]


def convert(sd: dict[str, torch.Tensor], fp16: bool = True) -> dict[str, torch.Tensor]:
    """Rename diffusers LoRA keys to kohya and synthesize ``.alpha`` (= rank) tensors."""
    out: dict[str, torch.Tensor] = {}
    skipped = []
    for k, v in sd.items():
        match: tuple[str, str] | None = None
        for src, dst in _PREFIXES:
            if k.startswith(src):
                match = (dst, k[len(src) :])
                break
        if match is None:
            skipped.append(k)
            continue
        prefix, body = match
        matched = False
        for tail, suf in _TAILS:
            if body.endswith(tail):
                module = body[: -len(tail)].replace(".", "_")  # to_out.0 -> to_out_0
                w = v.half() if fp16 else v
                out[prefix + module + "." + suf] = w.contiguous()
                matched = True
                break
        if not matched:
            skipped.append(k)
    # kohya scale = alpha / rank; training used alpha == rank, so alpha = rank => scale 1.0.
    dtype = torch.float16 if fp16 else torch.float32
    for k in [x for x in list(out) if x.endswith(".lora_down.weight")]:
        out[k[: -len("lora_down.weight")] + "alpha"] = torch.tensor(float(out[k].shape[0]), dtype=dtype)
    if skipped:
        print(f"WARNING: {len(skipped)} keys did not match (left out): {skipped[:5]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Checkpoint dir or path to pytorch_lora_weights.safetensors")
    ap.add_argument("--out", required=True, help="Output kohya .safetensors path")
    ap.add_argument("--fp16", action="store_true", help="Cast weights to fp16 (smaller, ComfyUI-standard)")
    ap.add_argument("--hf-repo", default=None, help="Optional HF repo id to upload to (e.g. user/bufo-soul-lora)")
    ap.add_argument("--hf-path", default=None, help="Path within the HF repo (defaults to the out basename)")
    args = ap.parse_args()

    src = args.checkpoint
    if os.path.isdir(src):
        src = os.path.join(src, "pytorch_lora_weights.safetensors")
    sd = load_file(src)
    ko = convert(sd, fp16=args.fp16)
    te1 = sum(k.startswith("lora_te1") for k in ko)
    te2 = sum(k.startswith("lora_te2") for k in ko)
    unet = sum(k.startswith("lora_unet") for k in ko)
    alpha = sum(k.endswith(".alpha") for k in ko)
    print(f"converted {len(ko)} keys | unet {unet} te1 {te1} te2 {te2} alpha {alpha}")
    save_file(ko, args.out, metadata={"format": "pt"})
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.0f}MB)")

    if args.hf_repo:
        tok = None
        if os.path.exists("/mnt/ray/hf/token"):
            with open("/mnt/ray/hf/token") as fh:
                tok = fh.read().strip()
        tok = tok or os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")
        if not tok:
            print("NO_TOKEN: skipped HF upload; file is at", args.out)
            return
        from huggingface_hub import HfApi

        api = HfApi(token=tok)
        api.create_repo(args.hf_repo, private=True, exist_ok=True, repo_type="model")
        path_in_repo = args.hf_path or os.path.basename(args.out)
        api.upload_file(path_or_fileobj=args.out, path_in_repo=path_in_repo, repo_id=args.hf_repo)
        print(f"UPLOADED {args.hf_repo}/{path_in_repo}")


if __name__ == "__main__":
    main()
