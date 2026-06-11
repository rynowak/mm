"""End-to-end LoRA smoke: 2 steps on a synthetic dataset.

Gated behind ``BUFO_SMOKE=1`` because it downloads the ~4GB Stable Diffusion base
model. Run it manually after changing the training loop::

    BUFO_SMOKE=1 uv run pytest bufo/tests/test_train_smoke.py -s
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BUFO_SMOKE") != "1",
    reason="Set BUFO_SMOKE=1 to run the SD LoRA smoke (downloads the base model).",
)


def _build_dataset(root, n: int = 4):
    from PIL import Image

    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (64, 64), (40 * i % 255, 80, 120)).save(images_dir / f"b{i}.png")
    (root / "metadata.jsonl").write_text(
        "\n".join(json.dumps({"file_name": f"b{i}.png", "caption": f"a bufo of color {i}"}) for i in range(n))
    )


def test_train_two_steps_writes_lora(tmp_path):
    pytest.importorskip("diffusers")
    from bufo.config import BufoLoRAConfig, DataConfig, LoRAConfig, TrainingConfig
    from bufo.train_lora import train

    ds_dir = tmp_path / "ds"
    _build_dataset(ds_dir)
    cfg = BufoLoRAConfig(
        data=DataConfig(resolution=128, random_flip=False),
        lora=LoRAConfig(rank=4, alpha=4),
        training=TrainingConfig(
            batch_size=1,
            grad_accum=1,
            max_steps=2,
            warmup_steps=1,
            snapshot_interval=0,  # skip image generation in the smoke
            checkpoint_interval=2,
        ),
    )
    run_dir = train(cfg, data_dir=str(ds_dir), run_dir=tmp_path / "run")
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "checkpoint-2" / "pytorch_lora_weights.safetensors").exists()
