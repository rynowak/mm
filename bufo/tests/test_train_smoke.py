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


def test_train_text_encoder_lora(tmp_path):
    pytest.importorskip("diffusers")
    from bufo.config import BufoLoRAConfig, DataConfig, LoRAConfig, TrainingConfig
    from bufo.train_lora import train

    ds_dir = tmp_path / "ds"
    _build_dataset(ds_dir)
    cfg = BufoLoRAConfig(
        data=DataConfig(resolution=128, random_flip=False),
        lora=LoRAConfig(rank=4, alpha=4, train_text_encoder=True),  # exercise the TE-LoRA + multi-module optimizer path
        training=TrainingConfig(
            batch_size=1, grad_accum=1, max_steps=2, warmup_steps=1, snapshot_interval=0, checkpoint_interval=2
        ),
    )
    run_dir = train(cfg, data_dir=str(ds_dir), run_dir=tmp_path / "run")
    assert (run_dir / "checkpoint-2" / "pytorch_lora_weights.safetensors").exists()


def test_resume_continues_training(tmp_path):
    pytest.importorskip("diffusers")
    from bufo.config import BufoLoRAConfig, DataConfig, LoRAConfig, TrainingConfig
    from bufo.train_lora import train

    ds_dir = tmp_path / "ds"
    _build_dataset(ds_dir)

    def _cfg(max_steps: int) -> BufoLoRAConfig:
        return BufoLoRAConfig(
            data=DataConfig(resolution=128, random_flip=False),
            lora=LoRAConfig(rank=4, alpha=4),
            training=TrainingConfig(
                batch_size=1,
                grad_accum=1,
                max_steps=max_steps,
                warmup_steps=1,
                snapshot_interval=0,
                checkpoint_interval=2,
            ),
        )

    run = tmp_path / "run"
    train(_cfg(2), data_dir=str(ds_dir), run_dir=run)
    assert (run / "checkpoint-2" / "training_state.pt").exists()  # resumable state written
    # Resume from step 2 -> train to step 4 in the same run dir.
    train(_cfg(4), data_dir=str(ds_dir), run_dir=run, resume=str(run / "checkpoint-2"))
    assert (run / "checkpoint-4" / "pytorch_lora_weights.safetensors").exists()
    assert (run / "checkpoint-4" / "training_state.pt").exists()


@pytest.mark.skipif(
    os.environ.get("BUFO_FLUX_SMOKE") != "1",
    reason="Set BUFO_FLUX_SMOKE=1 to run the FLUX smoke (downloads ~30GB + needs an A100).",
)
def test_train_flux(tmp_path):
    pytest.importorskip("diffusers")
    from bufo.config import BufoLoRAConfig, DataConfig, LoRAConfig, TrainingConfig
    from bufo.train_lora import train

    ds_dir = tmp_path / "ds"
    _build_dataset(ds_dir)
    cfg = BufoLoRAConfig(
        data=DataConfig(resolution=256, random_flip=False),
        lora=LoRAConfig(rank=4, alpha=4),
        training=TrainingConfig(
            base_kind="flux",
            base_model="black-forest-labs/FLUX.1-dev",
            batch_size=1,
            grad_accum=1,
            max_steps=2,
            warmup_steps=1,
            snapshot_interval=0,
            checkpoint_interval=2,
            flux_max_sequence_length=64,  # keep the smoke fast
        ),
    )
    run_dir = train(cfg, data_dir=str(ds_dir), run_dir=tmp_path / "run")
    assert (run_dir / "checkpoint-2" / "pytorch_lora_weights.safetensors").exists()


@pytest.mark.skipif(
    os.environ.get("BUFO_SDXL_SMOKE") != "1",
    reason="Set BUFO_SDXL_SMOKE=1 (and BUFO_SMOKE=1) to run the SDXL smoke (downloads ~7GB).",
)
def test_train_sdxl(tmp_path):
    pytest.importorskip("diffusers")
    from bufo.config import BufoLoRAConfig, DataConfig, LoRAConfig, TrainingConfig
    from bufo.train_lora import train

    ds_dir = tmp_path / "ds"
    _build_dataset(ds_dir)
    cfg = BufoLoRAConfig(
        data=DataConfig(resolution=256, random_flip=False),
        lora=LoRAConfig(rank=4, alpha=4),
        training=TrainingConfig(
            base_kind="sdxl",
            base_model="stabilityai/stable-diffusion-xl-base-1.0",
            batch_size=1,
            grad_accum=1,
            max_steps=2,
            warmup_steps=1,
            snapshot_interval=0,
            checkpoint_interval=2,
        ),
    )
    run_dir = train(cfg, data_dir=str(ds_dir), run_dir=tmp_path / "run")
    assert (run_dir / "checkpoint-2" / "pytorch_lora_weights.safetensors").exists()
