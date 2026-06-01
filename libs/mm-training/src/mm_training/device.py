"""Device auto-detection for training."""

from __future__ import annotations

import os
import platform
import subprocess

import torch


def get_device() -> torch.device:
    """Auto-detect best available device: MPS > CUDA > CPU.

    Sets PYTORCH_ENABLE_MPS_FALLBACK=1 when MPS is selected so ops without
    MPS kernels silently fall back to CPU instead of crashing.
    """
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict[str, str]:
    """Return device type, OS, and chip info for run manifests."""
    device = get_device()
    info: dict[str, str] = {
        "device_type": device.type,
        "os": platform.system(),
        "os_version": platform.version(),
        "python_arch": platform.machine(),
    }

    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = str(torch.cuda.device_count())
    elif device.type == "mps" and platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                timeout=5,
            ).strip()
            info["chip"] = chip
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            info["chip"] = "unknown"

    return info
