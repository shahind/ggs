"""GPU detection and VRAM profiles."""

from __future__ import annotations

import shutil
import subprocess

STANDARD_MIN_GB = 12.0
LITE_MIN_GB = 7.0


def detect_vram_gb() -> float | None:
    """Total VRAM of GPU 0 in GB, or None if no NVIDIA GPU can be found."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        pass
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip().splitlines()
        return float(out[0]) / 1024.0  # MiB -> GiB
    except Exception:
        return None


def profile_for(vram_gb: float | None) -> str:
    """Map VRAM to a profile name: standard, lite, minimal, or no-gpu."""
    if vram_gb is None:
        return "no-gpu"
    if vram_gb >= STANDARD_MIN_GB:
        return "standard"
    if vram_gb >= LITE_MIN_GB:
        return "lite"
    return "minimal"
