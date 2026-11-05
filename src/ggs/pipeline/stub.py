"""CPU stand-ins for the generator and matting model: deterministic, no downloads. Used by tests and
`ggs generate --model stub` to exercise the full recipe chain without a GPU."""

from __future__ import annotations

import random

from ggs.pipeline.post import flood
from ggs.pipeline.vram import Stages


class StubGenerator:
    name = "stub"

    def __init__(self, background=(255, 255, 255), cut_off: bool = False):
        self.background = background
        self.cut_off = cut_off

    supports_reference = True

    def render(self, jobs: list[tuple], width: int, height: int, stages: Stages, lora=None,
               lora_scale: float = 1.0) -> list:
        """(prompt, seed[, reference images]) jobs, like the real generators."""
        out = []
        for job in jobs:
            ref = job[2] if len(job) > 2 else None
            out += self.generate(job[0], [job[1]], width, height, stages, lora, lora_scale, reference=ref)
        return out

    def generate(self, prompt: str, seeds: list[int], width: int, height: int, stages: Stages,
                 lora=None, lora_scale: float = 1.0, reference: list | None = None) -> list:
        from PIL import Image, ImageDraw

        if reference:  # a perfect "edit": hand the reference back (CPU tests of the character master flow)
            return [reference[0].convert("RGB").resize((width, height)) for _ in seeds]

        out = []
        with stages.stage("denoise"):
            for seed in seeds:
                rnd = random.Random(f"{prompt}|{seed}")
                img = Image.new("RGB", (width * 4, height * 4), self.background)
                d = ImageDraw.Draw(img)
                cx, cy = width * 2 + rnd.randint(-width // 4, width // 4), height * 2
                r = (width * 4) // (2 if self.cut_off else 4)
                color = tuple(rnd.randint(40, 200) for _ in range(3))
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 20, 24))  # outline
                r2 = r - width // 8  # outline ~1/32 of the frame, like real outlined art
                d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=color)
                # area-average downscale: exact coverage mixes at the edge (no LANCZOS ringing)
                out.append(img.resize((width, height), Image.BOX))
        return out


class StubMatter:
    """Alpha from colour distance to the known background, for flat synthetic test images only. Like a real
    matting model, everything enclosed by the dark outline is solid (alpha 1), whatever its colour."""

    name = "stub"

    def __init__(self, background=(255, 255, 255)):
        self.background = background

    def alphas(self, images: list, stages: Stages) -> list:
        import numpy as np
        from PIL import Image

        out = []
        with stages.stage("matte"):
            bg = np.array(self.background, dtype=np.float32)
            for img in images:
                x = np.asarray(img.convert("RGB"), dtype=np.float32)
                d = np.abs(x - bg).max(axis=2)
                # an outline pixel fully covers the background when it differs by >= 200 levels
                a = np.clip(d / 200.0, 0, 1)
                border = np.zeros_like(d, dtype=bool)
                border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
                reachable = flood(border & (d < 200), d < 200)  # background + soft edge, outline blocks
                a = np.where(reachable, a, 1.0)
                out.append(Image.fromarray((a * 255).round().astype("uint8"), "L"))
        return out
