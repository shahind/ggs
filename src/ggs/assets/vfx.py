"""VFX particle texture for additive blending (sparks, glows, magic, fire).

Glowing effects are drawn as light on a pure black background. For additive blending, black adds nothing, so the
texture's alpha is its brightness and its colour is the light divided by that (premultiplied by nature): the
effect fades smoothly to nothing and there is no edge to fringe. This is the standard format for emissive
particles, not a chroma-key cut-out of an object (hard rule 5 is about cut-outs); use the ``sprite`` class for
solid objects. QA: the effect must fade out before the frame edge and cover a sensible area.
"""

from __future__ import annotations

import numpy as np

from ggs.assets import numbered
from ggs.recipes import VfxSettings


def additive_rgba(rgb: np.ndarray) -> np.ndarray:
    """Light on black -> straight RGBA whose alpha-blend equals the additive look (alpha = brightest channel)."""
    c = rgb.astype(np.float32) / 255.0
    a = c.max(axis=2)
    colour = np.where(a[..., None] > 1e-3, c / np.maximum(a[..., None], 1e-3), 0.0)
    out = np.zeros((*rgb.shape[:2], 4), np.uint8)
    out[..., :3] = np.clip(colour * 255, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(a * 255, 0, 255).astype(np.uint8)
    return out


def run(*, recipe, description, out, seeds, style, size, gen, matter, stages, lora, lora_scale, keep_failed,
        debug):
    from PIL import Image

    from ggs.generate import AssetResult

    v = recipe.vfx or VfxSettings()
    w, h = size
    prompt = recipe.prompt(description, style)
    images = gen.render([(prompt, seed) for seed in seeds], w, h, stages, lora=lora, lora_scale=lora_scale)
    results = []
    for i, (seed, img) in enumerate(zip(seeds, images)):
        rgb = np.asarray(img.convert("RGB"))
        rgba = additive_rgba(rgb)
        a = rgba[..., 3].astype(np.float32) / 255
        edge = float(max(a[0].max(), a[-1].max(), a[:, 0].max(), a[:, -1].max()))
        cover = float((a > 0.1).mean())
        failures = []
        if edge > v.max_edge:
            failures.append(f"the effect reaches the frame edge (brightness {edge:.2f})")
        if not 0.01 <= cover <= 0.7:
            failures.append(f"covers {cover:.0%} of the frame (1-70% expected)")
        ok = not failures
        name = numbered(out, i, len(seeds))
        if debug:
            img.save(name.with_name(f"{name.stem}_raw.png"))
        path = None
        if ok or keep_failed:
            path = name if ok else name.with_name(f"{name.stem}_QA_FAILED{name.suffix}")
            im = Image.fromarray(rgba, "RGBA").convert("RGBa").resize((v.export_px, v.export_px), Image.LANCZOS)
            im.convert("RGBA").save(path)
        results.append(AssetResult(seed, path, {"edge": round(edge, 3), "coverage": round(cover, 3),
                                                "blend": "additive"}, ok, (v.export_px, v.export_px), failures))
    return results, {"vfx": {"blend": "additive", "export_px": v.export_px}}
