"""Seamless tile: a texture that wraps in x (and y), for terrain, walls and floors.

generate (square) -> roll by half a tile and repaint the middle seam band (``ggs.assets.seamless``) -> measure
the seam -> resize to ``tile.export_px``. The tile is opaque (it fills its cell), so there is no matting.
"""

from __future__ import annotations

import numpy as np

from ggs.assets import numbered
from ggs.assets.seamless import make_seamless, resize_wrapped, seam_score
from ggs.recipes import TileSettings


def run(*, recipe, description, out, seeds, style, size, gen, matter, stages, lora, lora_scale, keep_failed,
        debug):
    from PIL import Image

    from ggs.generate import AssetResult

    t = recipe.tile or TileSettings()
    side = min(size)
    prompt = recipe.prompt(description, style)
    raw = gen.render([(prompt, s) for s in seeds], side, side, stages, lora=lora, lora_scale=lora_scale)
    fixed = make_seamless(raw, t.seamless, t.band, gen, stages, lora=lora, lora_scale=lora_scale)
    results = []
    for i, (seed, r, f) in enumerate(zip(seeds, raw, fixed)):
        arr = np.asarray(f)
        before, after = seam_score(np.asarray(r), t.seamless), seam_score(arr, t.seamless)
        ok = after <= t.max_seam
        name = numbered(out, i, len(seeds))
        if debug:
            r.save(name.with_name(f"{name.stem}_raw.png"))
            big = Image.fromarray(np.tile(arr, (2, 2, 1)))  # 2x2 repeat: seams would show here
            big.save(name.with_name(f"{name.stem}_repeat.png"))
        path = None
        if ok or keep_failed:
            path = name if ok else name.with_name(f"{name.stem}_QA_FAILED{name.suffix}")
            resize_wrapped(f, (t.export_px, t.export_px)).save(path)
        failures = [] if ok else [f"seam {after:.2f} (max {t.max_seam})"]
        results.append(AssetResult(seed, path, {"seam_before": round(before, 3), "seam": round(after, 3),
                                                "seamless": t.seamless}, ok, (t.export_px, t.export_px), failures))
    return results, {"tile": {"seamless": t.seamless, "export_px": t.export_px}}
