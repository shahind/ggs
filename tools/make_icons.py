"""Make the app's icon set with GGS itself: one style description and one seed for every icon, so they read as a
set. Writes src/ggs/ui/icons/<name>.png (64 px) and <name>@2x.png (128 px). Run: python tools/make_icons.py

The icons are generated with FLUX.2 klein (Apache-2.0) and BiRefNet (MIT), like any GGS asset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STYLE = ("a flat minimal app icon, soft rounded geometric shapes, two-tone lavender and deep indigo with a small "
         "white highlight, no outline, no gradient, no shadow, centered with generous empty space, plain flat pure "
         "white background, no text, no letters")
ICONS = {
    # asset types
    "sprite": "a small treasure box",
    "held_item": "a sword",
    "ui_icon": "a faceted gem",
    "background": "a landscape with mountains and a sun",
    "background_wrap": "a landscape with two circular arrows around it",
    "tile": "a square of brick tiles",
    "ui_panel": "a rectangular window frame",
    "vfx": "a four pointed sparkle star with small sparkles",
    "character": "a simple person figure",
    # tree sections
    "characters": "two simple person figures",
    "assets": "a stack of boxes",
    "animations": "a film strip",
    "styles": "a painter's palette with a brush",
    # actions
    "new": "a plus sign in a circle",
    "settings": "a gear",
    "projects": "a grid of four rounded squares",
    "regenerate": "a circular arrow",
    "variations": "three overlapping cards",
    "export": "a box with an arrow pointing up out of it",
    "advanced": "three horizontal sliders",
    "send": "a paper plane",
    "front": "a person seen from the front with arms out",
    "side": "a person seen from the side walking",
    "save": "a bookmark",
}
SEED = 11


def main() -> int:
    from PIL import Image

    from ggs.generate import build_backends, process
    from ggs.pipeline.vram import Stages
    from ggs.recipes import load_recipe

    out = ROOT / "src" / "ggs" / "ui" / "icons"
    out.mkdir(parents=True, exist_ok=True)
    recipe = load_recipe("ui_icon")
    gen, matter = build_backends(recipe, "flux2-klein-4b", "lite")
    names = list(ICONS)
    prompts = [" ".join(f"{STYLE}, of {ICONS[n]}".split()) for n in names]
    stages = Stages(profile="lite")
    images = gen.render([(p, SEED) for p in prompts], 768, 768, stages)
    alphas = matter.alphas(images, stages)
    report = {}
    for name, img, alpha in zip(names, images, alphas):
        arr, qa, _ = process(img, alpha, recipe)
        report[name] = {"passed": qa.passed, "failures": qa.failures}
        if arr is None:
            continue
        im = Image.fromarray(arr, "RGBA")
        side = max(im.size)
        sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        sq.alpha_composite(im, ((side - im.width) // 2, (side - im.height) // 2))
        for px, suffix in ((64, ""), (128, "@2x")):
            small = sq.convert("RGBa").resize((px - px // 8, px - px // 8), Image.LANCZOS).convert("RGBA")
            canvas = Image.new("RGBA", (px, px), (0, 0, 0, 0))
            canvas.alpha_composite(small, (px // 16, px // 16))
            canvas.save(out / f"{name}{suffix}.png")
        img.save(ROOT / "outputs" / f"icon_raw_{name}.png") if (ROOT / "outputs").exists() else None
    (out / "icons.json").write_text(json.dumps({"style": STYLE, "seed": SEED, "model": "flux2-klein-4b",
                                                "icons": {n: ICONS[n] for n in names}, "qa": report}, indent=1),
                                    encoding="utf-8")
    bad = [n for n, r in report.items() if not r["passed"]]
    print(f"{len(names) - len(bad)}/{len(names)} icons passed QA" + (f"; check {bad}" if bad else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
