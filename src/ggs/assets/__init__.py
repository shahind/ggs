"""Asset kinds beyond single sprites (M8): background (parallax layers), tile (seamless), ui_panel (9-slice),
vfx (additive particle textures). ``run_asset`` has the same contract as ``ggs.generate.run_recipe``:
(list of AssetResult, metadata), with a JSON sidecar next to ``out``."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ggs import __version__
from ggs.recipes import Recipe


def numbered(out: Path, i: int, n: int, suffix: str = "") -> Path:
    """``out`` for one result, ``out_01`` ... for several; ``suffix`` goes before the extension."""
    stem = out.stem if n == 1 else f"{out.stem}_{i + 1:02d}"
    return out.with_name(f"{stem}{suffix}{out.suffix}")


def run_asset(recipe: Recipe, description: str, out: Path, *, model: str, seeds: list[int], profile: str,
              style: str, keep_failed: bool, enforce_vram: bool, size: tuple[int, int], debug: bool, lora,
              lora_scale: float, style_meta) -> tuple[list, dict]:
    from ggs.generate import build_backends
    from ggs.pipeline.vram import Stages

    t0 = time.perf_counter()
    stages = Stages(profile=profile, enforce=enforce_vram)
    gen, matter = build_backends(recipe, model, profile)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kw = {"recipe": recipe, "description": description, "out": out, "seeds": seeds, "style": style, "size": size,
          "gen": gen, "matter": matter, "stages": stages, "lora": lora, "lora_scale": lora_scale,
          "keep_failed": keep_failed, "debug": debug}
    if recipe.kind == "background":
        from ggs.assets.background import run
    elif recipe.kind == "tile":
        from ggs.assets.tile import run
    elif recipe.kind == "ui_panel":
        from ggs.assets.ui_panel import run
    elif recipe.kind == "vfx":
        from ggs.assets.vfx import run
    else:
        raise ValueError(f"recipe kind {recipe.kind!r} is not implemented")
    results, extra = run(**kw)
    meta = {
        "generator": f"ggs {__version__}", "class": recipe.id, "kind": recipe.kind, "model": model,
        "profile": profile, "description": description, "style": style,
        "style_package": None if style_meta is None else {"name": style_meta.name, "trigger": style_meta.trigger,
                                                          "weight": lora_scale},
        "size": list(size), "seconds": round(time.perf_counter() - t0, 2), "stages": stages.summary(),
        "assets": [{"seed": r.seed, "file": r.path.name if r.path else None, "passed": r.passed, "size": r.size,
                    "failures": r.failures, "qa": r.qa} for r in results],
        **extra,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return results, meta
