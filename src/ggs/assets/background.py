"""Background with parallax layers.

The first layer (far: sky and distant scenery) is opaque: the whole scene is drawn, then a klein edit removes the
foreground and middle ground. Every later layer (mid, near) is drawn on its own on a plain white background (same
description, seed and style) and matted with BiRefNet for real alpha (no chroma key). Tried and dropped: the far
layer as a reference for the others (the model redraws the whole scene), and matting the scene itself (BiRefNet
finds one salient object, not a landscape's foreground).
With ``seamless_x`` every layer wraps horizontally (roll + seam repaint, before matting).
Outputs: ``<out>.png`` = the full scene (with extra layers: all layers composited), ``<out>_<layer>.png`` per
layer, and the JSON sidecar lists the layers back to front with their parallax factors.
v1 recipes use only the far layer: drawn foreground layers come out with painted haze instead of a flat white
background, so neither BiRefNet nor known-background matting separates them reliably (owner review pending).
"""

from __future__ import annotations

import numpy as np

from ggs.assets import numbered
from ggs.assets.seamless import make_seamless, seam_score
from ggs.recipes import layer_prompt

FAR_EDIT = ("Remove all trees, plants, rocks, buildings, characters and grass in the foreground and middle ground. "
            "Keep only the sky and the most distant scenery, and fill the removed areas with matching sky and "
            "distant scenery. Keep the same art style, colours and lighting.")


def run(*, recipe, description, out, seeds, style, size, gen, matter, stages, lora, lora_scale, keep_failed,
        debug):
    from PIL import Image

    from ggs.generate import AssetResult, real_rgba

    if not recipe.layers or not recipe.layers[0].opaque:
        raise ValueError(f"recipe {recipe.id!r}: a background needs layers, the first one opaque")
    w, h = size
    far_l, rest = recipe.layers[0], recipe.layers[1:]
    # far: draw the whole scene, then edit the foreground away (a plain "no foreground" prompt keeps trees)
    scenes = gen.render([(layer_prompt(recipe, far_l, description, style), s) for s in seeds], w, h, stages,
                        lora=lora, lora_scale=lora_scale)
    far = gen.render([(FAR_EDIT, s, [sc]) for s, sc in zip(seeds, scenes)], w, h, stages, lora=lora,
                     lora_scale=lora_scale)
    if recipe.seamless_x:
        scenes = make_seamless(scenes, "x", 0.14, gen, stages, lora=lora, lora_scale=lora_scale)
        far = make_seamless(far, "x", 0.14, gen, stages, lora=lora, lora_scale=lora_scale)
    # nearer layers: drawn on their own on white (a reference image makes the model redraw the whole scene)
    jobs = [(layer_prompt(recipe, layer, description, style), s) for s in seeds for layer in rest]
    drawn = gen.render(jobs, w, h, stages, lora=lora, lora_scale=lora_scale) if jobs else []
    if recipe.seamless_x and drawn:
        drawn = make_seamless(drawn, "x", 0.14, gen, stages, lora=lora, lora_scale=lora_scale)
    alphas = matter.alphas(drawn, stages) if drawn else []
    results, layers_meta = [], []
    for k, seed in enumerate(seeds):
        base = numbered(out, k, len(seeds))
        comp = far[k].convert("RGBA") if rest else scenes[k].convert("RGBA")
        files, qa, failures = [], {}, []
        far_path = base.with_name(f"{base.stem}_{far_l.name}{base.suffix}")
        far[k].save(far_path)
        files.append({"name": far_l.name, "file": far_path.name, "parallax": far_l.parallax, "opaque": True})
        if recipe.seamless_x:
            qa[f"{far_l.name}_seam"] = round(seam_score(np.asarray(far[k]), "x"), 3)
        for j, layer in enumerate(rest):
            idx = k * len(rest) + j
            # the layer touches the frame edges, so the background is the recipe's known colour, not estimated
            rgba, _, _ = real_rgba(drawn[idx], alphas[idx], recipe.post.alpha_low, recipe.post.alpha_high,
                                   background=recipe.background_rgb)
            cover = float((rgba[..., 3] >= 128).mean())
            qa[f"{layer.name}_coverage"] = round(cover, 3)
            if not 0.02 <= cover <= 0.85:
                failures.append(f"{layer.name}: covers {cover:.0%} of the frame (2-85% expected)")
            if recipe.seamless_x:
                qa[f"{layer.name}_seam"] = round(seam_score(np.asarray(drawn[idx]), "x"), 3)
            im = Image.fromarray(rgba, "RGBA")
            p = base.with_name(f"{base.stem}_{layer.name}{base.suffix}")
            im.save(p)
            if debug:
                drawn[idx].save(p.with_name(f"{p.stem}_raw.png"))
            comp.alpha_composite(im)
            files.append({"name": layer.name, "file": p.name, "parallax": layer.parallax, "opaque": False})
        if recipe.seamless_x:
            qa["scene_seam"] = round(seam_score(np.asarray(scenes[k]), "x"), 3)
        ok = not failures
        path = None
        if ok or keep_failed:
            path = base if ok else base.with_name(f"{base.stem}_QA_FAILED{base.suffix}")
            comp.convert("RGB").save(path)
        layers_meta.append({"seed": seed, "layers": files})
        results.append(AssetResult(seed, path, qa, ok, (w, h), failures))
    return results, {"background": {"seamless_x": recipe.seamless_x, "sets": layers_meta}}
