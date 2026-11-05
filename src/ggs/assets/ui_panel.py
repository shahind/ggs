"""UI panel / button with 9-slice margins and states.

generate on white -> BiRefNet matte -> trim -> find the frame: from each side, the first line of pixels that
matches the panel's plain centre colour ends the border -> QA (all four borders found, centre flat enough to
stretch) -> states: hover (brighter) and pressed (darker) -> PNGs + JSON with the slice margins in each engine's
own order (Unity sprite border L,B,R,T; Godot NinePatchRect patch margins; Defold slice9 L,T,R,B).
"""

from __future__ import annotations

import numpy as np

from ggs.assets import numbered
from ggs.recipes import PanelSettings


def find_slices(rgba: np.ndarray, s: PanelSettings) -> tuple[dict, dict]:
    """Slice margins (px) from the border inwards, and scores. The centre colour is the median of the middle."""
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3].astype(np.float32)
    on = rgba[..., 3] >= 128
    cy0, cy1, cx0, cx1 = int(h * 0.4), int(h * 0.6) + 1, int(w * 0.4), int(w * 0.6) + 1
    centre = np.median(rgb[cy0:cy1, cx0:cx1].reshape(-1, 3), axis=0)
    spread = float(np.abs(rgb[cy0:cy1, cx0:cx1] - centre).mean())
    close = (np.abs(rgb - centre).mean(axis=2) <= s.centre_tolerance) & on

    def margin(lines: np.ndarray, limit: int) -> int:
        """First index (from the outside) where most of the middle stretch of a line is centre-coloured."""
        for i in range(limit):
            if lines[i] >= 0.8:
                return i
        return -1

    rows_mid = close[:, int(w * 0.3):int(w * 0.7) + 1].mean(axis=1)  # per row, share of centre colour
    cols_mid = close[int(h * 0.3):int(h * 0.7) + 1, :].mean(axis=0)
    lim_y, lim_x = int(h * s.max_border), int(w * s.max_border)
    m = {"top": margin(rows_mid, lim_y), "bottom": margin(rows_mid[::-1], lim_y),
         "left": margin(cols_mid, lim_x), "right": margin(cols_mid[::-1], lim_x)}
    return m, {"centre_spread": round(spread, 2), "centre": [round(float(c), 1) for c in centre]}


def state(rgba: np.ndarray, factor: float) -> np.ndarray:
    out = rgba.copy()
    out[..., :3] = np.clip(rgba[..., :3].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out


def run(*, recipe, description, out, seeds, style, size, gen, matter, stages, lora, lora_scale, keep_failed,
        debug):
    import json

    from PIL import Image

    from ggs.generate import AssetResult, real_rgba
    from ggs.pipeline.post import trim

    s = recipe.panel or PanelSettings()
    w, h = size
    prompt = recipe.prompt(description, style)
    images = gen.render([(prompt, seed) for seed in seeds], w, h, stages, lora=lora, lora_scale=lora_scale)
    alphas = matter.alphas(images, stages)
    results = []
    for i, (seed, img, alpha) in enumerate(zip(seeds, images, alphas)):
        rgba, _, _ = real_rgba(img, alpha, recipe.post.alpha_low, recipe.post.alpha_high)
        name = numbered(out, i, len(seeds))
        if debug:
            img.save(name.with_name(f"{name.stem}_raw.png"))
        if not rgba[..., 3].any():
            results.append(AssetResult(seed, None, {}, False, None, ["empty"]))
            continue
        panel, _ = trim(rgba, 0)
        m, scores = find_slices(panel, s)
        failures = [f"no {side} border found" for side, v in m.items() if v < s.min_border]
        if scores["centre_spread"] > s.max_centre_spread:
            failures.append(f"centre not flat enough to stretch (spread {scores['centre_spread']})")
        ok = not failures
        path = None
        ph, pw = panel.shape[:2]
        if ok or keep_failed:
            path = name if ok else name.with_name(f"{name.stem}_QA_FAILED{name.suffix}")
            Image.fromarray(panel, "RGBA").save(path)
            if s.states:
                Image.fromarray(state(panel, 1.12), "RGBA").save(path.with_name(f"{path.stem}_hover.png"))
                Image.fromarray(state(panel, 0.85), "RGBA").save(path.with_name(f"{path.stem}_pressed.png"))
            mm = {k: max(v, 0) for k, v in m.items()}
            (path.with_name(f"{path.stem}_slices.json")).write_text(json.dumps({
                "image": path.name, "size": [pw, ph], "slices": mm,
                "unity_border_LBRT": [mm["left"], mm["bottom"], mm["right"], mm["top"]],
                "godot_patch_margin": {"left": mm["left"], "top": mm["top"], "right": mm["right"],
                                       "bottom": mm["bottom"]},
                "defold_slice9_LTRB": [mm["left"], mm["top"], mm["right"], mm["bottom"]],
                "states": ["normal", "hover", "pressed"] if s.states else ["normal"]}, indent=1), encoding="utf-8")
        results.append(AssetResult(seed, path, {**m, **scores}, ok, (pw, ph), failures))
    return results, {}
