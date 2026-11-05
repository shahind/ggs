"""Seamless wrapping: roll the image by half a period so the wrap seams meet in the middle, let the generator
repaint only a band around those middle lines, and measure the seam."""

from __future__ import annotations

import numpy as np

SEAM_PROMPT = ("Repair the straight seams running through the middle of this image so it becomes one continuous "
               "picture with no visible lines or breaks. Keep everything else exactly the same: same content, "
               "colours, drawing style and lighting.")


def roll_half(rgb: np.ndarray, axes: str) -> np.ndarray:
    out = rgb
    if "x" in axes:
        out = np.roll(out, out.shape[1] // 2, axis=1)
    if "y" in axes:
        out = np.roll(out, out.shape[0] // 2, axis=0)
    return out


def band_mask(h: int, w: int, axes: str, band: float) -> np.ndarray:
    """Soft weight, 1 on the middle seam lines, fading to 0 at ``band`` / 2 of the size from them."""
    m = np.zeros((h, w), np.float32)
    if "x" in axes:
        half = max(2.0, band * w / 2)
        d = np.abs(np.arange(w) - (w // 2 - 0.5))
        m = np.maximum(m, np.clip(1.5 - 1.5 * d / half, 0, 1)[None, :])
    if "y" in axes:
        half = max(2.0, band * h / 2)
        d = np.abs(np.arange(h) - (h // 2 - 0.5))
        m = np.maximum(m, np.clip(1.5 - 1.5 * d / half, 0, 1)[:, None])
    return m


def seam_score(rgb: np.ndarray, axes: str) -> float:
    """How much the seams stand out: the percentile rank (0..1) of the wrap edge (last line vs first) and of the
    middle line (where repaints meet) among all line-to-line changes of the image. A seam that is just another
    line of the texture ranks like any other line; a visible seam is an outlier near 1. The worst seam counts."""
    a = rgb.astype(np.float32)
    ranks = []
    for axis, flag in ((1, "x"), (0, "y")):
        if flag not in axes:
            continue
        other = tuple(i for i in range(a.ndim) if i != axis)
        steps = np.abs(np.diff(a, axis=axis)).mean(axis=other)
        wrap = float(np.abs(a.take(0, axis=axis) - a.take(-1, axis=axis)).mean())
        mid = float(steps[a.shape[axis] // 2 - 1])
        for v in (wrap, mid):
            ranks.append(float((steps < v).mean()))
    return max(ranks) if ranks else 0.0


def make_seamless(images: list, axes: str, band: float, gen, stages, seed: int = 0, lora=None,
                  lora_scale: float = 1.0, prompt: str = SEAM_PROMPT) -> list:
    """Wrap-seamless versions of PIL RGB ``images`` (one generator load for all)."""
    from PIL import Image

    rolled = [roll_half(np.asarray(im.convert("RGB")), axes) for im in images]
    w, h = images[0].size
    fixed = gen.render([(prompt, seed, [Image.fromarray(r)]) for r in rolled], w, h, stages, lora=lora,
                       lora_scale=lora_scale)
    m = band_mask(h, w, axes, band)[..., None]
    out = []
    for r, f in zip(rolled, fixed):
        f = match_colours(np.asarray(f.convert("RGB").resize((w, h)), dtype=np.float32), r.astype(np.float32),
                          m[..., 0] < 0.05)
        out.append(Image.fromarray(np.clip(r * (1 - m) + f * m, 0, 255).astype(np.uint8)))
    return out


def match_colours(edited: np.ndarray, original: np.ndarray, where: np.ndarray) -> np.ndarray:
    """Shift ``edited`` so its per-channel mean and spread match ``original`` over ``where`` (outside the seam
    band): an edit often changes the overall brightness a little, which would show as a band."""
    out = edited.copy()
    for c in range(3):
        e, o = edited[..., c][where], original[..., c][where]
        se, so = float(e.std()) + 1e-3, float(o.std()) + 1e-3
        out[..., c] = (edited[..., c] - float(e.mean())) * (so / se) + float(o.mean())
    return out


def resize_wrapped(img, size: tuple[int, int]):
    """Resize a wrapping image so its edges stay continuous: resize a 3x3 repeat, keep the centre."""
    from PIL import Image

    a = np.asarray(img)
    big = Image.fromarray(np.tile(a, (3, 3, 1) if a.ndim == 3 else (3, 3)))
    w, h = size
    big = big.resize((w * 3, h * 3), Image.LANCZOS)
    return big.crop((w, h, 2 * w, 2 * h))
