"""Post-processing on numpy arrays (needs the ``image`` extra): background estimate, colour decontamination,
alpha cleanup and trimming.

Decontamination: the generator paints the object over a known flat background B. A soft edge pixel shows
C = a*F + (1-a)*B, so the true colour is F = (C - (1-a)*B) / a. Solving for F removes the light/dark halo that
a plain alpha cut leaves (no chroma keying: alpha comes from the matting model).
"""

from __future__ import annotations

import numpy as np


def estimate_background(rgb: np.ndarray, border: int = 8) -> tuple[np.ndarray, float]:
    """Median colour of the image border and its spread (max per-channel MAD). A large spread means the
    background was not flat."""
    h, w = rgb.shape[:2]
    b = min(border, h // 4, w // 4)
    ring = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                           rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)]).astype(np.float32)
    med = np.median(ring, axis=0)
    spread = float(np.median(np.abs(ring - med), axis=0).max())
    return med, spread


def clean_alpha(alpha: np.ndarray, low: float = 0.04, high: float = 0.96) -> np.ndarray:
    """Snap near-transparent and near-opaque values (matting noise) to 0 and 1; keep the soft edge between."""
    a = alpha.astype(np.float32)
    a = np.where(a <= low, 0.0, a)
    a = np.where(a >= high, 1.0, a)
    return a


def decontaminate(rgb: np.ndarray, alpha: np.ndarray, background: np.ndarray, min_alpha: float = 0.05) -> np.ndarray:
    """Recover foreground colour on soft edges. ``rgb`` uint8/float (H,W,3), ``alpha`` float 0..1 (H,W)."""
    c = rgb.astype(np.float32)
    a = alpha.astype(np.float32)[..., None]
    b = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    safe = np.maximum(a, min_alpha)
    f = (c - (1.0 - a) * b) / safe
    soft = (a > 0) & (a < 1)
    out = np.where(soft, f, c)
    return np.clip(out, 0, 255)


def _shift(x: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(x)
    h, w = x.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = x[ys, xs]
    return out


def nearest_solid_color(rgb: np.ndarray, solid: np.ndarray, steps: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Colour of the nearest solid pixel, spread outward ``steps`` px by neighbour averaging.
    Returns (colour (H,W,3) float, known mask)."""
    col = np.where(solid[..., None], rgb.astype(np.float32), 0.0)
    known = solid.astype(np.float32)
    for _ in range(steps):
        acc_c = np.zeros_like(col)
        acc_n = np.zeros_like(known)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            acc_c += _shift(col * known[..., None], dy, dx)
            acc_n += _shift(known, dy, dx)
        new = (known == 0) & (acc_n > 0)
        col = np.where(new[..., None], acc_c / np.maximum(acc_n, 1)[..., None], col)
        known = np.where(new, 1.0, known)
    return col, known > 0


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(px):
        out = out | _shift(out, 1, 0) | _shift(out, -1, 0) | _shift(out, 0, 1) | _shift(out, 0, -1)
    return out


def flood(seed: np.ndarray, allowed: np.ndarray, max_iter: int = 4096) -> np.ndarray:
    """Grow ``seed`` through 4-connected ``allowed`` pixels."""
    region = seed.copy()
    for _ in range(max_iter):
        grown = region | (_dilate(region, 1) & allowed)
        if np.array_equal(grown, region):
            break
        region = grown
    return region


def refine_alpha_known_background(rgb: np.ndarray, alpha: np.ndarray, background: np.ndarray,
                                  min_contrast: float = 30.0, band_px: int = 2) -> np.ndarray:
    """Tighten the matte edge using the flat, known background B.

    1. Background-coloured pixels connected to the outer background (the transparent area touching the image
       border) are leftover background whatever the matte says: alpha 0. Enclosed light areas (inside an
       outline) and see-through holes the matting model found inside the object are not grown.
    2. In a ``band_px`` band next to transparency, alpha = min(matte, a_colour), where a_colour is the
       projection of (C - B) onto (F - B) and F is the nearest solid object colour that clearly differs from
       B. Mostly-background edge pixels stop carrying a pale/dark rim.
    """
    c = rgb.astype(np.float32)
    b = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    contrast = np.abs(c - b).max(axis=2)
    border = np.zeros(alpha.shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    transparent = alpha < 0.5
    outer = flood(border & transparent, transparent)  # only the background around the object, not holes in it
    outside = flood(outer, contrast < min_contrast)
    alpha = np.where(outside, 0.0, alpha).astype(np.float32)
    near_edge = _dilate(alpha < 0.5, band_px)
    core = (alpha >= 0.95) & (contrast >= min_contrast) & ~_dilate(alpha < 0.95, 1)
    f, known = nearest_solid_color(c, core, steps=band_px + 4)
    fb = f - b
    denom = (fb * fb).sum(axis=2)
    a_col = np.clip(((c - b) * fb).sum(axis=2) / np.maximum(denom, 1e-6), 0.0, 1.0)
    band = (alpha > 0) & near_edge & known & (np.sqrt(denom) >= min_contrast)
    return np.where(band, np.minimum(alpha, a_col), alpha).astype(np.float32)


def halo_fraction(rgba: np.ndarray, background: np.ndarray, threshold: float = 0.25) -> float:
    """Share of soft edge pixels whose colour has drifted from the nearest solid colour F toward the background
    B by more than ``threshold`` of |B - F| (the visible light/dark rim on a contrasting backdrop)."""
    a = rgba[..., 3].astype(np.float32) / 255.0
    c = rgba[..., :3].astype(np.float32)
    b = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    solid = a >= 0.95
    f, known = nearest_solid_color(c, solid, steps=4)
    bf = b - f
    denom = (bf * bf).sum(axis=2)
    soft = (a > 0.05) & (a < 0.95) & known & (np.sqrt(denom) >= 30.0)
    if not soft.any():
        return 0.0
    drift = ((c - f) * bf).sum(axis=2) / np.maximum(denom, 1e-6)
    return float((drift[soft] > threshold).mean())


def to_rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgb8 = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    rgb8[a == 0] = 0  # no hidden colour under fully transparent pixels
    return np.dstack([rgb8, a])


def trim(rgba: np.ndarray, padding: int = 4) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop to the alpha bounding box plus ``padding`` px of transparent margin on every side.
    Returns (image, box in source pixels: x0, y0, x1, y1)."""
    ys, xs = np.nonzero(rgba[..., 3])
    if ys.size == 0:
        raise ValueError("image is fully transparent")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    crop = rgba[y0:y1, x0:x1]
    out = np.zeros((crop.shape[0] + 2 * padding, crop.shape[1] + 2 * padding, 4), dtype=np.uint8)
    out[padding:padding + crop.shape[0], padding:padding + crop.shape[1]] = crop
    return out, (x0, y0, x1, y1)


def pad_to_pot(rgba: np.ndarray) -> np.ndarray:
    """Centre on the next power-of-two canvas (each side independently)."""
    def pot(n: int) -> int:
        p = 1
        while p < n:
            p *= 2
        return p

    h, w = rgba.shape[:2]
    H, W = pot(h), pot(w)
    out = np.zeros((H, W, 4), dtype=np.uint8)
    y, x = (H - h) // 2, (W - w) // 2
    out[y:y + h, x:x + w] = rgba
    return out
