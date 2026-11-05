"""QA for single RGBA sprites (needs numpy).

Checks, measured on the full matted frame before trimming:
- coverage: the object fills a sensible share of the frame (not empty, not the whole frame)
- cut_off: the object does not touch the frame edge (it would be clipped)
- background_flat: the generation background was flat enough for decontamination
- halo: soft edge pixels must not drift from the object's edge colour toward the background colour
  (the light/dark rim you see on a contrasting backdrop)
- leftover_background: the opaque rim of the object must not be background-coloured (under-cut matte)
- islands: no stray specks far from the object
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from pydantic import BaseModel

from ggs.pipeline.post import halo_fraction


class SpriteQaRules(BaseModel):
    min_coverage: float = 0.01
    max_coverage: float = 0.85
    edge_margin_px: int = 2
    max_background_spread: float = 12.0
    fringe_distance: float = 40.0  # an opaque rim pixel this close to the background colour is leftover
    halo_threshold: float = 0.25  # drift toward the background, as a share of |background - edge colour|
    max_halo: float = 0.10
    max_leftover_background: float = 0.05
    max_island_fraction: float = 0.002


@dataclass
class QaResult:
    passed: bool
    checks: dict[str, dict] = field(default_factory=dict)

    @property
    def failures(self) -> list[str]:
        return [f"{k}: {v['detail']}" for k, v in self.checks.items() if not v["ok"]]

    def to_dict(self) -> dict:
        return asdict(self)


def _rim(mask: np.ndarray) -> np.ndarray:
    """Pixels of ``mask`` that touch a pixel outside it (4-neighbourhood)."""
    m = mask
    inner = m.copy()
    inner[1:] &= m[:-1]
    inner[:-1] &= m[1:]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    return m & ~inner


def _largest_component_fraction(mask: np.ndarray) -> float:
    """Share of masked pixels outside the largest 4-connected component (iterative flood fill)."""
    total = int(mask.sum())
    if total == 0:
        return 0.0
    h, w = mask.shape
    seen = np.zeros_like(mask)
    best = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        best = max(best, n)
        if best > total // 2:
            break
    return (total - best) / total


def check_sprite(rgba: np.ndarray, background: np.ndarray, background_spread: float,
                 rules: SpriteQaRules | None = None) -> QaResult:
    rules = rules or SpriteQaRules()
    a = rgba[..., 3].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)
    bg = np.asarray(background, dtype=np.float32)

    checks: dict[str, dict] = {}

    def add(name, ok, value, detail):
        checks[name] = {"ok": bool(ok), "value": round(float(value), 4), "detail": detail}

    cov = float((a > 0.5).mean())
    add("coverage", rules.min_coverage <= cov <= rules.max_coverage, cov,
        f"object covers {cov:.1%} of the frame (allowed {rules.min_coverage:.0%}..{rules.max_coverage:.0%})")

    m = rules.edge_margin_px
    border = np.concatenate([a[:m].ravel(), a[-m:].ravel(), a[:, :m].ravel(), a[:, -m:].ravel()])
    edge_hits = float((border > 0.5).mean())
    add("cut_off", edge_hits == 0, edge_hits, "object touches the frame edge" if edge_hits else "clear of edges")

    add("background_flat", background_spread <= rules.max_background_spread, background_spread,
        f"background colour spread {background_spread:.1f} (max {rules.max_background_spread})")

    halo = halo_fraction(rgba, bg, rules.halo_threshold)
    add("halo", halo <= rules.max_halo, halo,
        f"{halo:.1%} of soft edge pixels drift toward the background colour")

    dist = np.abs(rgb - bg).max(axis=2)

    rim = _rim(a >= 0.95)
    left = float((dist[rim] < rules.fringe_distance).mean()) if rim.any() else 0.0
    add("leftover_background", left <= rules.max_leftover_background, left,
        f"{left:.1%} of the opaque rim is background-coloured")

    small = a > 0.5
    if small.sum() > 200_000:  # keep the flood fill fast on big frames
        small = small[::2, ::2]
    islands = _largest_component_fraction(small)
    add("islands", islands <= rules.max_island_fraction, islands,
        f"{islands:.2%} of the object is in specks apart from the main shape")

    return QaResult(all(c["ok"] for c in checks.values()), checks)
