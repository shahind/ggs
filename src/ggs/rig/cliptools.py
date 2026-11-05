"""Clip editing: find and cut a seamless loop (e.g. one walk cycle out of a longer take)."""

from __future__ import annotations

import numpy as np

from ggs.rig.clip import Clip, RootTrack


class ClipError(ValueError):
    pass


def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def find_loop(clip: Clip, min_seconds: float = 0.5, max_seconds: float = 2.0) -> tuple[int, int, float]:
    """Best (start, end, cost) so that frame ``end`` looks like frame ``start`` (pose and velocity).

    The loop plays frames start..end-1. Cost = RMS angle mismatch in degrees (pose + 0.5 x velocity)."""
    n = clip.frame_count
    lo = max(2, round(min_seconds * clip.fps))
    hi = min(n - 2, round(max_seconds * clip.fps))
    if lo > hi:
        raise ClipError(f"clip has {n} frames; too short for a {min_seconds:g}-{max_seconds:g} s loop")
    A = np.array([clip.angles[b] for b in clip.bones]).T  # (F, B)
    V = np.vstack([np.diff(A, axis=0), np.zeros((1, A.shape[1]))])
    ry = np.asarray(clip.root.y)
    best = (0, 0, float("inf"))
    for i in range(n - lo - 1):
        js = np.arange(i + lo, min(i + hi, n - 2) + 1)
        if js.size == 0:
            continue
        dp = _wrap(A[js] - A[i])
        dv = V[js] - V[i]
        dy = (ry[js] - ry[i]) * 360.0  # 1/360 standing height ~ 1 degree
        cost = np.sqrt(np.mean(dp**2 + 0.25 * dv**2, axis=1) + dy**2)
        k = int(np.argmin(cost))
        if cost[k] < best[2] - 1e-9:
            best = (i, int(js[k]), float(cost[k]))
    return best


def cut_loop(clip: Clip, start: int, end: int) -> Clip:
    """Frames start..end-1 as a looping clip. The small mismatch between frame ``end`` and ``start`` is
    spread linearly over the cycle, so the wrap from the last frame back to frame 0 is seamless."""
    if not (0 <= start < end < clip.frame_count):
        raise ClipError(f"loop {start}:{end} is outside the clip (0..{clip.frame_count - 1})")
    m = end - start
    ramp = np.arange(m) / m

    def seam(track: list[float], wrap: bool, nd: int = 3) -> list[float]:
        t = np.asarray(track, dtype=float)
        err = t[end] - t[start]
        if wrap:
            err = float(_wrap(np.array(err)))
        return [round(float(v), nd) for v in t[start:end] - err * ramp]

    angles = {b: seam(clip.angles[b], True) for b in clip.bones}
    parents_local = {b: seam(clip.local_angles[b], True) for b in clip.bones}
    local = {b: [round(float(v), 3) for v in _wrap(np.asarray(t))] for b, t in parents_local.items()}
    x = np.asarray(clip.root.x, dtype=float)
    travel = float(x[end] - x[start])
    root = RootTrack(x=[round(float(v), 5) for v in x[start:end] - x[start]],
                     y=seam(clip.root.y, False, 5))
    return clip.model_copy(update={
        "frame_count": m,
        "angles": angles,
        "local_angles": local,
        "root": root,
        "swaps": {g: v[start:end] for g, v in clip.swaps.items()},
        "loop": True,
        "loop_travel": round(travel, 5),
        "warnings": [*clip.warnings, f"loop cut from frames {start}..{end - 1}"],
    })
