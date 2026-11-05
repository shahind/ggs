"""Touching up an animation for one character: turn a bone in one frame, blended into the frames around it.

The edited animation is a normal clip file saved with the character (``<item>/edits/<name>.json``), so exports
(app or CLI) play exactly what was seen in the editor. A turn moves the bone and everything hanging from it
(rotating the upper arm carries the forearm and hand), and fades out over ``spread`` frames each side.
"""

from __future__ import annotations

import math
from pathlib import Path

from ggs.rig.clip import Clip


def edits_dir(item_dir: Path) -> Path:
    return Path(item_dir) / "edits"


def edited_clip(item_dir: Path, name: str) -> Path | None:
    p = edits_dir(item_dir) / f"{name}.json"
    return p if p.exists() else None


def descendants(clip_bones: list[str], parents: dict[str, str | None], bone: str) -> list[str]:
    out, todo = [], [bone]
    while todo:
        b = todo.pop()
        out.append(b)
        todo += [c for c in clip_bones if parents.get(c) == b]
    return out


def turn_bone(clip: Clip, parents: dict[str, str | None], bone: str, frame: int, degrees: float,
              spread: int = 3) -> Clip:
    """A copy of ``clip`` with ``bone`` (and its children) turned by ``degrees`` at ``frame``, fading out over
    ``spread`` frames each side (a smooth bump, wrapping round for looping clips)."""
    angles = {b: list(v) for b, v in clip.angles.items()}
    local = {b: list(v) for b, v in clip.local_angles.items()}
    n = clip.frame_count
    moved = descendants(clip.bones, parents, bone)
    for k in range(-spread, spread + 1):
        f = frame + k
        if clip.loop:
            f %= n
        elif not 0 <= f < n:
            continue
        w = 0.5 * (1 + math.cos(math.pi * k / (spread + 1)))  # 1 at the frame, 0 past the spread
        for b in moved:
            angles[b][f] += degrees * w
        local[bone][f] = ((local[bone][f] + degrees * w + 180.0) % 360.0) - 180.0
    return clip.model_copy(update={"angles": angles, "local_angles": local})
