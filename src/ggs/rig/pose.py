"""2D forward kinematics: clip angles + bone lengths -> joint positions on screen."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ggs.rig.clip import Clip
from ggs.rig.spec import RigSpec

Point = tuple[float, float]


@dataclass(frozen=True)
class BoneDef:
    name: str
    parent: str | None
    attach: float = 1.0
    attach_side: float = 0.0  # standing heights along the parent's local x (see RIG_SPEC_V1)


def bone_defs(spec: RigSpec) -> list[BoneDef]:
    return [BoneDef(b.name, b.parent, b.attach, b.attach_side) for b in spec.bones]


def bone_vector(angle_deg: float, length: float) -> Point:
    """Screen vector of a bone: angle clockwise from +y (down), +y down."""
    a = math.radians(angle_deg)
    return (-length * math.sin(a), length * math.cos(a))


def leg_length(bones: Sequence[BoneDef], lengths: dict[str, float]) -> float:
    """Length of the first two bones of the near leg (the near chain hanging off the root)."""
    root = next(b.name for b in bones if b.parent is None)
    children: dict[str, list[str]] = {}
    for b in bones:
        if b.parent is not None:
            children.setdefault(b.parent, []).append(b.name)
    start = next((c for c in children.get(root, []) if c.endswith("_near")), None)
    if start is None:
        return 0.0
    total, cur = lengths[start], start
    nxt = children.get(cur, [])
    if nxt:
        total += lengths[nxt[0]]
    return total


def root_scale(bones: Sequence[BoneDef], clip: Clip, lengths: dict[str, float]) -> float:
    """Scale for the clip's root track when the character's legs differ from the mocap skeleton's.

    Scaling travel and hip height by the leg ratio keeps feet on the ground and limits foot sliding.
    """
    mine, theirs = leg_length(bones, lengths), leg_length(bones, clip.bone_lengths)
    return mine / theirs if mine > 0 and theirs > 0 else 1.0


def pose_bones(
    bones: Sequence[BoneDef],
    clip: Clip,
    frame: int,
    lengths: dict[str, float],
    scale_root: float = 1.0,
    in_place: bool = False,
) -> dict[str, tuple[Point, Point]]:
    """(start, end) of every bone at ``frame``, in standing heights."""
    by_name = {b.name: b for b in bones}
    out: dict[str, tuple[Point, Point]] = {}
    rx = 0.0 if in_place else clip.root.x[frame] * scale_root
    ry = clip.root.y[frame] * scale_root

    def place(name: str) -> tuple[Point, Point]:
        if name in out:
            return out[name]
        bone = by_name[name]
        if bone.parent is None:
            start = (rx, ry)
        else:
            (px0, py0), (px1, py1) = place(bone.parent)
            t = bone.attach
            start = (px0 + t * (px1 - px0), py0 + t * (py1 - py0))
            if bone.attach_side:  # local +x of the parent: (cos a, sin a) for parent angle a
                a = math.radians(clip.angles[bone.parent][frame])
                start = (start[0] + bone.attach_side * math.cos(a), start[1] + bone.attach_side * math.sin(a))
        dx, dy = bone_vector(clip.angles[name][frame], lengths[name])
        out[name] = (start, (start[0] + dx, start[1] + dy))
        return out[name]

    for b in bones:
        place(b.name)
    return out


def pose_frame(
    spec: RigSpec, clip: Clip, frame: int, lengths: dict[str, float] | None = None
) -> dict[str, tuple[Point, Point]]:
    """(start, end) of every bone at ``frame``. Lengths default to the clip's measured mocap lengths;
    other lengths also rescale the root track by the leg-length ratio."""
    bones = bone_defs(spec)
    if lengths is None:
        return pose_bones(bones, clip, frame, clip.bone_lengths)
    return pose_bones(bones, clip, frame, lengths, root_scale(bones, clip, lengths))
