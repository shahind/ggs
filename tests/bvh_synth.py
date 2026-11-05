"""Hand-made BVH files for tests. Joint names follow the CMU BVH conversions (cgspeed naming).

Skeleton (rest pose, Y up, character faces +Z, so its right side is -X), standing height 20:
legs 10.5 below the hips, spine/neck/head 9.5 above. Arms hang down in the rest pose.
"""

from __future__ import annotations

import math

CH3 = "CHANNELS 3 Zrotation Yrotation Xrotation"

# name, parent, offset, has_channels
_SKELETON: list[tuple[str, str | None, tuple[float, float, float]]] = [
    ("Hips", None, (0, 0, 0)),
    ("LHipJoint", "Hips", (0, 0, 0)),
    ("LeftUpLeg", "LHipJoint", (1, -1, 0)),
    ("LeftLeg", "LeftUpLeg", (0, -4.5, 0)),
    ("LeftFoot", "LeftLeg", (0, -4.5, 0)),
    ("LeftToeBase", "LeftFoot", (0, -0.5, 1.5)),
    ("RHipJoint", "Hips", (0, 0, 0)),
    ("RightUpLeg", "RHipJoint", (-1, -1, 0)),
    ("RightLeg", "RightUpLeg", (0, -4.5, 0)),
    ("RightFoot", "RightLeg", (0, -4.5, 0)),
    ("RightToeBase", "RightFoot", (0, -0.5, 1.5)),
    ("LowerBack", "Hips", (0, 0, 0)),
    ("Spine", "LowerBack", (0, 1.5, 0)),
    ("Spine1", "Spine", (0, 2, 0)),
    ("Neck", "Spine1", (0, 2, 0)),
    ("Neck1", "Neck", (0, 1, 0)),
    ("Head", "Neck1", (0, 1, 0)),
    ("LeftShoulder", "Spine1", (0, 2, 0)),
    ("LeftArm", "LeftShoulder", (1.5, 0, 0)),
    ("LeftForeArm", "LeftArm", (0, -3, 0)),
    ("LeftHand", "LeftForeArm", (0, -3, 0)),
    ("LeftFingerBase", "LeftHand", (0, 0, 0)),
    ("LeftHandIndex1", "LeftFingerBase", (0, -1, 0)),
    ("RightShoulder", "Spine1", (0, 2, 0)),
    ("RightArm", "RightShoulder", (-1.5, 0, 0)),
    ("RightForeArm", "RightArm", (0, -3, 0)),
    ("RightHand", "RightForeArm", (0, -3, 0)),
    ("RightFingerBase", "RightHand", (0, 0, 0)),
    ("RightHandIndex1", "RightFingerBase", (0, -1, 0)),
]
_END_SITES = {"LeftToeBase": (0, 0, 0.5), "RightToeBase": (0, 0, 0.5), "Head": (0, 2, 0),
              "LeftHandIndex1": (0, -0.5, 0), "RightHandIndex1": (0, -0.5, 0)}
JOINTS = [j[0] for j in _SKELETON]
HEIGHT = 20.0
HIP_HEIGHT = 10.5


def _hierarchy() -> str:
    children: dict[str, list[str]] = {}
    for name, parent, _ in _SKELETON:
        if parent:
            children.setdefault(parent, []).append(name)
    offsets = {n: o for n, _, o in _SKELETON}
    lines = ["HIERARCHY"]

    def emit(name: str, depth: int, root: bool) -> None:
        pad = "  " * depth
        lines.append(f"{pad}{'ROOT' if root else 'JOINT'} {name}")
        lines.append(pad + "{")
        lines.append(f"{pad}  OFFSET {' '.join(f'{v:g}' for v in offsets[name])}")
        if root:
            lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            lines.append(f"{pad}  {CH3}")
        for c in children.get(name, []):
            emit(c, depth + 1, False)
        if name in _END_SITES:
            lines.append(f"{pad}  End Site")
            lines.append(pad + "  {")
            lines.append(f"{pad}    OFFSET {' '.join(f'{v:g}' for v in _END_SITES[name])}")
            lines.append(pad + "  }")
        lines.append(pad + "}")

    emit("Hips", 0, True)
    return "\n".join(lines)


def make_bvh(pose_fn, frames: int, fps: float) -> str:
    """``pose_fn(i, t)`` returns {joint: (z, y, x) degrees} plus optional "_root": (x, y, z) translation."""
    rows = []
    for i in range(frames):
        pose = pose_fn(i, i / fps)
        root = pose.get("_root", (0.0, HIP_HEIGHT, 0.0))
        vals = list(root)
        for name in JOINTS:
            vals.extend(pose.get(name, (0.0, 0.0, 0.0)))
        rows.append(" ".join(f"{v:.6f}" for v in vals))
    return "\n".join([_hierarchy(), "MOTION", f"Frames: {frames}", f"Frame Time: {1 / fps:.8f}", *rows]) + "\n"


def walk_bvh(frames: int = 120, fps: float = 120.0, cycle_s: float = 1.0, yaw: float = 0.0,
             speed: float = 12.0, leg_amp: float = 25.0, knee_amp: float = 40.0,
             arm_amp: float = 20.0) -> str:
    """Procedural walk along the facing direction. Positive Xrotation swings a limb backward, so the
    forward swing is ``-amp``. ``yaw`` turns the whole character (and its path) about +Y."""
    w = 2 * math.pi / cycle_s
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))

    def pose(i: int, t: float):
        ph = w * t
        left_hip = -leg_amp * math.sin(ph)
        right_hip = leg_amp * math.sin(ph)
        left_knee = knee_amp * 0.5 * (1 - math.cos(ph + 0.6))
        right_knee = knee_amp * 0.5 * (1 - math.cos(ph + math.pi + 0.6))
        d = speed * t
        bob = 0.15 * math.cos(2 * ph)
        return {
            "_root": (d * sy, HIP_HEIGHT - 0.4 + bob, d * cy),
            "Hips": (0.0, yaw, 0.0),
            "LeftUpLeg": (0.0, 0.0, left_hip),
            "RightUpLeg": (0.0, 0.0, right_hip),
            "LeftLeg": (0.0, 0.0, left_knee),
            "RightLeg": (0.0, 0.0, right_knee),
            "LeftArm": (0.0, 0.0, arm_amp * math.sin(ph)),
            "RightArm": (0.0, 0.0, -arm_amp * math.sin(ph)),
            "LeftForeArm": (0.0, 0.0, -15.0),
            "RightForeArm": (0.0, 0.0, -15.0),
        }

    return make_bvh(pose, frames, fps)
