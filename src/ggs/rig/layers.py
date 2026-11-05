"""Keyframed clips and animation layers (M7).

Keyframe files (``clips/*.yaml``) hold short hand-made motions that mocap does not cover: hold poses, sword swings.

    format: ggs.keys
    version: 1
    name: swing_one_hand
    fps: 24
    frames: 16
    loop: false
    keys:                      # absolute angles (degrees clockwise from straight down), body upright
      - {frame: 0, angles: {upper_arm_near: -170, forearm_near: -190, hand_near: -190}}
      - {frame: 6, angles: {upper_arm_near: -80, ...}}

Bones a key does not name keep the previous key's value (or the rest pose). Between keys the angle eases in and
out (smoothstep). Values are not wrapped, so -190 means "past straight up", and the motion goes the way written.

Layers: ``layer_clips(base, overlay, rig, bones)`` plays ``overlay`` on ``bones`` (by default the upper body)
over ``base``. The overlay's angles are taken *relative to the parent bone*, so the arms stay on a torso that
moves with the base clip (a swing while walking).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ggs.rig.clip import Clip, RootTrack
from ggs.rig.master import wrap180
from ggs.rig.spec import RigSpec

REST_POSE = {"pelvis": 180.0, "torso": 180.0, "head": 180.0, "foot_near": -90.0, "foot_far": -90.0}

BONE_SETS = {
    "upper": ("torso", "head", "upper_arm_near", "forearm_near", "hand_near", "upper_arm_far", "forearm_far",
              "hand_far"),
    "arms": ("upper_arm_near", "forearm_near", "hand_near", "upper_arm_far", "forearm_far", "hand_far"),
    "near_arm": ("upper_arm_near", "forearm_near", "hand_near"),
}


class Key(BaseModel):
    frame: int = Field(ge=0)
    angles: dict[str, float]


class KeyClip(BaseModel):
    format: Literal["ggs.keys"] = "ggs.keys"
    version: Literal[1] = 1
    name: str
    fps: float = Field(24.0, gt=0)
    frames: int = Field(gt=0)
    loop: bool = False
    keys: list[Key]


def load_keys(path: str | Path) -> KeyClip:
    path = Path(path)
    try:
        return KeyClip.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def keys_to_clip(keys: KeyClip, rig: RigSpec, lengths: dict[str, float] | None = None) -> Clip:
    names = [b.name for b in rig.bones]
    unknown = {b for k in keys.keys for b in k.angles} - set(names)
    if unknown:
        raise ValueError(f"keys name bones the rig does not have: {sorted(unknown)}")
    ks = sorted(keys.keys, key=lambda k: k.frame)
    if not ks or ks[0].frame != 0:
        raise ValueError("the first key must be at frame 0")
    if ks[-1].frame >= keys.frames:
        raise ValueError(f"key at frame {ks[-1].frame} is past the clip's {keys.frames} frames")
    # fill each key with every bone (carry the previous key's values)
    full, cur = [], {n: REST_POSE.get(n, 0.0) for n in names}
    for k in ks:
        cur = {**cur, **k.angles}
        full.append((k.frame, dict(cur)))
    if keys.loop:
        full.append((keys.frames, dict(full[0][1])))
    angles = {n: [] for n in names}
    for f in range(keys.frames):
        j = max(i for i, (fr, _) in enumerate(full) if fr <= f)
        fa, a = full[j]
        if j + 1 < len(full):
            fb, b = full[j + 1]
            t = _ease((f - fa) / (fb - fa))
        else:
            b, t = a, 0.0
        for n in names:
            angles[n].append(a[n] + (b[n] - a[n]) * t)
    lengths = lengths or {b.name: b.rest_length for b in rig.bones}
    hip = lengths.get("thigh_near", 0.24) + lengths.get("shin_near", 0.24)
    parents = {b.name: b.parent for b in rig.bones}
    local = {n: [wrap180(angles[n][i] - (angles[parents[n]][i] if parents[n] else 0.0))
                 for i in range(keys.frames)] for n in names}
    return Clip(rig=rig.id, source=f"keys:{keys.name}", profile="keys", fps=keys.fps, frame_count=keys.frames,
                in_place=True, loop=keys.loop, bones=names, bone_lengths={n: lengths[n] for n in names},
                angles=angles, local_angles=local,
                root=RootTrack(x=[0.0] * keys.frames, y=[-hip] * keys.frames))


def _unwrapped_local(clip: Clip, rig: RigSpec, bone: str) -> list[float]:
    """Local angle track without wrapping (keeps continuity, e.g. a swing past straight up)."""
    parent = rig.bone(bone).parent
    return [clip.angles[bone][i] - (clip.angles[parent][i] if parent else 0.0) for i in range(clip.frame_count)]


def layer_clips(base: Clip, overlay: Clip, rig: RigSpec, bones: tuple[str, ...] | str = "upper") -> Clip:
    """``overlay`` on ``bones`` over ``base``. Length: the base's when the overlay loops, else the overlay's
    (the base loops under it if it is a loop, or holds its last frame). Frames are matched by time."""
    names = [b.name for b in rig.bones]
    mask = set(BONE_SETS[bones] if isinstance(bones, str) else bones)
    unknown = mask - set(names)
    if unknown:
        raise ValueError(f"layer bones the rig does not have: {sorted(unknown)}")
    fps = max(base.fps, overlay.fps)  # a still base pose (1 fps) must not slow a swing down
    n = (max(1, round(base.frame_count / base.fps * fps)) if overlay.loop else
         max(1, round(overlay.frame_count / overlay.fps * fps)))

    def at(clip: Clip, i: int) -> int:
        k = int(i * clip.fps / fps + 1e-6)
        return k % clip.frame_count if clip.loop else min(k, clip.frame_count - 1)

    ov_local = {b: _unwrapped_local(overlay, rig, b) for b in mask}
    angles = {b: [0.0] * n for b in names}
    order = []  # parents before children
    seen = set()

    def visit(b: str) -> None:
        p = rig.bone(b).parent
        if p and p not in seen:
            visit(p)
        if b not in seen:
            seen.add(b)
            order.append(b)

    for b in names:
        visit(b)
    for i in range(n):
        bi, oi = at(base, i), at(overlay, i)
        for b in order:
            p = rig.bone(b).parent
            if b in mask:
                angles[b][i] = (angles[p][i] if p else 0.0) + ov_local[b][oi]
            else:
                angles[b][i] = base.angles[b][bi]
    parents = {b.name: b.parent for b in rig.bones}
    local = {b: [wrap180(angles[b][i] - (angles[parents[b]][i] if parents[b] else 0.0)) for i in range(n)]
             for b in names}
    swaps = {}
    for g, track in base.swaps.items():
        arm = g.startswith(("upper_arm", "forearm", "hand"))
        src = overlay.swaps.get(g) if arm and any(x in mask for x in (g + "_near", g + "_far")) else None
        swaps[g] = [(src[at(overlay, i)] if src else (False if arm and mask & {g + "_near"} else
                                                       track[at(base, i)])) for i in range(n)]
    xs = [base.root.x[at(base, i)] for i in range(n)]
    ys = [base.root.y[at(base, i)] for i in range(n)]
    return base.model_copy(update={
        "source": f"layer({base.source} + {overlay.source} on {sorted(mask)})", "frame_count": n, "fps": fps,
        "loop": base.loop and overlay.loop, "angles": angles, "local_angles": local,
        "root": RootTrack(x=xs, y=ys), "swaps": swaps,
        "loop_travel": base.loop_travel if overlay.loop else base.loop_travel * n / base.frame_count})
