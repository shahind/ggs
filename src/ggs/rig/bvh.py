"""BVH motion-capture parser and forward kinematics.

Parses the HIERARCHY and MOTION sections of a .bvh file and computes world-space joint positions for every
frame. End Sites become pseudo-joints named ``<parent>_End`` so a mocap profile can point at a limb tip.

Math (column vectors, per frame):
    local_rot  = R(ch1) @ R(ch2) @ R(ch3)        rotation channels applied in the order listed
    world_rot  = parent_world_rot @ local_rot
    world_pos  = parent_world_pos + parent_world_rot @ (offset + translation_channels)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROT = ("Xrotation", "Yrotation", "Zrotation")
_POS = ("Xposition", "Yposition", "Zposition")
_VALID_CHANNELS = set(_ROT) | set(_POS)


@dataclass
class Joint:
    name: str
    parent: int | None
    offset: np.ndarray  # (3,)
    channels: list[str] = field(default_factory=list)
    channel_start: int = 0  # column of the first channel in the motion table
    is_end_site: bool = False


@dataclass
class BvhMotion:
    joints: list[Joint]
    frame_time: float
    frames: np.ndarray  # (n_frames, n_channels) raw channel values
    source: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]

    def index(self, name: str) -> int:
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        raise KeyError(name)

    def world_positions(self, rest: bool = False) -> np.ndarray:
        """World positions of every joint, shape (n_frames, n_joints, 3).

        ``rest=True`` ignores all channels and returns the offset-only rest pose, shape (1, n_joints, 3).
        """
        n = 1 if rest else self.n_frames
        pos = np.zeros((n, len(self.joints), 3))
        rot = np.zeros((n, len(self.joints), 3, 3))
        for i, j in enumerate(self.joints):
            local_t = np.broadcast_to(j.offset, (n, 3)).copy()
            local_r = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
            if not rest:
                for k, ch in enumerate(j.channels):
                    col = self.frames[:, j.channel_start + k]
                    if ch in _POS:
                        local_t[:, _POS.index(ch)] += col
                    else:
                        local_r = local_r @ _axis_rot(_ROT.index(ch), np.radians(col))
            if j.parent is None:
                pos[:, i] = local_t
                rot[:, i] = local_r
            else:
                prot = rot[:, j.parent]
                pos[:, i] = pos[:, j.parent] + np.einsum("fij,fj->fi", prot, local_t)
                rot[:, i] = prot @ local_r
        return pos


def _axis_rot(axis: int, angle: np.ndarray) -> np.ndarray:
    """Right-handed rotation matrices about X/Y/Z for an array of angles (radians), shape (n, 3, 3)."""
    c, s = np.cos(angle), np.sin(angle)
    m = np.zeros((angle.shape[0], 3, 3))
    a, b = [(1, 2), (2, 0), (0, 1)][axis]  # the two axes that rotate into each other
    m[:, axis, axis] = 1.0
    m[:, a, a] = c
    m[:, a, b] = -s
    m[:, b, a] = s
    m[:, b, b] = c
    return m


class BvhError(ValueError):
    pass


def parse_bvh(text: str, source: str = "<string>") -> BvhMotion:
    tokens = text.split()
    pos = 0

    def fail(msg: str) -> BvhError:
        return BvhError(f"{source}: {msg} (token {pos})")

    def take() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise fail("unexpected end of file")
        tok = tokens[pos]
        pos += 1
        return tok

    def expect(word: str) -> None:
        tok = take()
        if tok != word:
            raise fail(f"expected {word!r}, got {tok!r}")

    def floats(k: int) -> np.ndarray:
        try:
            return np.array([float(take()) for _ in range(k)])
        except ValueError as exc:
            raise fail(f"bad number: {exc}") from exc

    expect("HIERARCHY")
    joints: list[Joint] = []
    n_channels = 0

    def parse_joint(parent: int | None, end_site: bool) -> None:
        nonlocal n_channels
        if end_site:
            expect("Site")
            if parent is None:
                raise fail("End Site without a parent joint")
            name = f"{joints[parent].name}_End"
        else:
            name = take()
        expect("{")
        expect("OFFSET")
        joint = Joint(name=name, parent=parent, offset=floats(3), is_end_site=end_site)
        idx = len(joints)
        joints.append(joint)
        while True:
            tok = take()
            if tok == "}":
                break
            if tok == "CHANNELS":
                if end_site:
                    raise fail(f"End Site of {joints[parent].name!r} has channels")
                try:
                    k = int(take())
                except ValueError as exc:
                    raise fail("CHANNELS count is not an integer") from exc
                chans = [take() for _ in range(k)]
                bad = [c for c in chans if c not in _VALID_CHANNELS]
                if bad:
                    raise fail(f"joint {name!r} has unknown channels {bad}")
                joint.channels = chans
                joint.channel_start = n_channels
                n_channels += k
            elif tok == "JOINT":
                parse_joint(idx, end_site=False)
            elif tok == "End":
                parse_joint(idx, end_site=True)
            else:
                raise fail(f"unexpected token {tok!r} in joint {name!r}")

    expect("ROOT")
    parse_joint(None, end_site=False)
    if pos < len(tokens) and tokens[pos] == "ROOT":
        raise fail("multiple ROOT skeletons are not supported")

    names = [j.name for j in joints]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise fail(f"duplicate joint names {dupes}")

    expect("MOTION")
    expect("Frames:")
    try:
        n_frames = int(take())
    except ValueError as exc:
        raise fail("Frames: is not an integer") from exc
    expect("Frame")
    expect("Time:")
    frame_time = float(floats(1)[0])
    if frame_time <= 0:
        raise fail("Frame Time must be positive")

    values = tokens[pos:]
    if len(values) != n_frames * n_channels:
        raise BvhError(
            f"{source}: expected {n_frames} frames x {n_channels} channels = {n_frames * n_channels} "
            f"values, found {len(values)}"
        )
    try:
        frames = np.array(values, dtype=float).reshape(n_frames, n_channels)
    except ValueError as exc:
        raise BvhError(f"{source}: bad number in MOTION section: {exc}") from exc
    if n_frames == 0:
        raise BvhError(f"{source}: no frames")
    return BvhMotion(joints=joints, frame_time=frame_time, frames=frames, source=source)


def load_bvh(path: str | Path) -> BvhMotion:
    path = Path(path)
    return parse_bvh(path.read_text(encoding="utf-8", errors="replace"), source=str(path))
