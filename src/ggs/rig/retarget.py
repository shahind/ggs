"""Retarget 3D mocap onto a side-view 2D rig.

Camera model (orthographic, no mirroring):
    V  depth axis   = horizontal direction from the far hip to the near hip (points at the camera),
                      averaged over the clip
    U  up axis      = the file's up axis (BVH default: +Y)
    X  screen right = U x V   (the character's forward direction when the near side is its right side)
Screen coordinates are in standing heights: x = forward travel, y = -(height above ground), so +y is down.
Bone angle = atan2(-dx, dy) of the projected bone vector: degrees clockwise from +y (straight down = 0).
Depth (V component) decides which limb of each near/far pair is closer to the camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ggs.rig.bvh import BvhMotion
from ggs.rig.clip import Clip, RootTrack
from ggs.rig.spec import RigSpec

_AXES = {"x": 0, "y": 1, "z": 2}
FORESHORTEN_WARN = 0.35  # warn when a bone's projected length drops below this fraction of its 3D length
TURN_WARN_DEG = 45.0  # warn when the per-frame heading deviates this far from the clip's mean heading


class RetargetError(ValueError):
    pass


@dataclass(frozen=True)
class LimbGroup:
    """A near/far limb pair, e.g. name='thigh', near=[thigh_near, shin_near, foot_near]."""

    name: str
    near: tuple[str, ...]
    far: tuple[str, ...]


def limb_groups(spec: RigSpec) -> list[LimbGroup]:
    """Near/far limb chains: a ``*_near`` bone whose parent is not ``*_near``, plus its descendants."""
    children: dict[str, list[str]] = {}
    for b in spec.bones:
        if b.parent is not None:
            children.setdefault(b.parent, []).append(b.name)
    parents = {b.name: b.parent for b in spec.bones}

    def chain(root: str) -> list[str]:
        out, stack = [], [root]
        while stack:
            cur = stack.pop(0)
            out.append(cur)
            stack[:0] = children.get(cur, [])
        return out

    groups = []
    for b in spec.bones:
        par = parents[b.name]
        if b.name.endswith("_near") and not (par or "").endswith("_near"):
            near = chain(b.name)
            far = [n[: -len("_near")] + "_far" for n in near]
            if chain(far[0]) != far:
                raise RetargetError(f"rig {spec.id!r}: far chain for {b.name!r} does not mirror the near chain")
            groups.append(LimbGroup(b.name[: -len("_near")], tuple(near), tuple(far)))
    return groups


def swap_draw_order(order: list[str], groups: list[LimbGroup], swapped: dict[str, bool]) -> list[str]:
    """Back-to-front ``order`` with the slots of every swapped near/far group exchanged."""
    order = list(order)
    for g in groups:
        if not swapped.get(g.name, False):
            continue
        near_slots = sorted(order.index(b) for b in g.near)
        far_slots = sorted(order.index(b) for b in g.far)
        near_bones = [order[i] for i in near_slots]
        far_bones = [order[i] for i in far_slots]
        for i, b in zip(far_slots, near_bones):
            order[i] = b
        for i, b in zip(near_slots, far_bones):
            order[i] = b
    return order


def draw_order_for(spec: RigSpec, swapped: dict[str, bool]) -> list[str]:
    """Back-to-front draw order of ``spec`` with the slots of every swapped near/far group exchanged."""
    return swap_draw_order(spec.draw_order, limb_groups(spec), swapped)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise RetargetError("degenerate direction vector")
    return v / n


def _wrap(deg: np.ndarray) -> np.ndarray:
    """Wrap to (-180, 180]."""
    return -((-deg + 180.0) % 360.0 - 180.0)


def _hysteresis(diff: np.ndarray, h: float) -> np.ndarray:
    state = bool(diff[0] > 0)
    out = np.zeros(diff.shape[0], dtype=bool)
    for i, d in enumerate(diff):
        if state and d < -h:
            state = False
        elif not state and d > h:
            state = True
        out[i] = state
    return out


def _is_tpose(P0: np.ndarray, lateral0: np.ndarray, motion: BvhMotion, spec: RigSpec,
              mapping: dict[str, list[str]]) -> bool:
    """True when both arms of a frame point straight out sideways (near arm to the near side)."""
    n = np.linalg.norm(lateral0)
    if n < 1e-9:
        return False
    side = lateral0 / n
    root = next(b.name for b in spec.bones if b.parent is None)
    parents = {b.name: b.parent for b in spec.bones}
    arms = [g for g in limb_groups(spec) if parents[g.near[0]] != root]
    if not arms:
        return False
    for g in arms:
        for chain, sign in ((g.near, 1.0), (g.far, -1.0)):
            a = P0[motion.index(mapping[chain[0]][0])]
            b = P0[motion.index(mapping[chain[min(1, len(chain) - 1)]][1])]
            d = b - a
            if np.linalg.norm(d) < 1e-9 or sign * float(d @ side) / np.linalg.norm(d) < 0.9:
                return False
    return True


def retarget(
    motion: BvhMotion,
    spec: RigSpec,
    profile: str,
    *,
    up_axis: str = "y",
    fps: float | None = None,
    start: int = 0,
    end: int | None = None,
    in_place: bool = False,
    swap_hysteresis: float = 0.02,
    drop_tpose_frame: bool = True,
) -> Clip:
    """Retarget ``motion`` onto ``spec`` using the joint mapping ``spec.mocap_profiles[profile]``.

    ``drop_tpose_frame``: when the range starts at frame 0 and that frame is a T-pose (both arms straight
    out sideways), skip it. The CMU BVH release prepends a T-pose frame to every clip.
    """
    if spec.facing != "right" or spec.near_side != "right":
        raise RetargetError("only rigs facing right with the near side = right are supported")
    if profile not in spec.mocap_profiles or not spec.mocap_profiles[profile]:
        avail = sorted(k for k, v in spec.mocap_profiles.items() if v)
        raise RetargetError(f"rig {spec.id!r} has no mapped mocap profile {profile!r}; available: {avail}")
    mapping = spec.mocap_profiles[profile]
    if up_axis not in _AXES:
        raise RetargetError(f"up axis must be one of x, y, z, got {up_axis!r}")

    needed = sorted({j for pair in mapping.values() for j in pair})
    have = set(motion.joint_names)
    missing = [j for j in needed if j not in have]
    if missing:
        raise RetargetError(
            f"{motion.source}: profile {profile!r} needs joints missing from the file: {missing}. "
            f"File joints: {motion.joint_names}"
        )

    n_total = motion.n_frames
    end = n_total if end is None else end
    if not (0 <= start < end <= n_total):
        raise RetargetError(f"frame range {start}:{end} is outside 0:{n_total}")

    warnings: list[str] = []
    P = motion.world_positions()[start:end]  # (F, J, 3)
    rest = motion.world_positions(rest=True)[0]
    up = np.zeros(3)
    up[_AXES[up_axis]] = 1.0

    height = float(rest[:, _AXES[up_axis]].max() - rest[:, _AXES[up_axis]].min())
    if height <= 1e-9:
        raise RetargetError(f"{motion.source}: rest pose has zero height along {up_axis!r}")

    def jpos(name: str) -> np.ndarray:
        return P[:, motion.index(name)]

    # Depth axis from far hip/shoulder -> near hip/shoulder, horizontal, averaged over the clip.
    groups = limb_groups(spec)
    lateral = np.zeros((P.shape[0], 3))
    for g in groups:
        lateral += jpos(mapping[g.near[0]][0]) - jpos(mapping[g.far[0]][0])
    lateral -= np.outer(lateral @ up, up)
    if drop_tpose_frame and start == 0 and P.shape[0] > 1 and _is_tpose(P[0], lateral[0], motion, spec,
                                                                        mapping):
        P, lateral, start = P[1:], lateral[1:], 1
        warnings.append("dropped frame 0: it is a T-pose (added by converters such as the CMU BVH release)")
    try:
        V = _unit(lateral.mean(axis=0))
    except RetargetError as exc:
        raise RetargetError(f"{motion.source}: near and far limbs coincide; cannot find the side plane") from exc
    X = np.cross(up, V)

    norms = np.linalg.norm(lateral, axis=1)
    ok = norms > 1e-9
    if ok.any():
        cosang = np.clip((lateral[ok] @ V) / norms[ok], -1.0, 1.0)
        worst = float(np.degrees(np.arccos(cosang)).max())
        if worst > TURN_WARN_DEG:
            warnings.append(
                f"character turns up to {worst:.0f} deg away from the side view; projection loses detail"
            )

    root_joint = mapping[spec.bones[[b.parent for b in spec.bones].index(None)].name][0]
    origin_x = float(jpos(root_joint)[0] @ X)
    ground = float((P @ up).min())

    sx = (P @ X - origin_x) / height  # (F, J)
    sy = -(P @ up - ground) / height
    depth = (P @ V) / height

    # Sanity check: feet should point forward. A reversed result usually means a left-handed file.
    foot_like = [b for b in mapping if b.startswith("foot")]
    if foot_like:
        fwd = np.mean([np.mean(sx[:, motion.index(mapping[b][1])] - sx[:, motion.index(mapping[b][0])])
                       for b in foot_like])
        if fwd < 0:
            warnings.append("feet point backwards on screen: the file may be left-handed or mirrored")

    F = P.shape[0]
    abs_deg: dict[str, np.ndarray] = {}
    lengths: dict[str, float] = {}
    for bone in spec.bones:
        a, b = (motion.index(j) for j in mapping[bone.name])
        dx = sx[:, b] - sx[:, a]
        dy = sy[:, b] - sy[:, a]
        len3 = np.linalg.norm(P[:, b] - P[:, a], axis=1) / height
        if float(np.median(len3)) < 1e-6:
            raise RetargetError(
                f"profile {profile!r} bone {bone.name!r} maps to coincident joints {mapping[bone.name]}; "
                "choose joints that are apart"
            )
        len2 = np.hypot(dx, dy)
        ratio = len2 / np.maximum(len3, 1e-12)
        if float(ratio.min()) < FORESHORTEN_WARN:
            warnings.append(
                f"{bone.name} points toward the camera (projected to {ratio.min():.0%} of its length)"
            )
        ang = np.degrees(np.arctan2(-dx, dy))
        valid = len2 > 1e-9
        if not valid.all():  # undefined angle: hold the last valid value
            idx = np.where(valid, np.arange(F), 0)
            np.maximum.accumulate(idx, out=idx)
            first = int(np.argmax(valid)) if valid.any() else 0
            ang = ang[idx]
            ang[: first] = ang[first]
        abs_deg[bone.name] = np.degrees(np.unwrap(np.radians(ang)))
        lengths[bone.name] = float(np.median(len3))

    root_x = sx[:, motion.index(root_joint)]
    root_y = sy[:, motion.index(root_joint)]

    swaps: dict[str, np.ndarray] = {}
    for g in groups:
        def gdepth(bones: tuple[str, ...]) -> np.ndarray:
            idx = sorted({motion.index(j) for bn in bones for j in mapping[bn]})
            return depth[:, idx].mean(axis=1)

        swaps[g.name] = _hysteresis(gdepth(g.far) - gdepth(g.near), swap_hysteresis)

    src_fps = motion.fps
    out_fps = src_fps if fps is None else float(fps)
    if out_fps <= 0:
        raise RetargetError("fps must be positive")
    if fps is not None and abs(out_fps - src_fps) > 1e-6:
        t_src = np.arange(F) / src_fps
        n_out = max(1, int(np.floor(t_src[-1] * out_fps + 1e-3)) + 1)
        t_out = np.arange(n_out) / out_fps
        abs_deg = {k: np.interp(t_out, t_src, v) for k, v in abs_deg.items()}
        root_x = np.interp(t_out, t_src, root_x)
        root_y = np.interp(t_out, t_src, root_y)
        nearest = np.clip(np.rint(t_out * src_fps).astype(int), 0, F - 1)
        swaps = {k: v[nearest] for k, v in swaps.items()}
        F = n_out

    if in_place:
        root_x = np.zeros_like(root_x)

    parents = {b.name: b.parent for b in spec.bones}
    local = {
        name: (_wrap(v) if parents[name] is None else _wrap(v - abs_deg[parents[name]]))
        for name, v in abs_deg.items()
    }

    def r(v: np.ndarray, nd: int) -> list[float]:
        return [round(float(x), nd) for x in v]

    return Clip(
        rig=spec.id,
        source=motion.source,
        profile=profile,
        fps=round(out_fps, 6),
        frame_count=F,
        in_place=in_place,
        bones=[b.name for b in spec.bones],
        bone_lengths={k: round(v, 5) for k, v in lengths.items()},
        angles={k: r(v, 3) for k, v in abs_deg.items()},
        local_angles={k: r(v, 3) for k, v in local.items()},
        root=RootTrack(x=r(root_x, 5), y=r(root_y, 5)),
        swaps={k: [bool(x) for x in v] for k, v in swaps.items()},
        warnings=warnings,
    )
