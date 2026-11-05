"""Pose presets per view, for the body & pose editor and character generation. Absolute bone angles in degrees,
clockwise from straight down (screen space). A pose names only some bones; the rest take the view's rest pose."""

from __future__ import annotations

from ggs.rig.master import MASTER_POSE, SPREAD_POSE
from ggs.rig.spec import RigSpec

RIGS = {"side": "humanoid_side_v1", "front": "humanoid_front_v1"}

FRONT_REST = {"pelvis": 180.0, "torso": 180.0, "head": 180.0,
              "upper_arm_l": 10.0, "forearm_l": 6.0, "hand_l": 4.0,
              "upper_arm_r": -10.0, "forearm_r": -6.0, "hand_r": -4.0,
              "thigh_l": 3.0, "shin_l": 2.0, "foot_l": 0.0, "thigh_r": -3.0, "shin_r": -2.0, "foot_r": 0.0}
SIDE_REST = {"pelvis": 180.0, "torso": 180.0, "head": 180.0,
             "upper_arm_near": -4.0, "forearm_near": -12.0, "hand_near": -12.0,
             "upper_arm_far": 6.0, "forearm_far": -2.0, "hand_far": -2.0,
             "thigh_near": -3.0, "shin_near": 0.0, "foot_near": -90.0,
             "thigh_far": 3.0, "shin_far": 3.0, "foot_far": -90.0}

POSES: dict[str, dict[str, dict[str, float]]] = {
    "front": {
        "t_pose": {**FRONT_REST, "upper_arm_l": 90.0, "forearm_l": 90.0, "hand_l": 90.0,
                   "upper_arm_r": -90.0, "forearm_r": -90.0, "hand_r": -90.0,
                   "thigh_l": 5.0, "shin_l": 5.0, "thigh_r": -5.0, "shin_r": -5.0},
        "a_pose": {**FRONT_REST, "upper_arm_l": 45.0, "forearm_l": 45.0, "hand_l": 45.0,
                   "upper_arm_r": -45.0, "forearm_r": -45.0, "hand_r": -45.0,
                   "thigh_l": 6.0, "shin_l": 6.0, "thigh_r": -6.0, "shin_r": -6.0},
        "idle": dict(FRONT_REST),
    },
    "side": {
        "idle": dict(SIDE_REST),
        "walk": dict(MASTER_POSE),
        "spread": dict(SPREAD_POSE),
    },
}
LABELS = {"t_pose": "T-pose", "a_pose": "A-pose", "idle": "Standing", "walk": "Walking", "spread": "Arms out"}
DEFAULT = {"front": "a_pose", "side": "idle"}


def rig_for(view: str) -> RigSpec:
    from ggs.paths import rigs_dir
    from ggs.rig.spec import load_rig_spec

    if view not in RIGS:
        raise ValueError(f"unknown view {view!r}; known: {sorted(RIGS)}")
    return load_rig_spec(rigs_dir() / f"{RIGS[view]}.yaml")


def pose_angles(view: str, pose: str | dict | None) -> dict[str, float]:
    """A preset name, a full/partial angle dict (filled from the view's rest pose), or None (the default)."""
    rest = FRONT_REST if view == "front" else SIDE_REST
    if pose is None:
        pose = DEFAULT[view]
    if isinstance(pose, str):
        if pose not in POSES[view]:
            raise ValueError(f"unknown {view} pose {pose!r}; known: {sorted(POSES[view])}")
        return dict(POSES[view][pose])
    return {**rest, **{k: float(v) for k, v in pose.items()}}
