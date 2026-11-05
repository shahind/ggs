"""Master pose, pose guide and the master.json that ties a master image to known joints. Needs Pillow + numpy.

GGS does not detect joints in a drawing. It renders a *pose guide* (the programmer-art puppet in the fixed
master pose), the generator redraws that guide as the user's character, and the joints and part masks of the
guide carry over to the master. See docs/PART_CUTTER.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ggs.rig.clip import Clip, RootTrack
from ggs.rig.pose import bone_defs, pose_bones
from ggs.rig.spec import RigSpec

# Absolute bone angles (degrees clockwise from straight down). Side view facing right: the near arm hangs a
# little forward with the elbow bent, the legs take a small stride so near and far limbs separate.
MASTER_POSE: dict[str, float] = {
    "pelvis": 180.0, "torso": 180.0, "head": 180.0,
    "upper_arm_near": -16.0, "forearm_near": -48.0, "hand_near": -48.0,
    "upper_arm_far": 18.0, "forearm_far": -6.0, "hand_far": -6.0,
    "thigh_near": -14.0, "shin_near": 4.0, "foot_near": -90.0,
    "thigh_far": 14.0, "shin_far": 20.0, "foot_far": -90.0,
}

# Cut-friendly master pose: the near arm held forward (it only meets the torso at the shoulder), the far arm
# swung back so it shows behind the body, and a wide stride so the legs do not cover each other. Less is hidden,
# so the cutter invents less.
SPREAD_POSE: dict[str, float] = {
    "pelvis": 180.0, "torso": 180.0, "head": 180.0,
    "upper_arm_near": -75.0, "forearm_near": -95.0, "hand_near": -95.0,
    "upper_arm_far": 55.0, "forearm_far": 40.0, "hand_far": 40.0,
    "thigh_near": -32.0, "shin_near": -8.0, "foot_near": -90.0,
    "thigh_far": 30.0, "shin_far": 30.0, "foot_far": -90.0,
}
MASTER_POSES: dict[str, dict[str, float]] = {"walk": MASTER_POSE, "spread": SPREAD_POSE}

# Poses far from the master pose, used by the validator to look for joint gaps (overrides of MASTER_POSE).
TEST_POSES: dict[str, dict[str, float]] = {
    "reach": {"upper_arm_near": -100.0, "forearm_near": -100.0, "hand_near": -100.0,
              "upper_arm_far": -80.0, "forearm_far": -80.0, "hand_far": -80.0},
    "bend": {"upper_arm_near": 15.0, "forearm_near": -95.0, "hand_near": -95.0,
             "thigh_near": -80.0, "shin_near": 10.0},
    "crouch": {"torso": 200.0, "head": 190.0, "thigh_near": -70.0, "shin_near": 30.0, "thigh_far": -55.0,
               "shin_far": 40.0, "upper_arm_near": -40.0, "forearm_near": -110.0, "hand_near": -110.0},
}


def wrap180(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def pose_clip(rig: RigSpec, angles: dict[str, float], lengths: dict[str, float], source: str = "pose") -> Clip:
    """A one-frame clip holding a still pose (root at the origin)."""
    parents = {b.name: b.parent for b in rig.bones}
    names = [b.name for b in rig.bones]
    local = {n: [wrap180(angles[n] - (angles[parents[n]] if parents[n] else 0.0))] for n in names}
    return Clip(rig=rig.id, source=source, profile="pose", fps=1.0, frame_count=1, in_place=True,
                bones=names, bone_lengths={n: lengths[n] for n in names}, angles={n: [angles[n]] for n in names},
                local_angles=local, root=RootTrack(x=[0.0], y=[0.0]))


def bone_angle(start: tuple[float, float], end: tuple[float, float]) -> float:
    """Angle of a screen-space bone vector, clockwise from +y (the inverse of ``bone_vector``)."""
    return math.degrees(math.atan2(-(end[0] - start[0]), end[1] - start[1]))


class MasterSpec(BaseModel):
    """master.json: a master image plus the guide it was drawn from."""

    format: Literal["ggs.master"] = "ggs.master"
    version: Literal[1] = 1
    rig: str
    image: str = "master.png"
    guide: str = "guide.png"
    labels: str = "guide_labels.png"  # L image: 0 = background, i + 1 = rig.bones[i]
    size: tuple[int, int]
    height_px: float = Field(gt=0)  # pixels per standing height in the master
    origin: tuple[float, float]  # pixel position of the root (pelvis start)
    angles: dict[str, float]
    joints: dict[str, tuple[tuple[float, float], tuple[float, float]]]  # bone -> (start, end) pixels
    prompt: str = ""
    seed: int | None = None
    model: str = ""
    body: dict = Field(default_factory=dict)  # the guide's body shape (ggs.rig.body.BodyShape fields)


@dataclass
class Guide:
    image: object  # PIL RGBA
    labels: object  # numpy int16, -1 = background, else index into rig.bones
    spec: MasterSpec


def render_guide(rig: RigSpec, puppet, lengths: dict[str, float], size: int = 1024, fill: float = 0.78,
                 angles: dict[str, float] | None = None) -> Guide:
    """Render ``puppet`` in the master pose, centred, ``fill`` x ``size`` pixels per standing height."""
    import numpy as np
    from PIL import Image

    angles = dict(angles or MASTER_POSE)
    clip = pose_clip(rig, angles, lengths, "master_pose")
    height_px = fill * size
    pose = pose_bones(bone_defs(rig), clip, 0, lengths)
    xs = [p[0] for se in pose.values() for p in se]
    ys = [p[1] for se in pose.values() for p in se]
    # head end is the top of the skull; feet add about half a foot width below the lowest joint
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys) + 0.03) / 2
    origin = (size / 2 - cx * height_px, size / 2 - cy * height_px)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    labels = np.full((size, size), -1, dtype=np.int16)
    index = {b.name: i for i, b in enumerate(rig.bones)}
    for name, layer in puppet.render_layers(clip, 0, height_px, (size, size), origin):
        canvas.alpha_composite(layer)
        labels[np.asarray(layer.getchannel("A")) > 127] = index[name]
    joints = {n: ((origin[0] + s[0] * height_px, origin[1] + s[1] * height_px),
                  (origin[0] + e[0] * height_px, origin[1] + e[1] * height_px)) for n, (s, e) in pose.items()}
    spec = MasterSpec(rig=rig.id, size=(size, size), height_px=height_px, origin=origin, angles=angles,
                      joints=joints)
    return Guide(canvas, labels, spec)


def save_master(out_dir: str | Path, spec: MasterSpec, master, guide: Guide) -> Path:
    """Write master.png, guide.png, guide_labels.png and master.json into ``out_dir``."""
    import numpy as np
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    master.save(out / spec.image)
    guide.image.save(out / spec.guide)
    Image.fromarray((guide.labels + 1).astype(np.uint8), "L").save(out / spec.labels)
    path = out / "master.json"
    path.write_text(json.dumps(spec.model_dump(), indent=1), encoding="utf-8")
    return path


def load_master(path: str | Path):
    """(MasterSpec, master RGBA array, guide labels array) from a master.json."""
    import numpy as np
    from PIL import Image

    path = Path(path)
    try:
        spec = MasterSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc
    master = np.asarray(Image.open(path.parent / spec.image).convert("RGBA"))
    labels = np.asarray(Image.open(path.parent / spec.labels).convert("L")).astype(np.int16) - 1
    if master.shape[:2] != labels.shape or master.shape[1::-1] != tuple(spec.size):
        raise ValueError(f"{path}: master, labels and size disagree ({master.shape}, {labels.shape}, {spec.size})")
    return spec, master, labels
