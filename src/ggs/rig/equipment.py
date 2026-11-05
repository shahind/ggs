"""Held equipment (M7): item sprites attached to a hand socket, plus the two-hand IK that brings the other hand
onto the item.

``item.yaml`` sits next to its PNG:

    format: ggs.item
    version: 1
    name: sword
    image: sword.png
    px_per_height: 900      # the art is drawn at this many pixels per standing height (sets the item's size)
    grip: [40, 210]         # pixel the near hand holds
    grip2: [40, 170]        # optional: where the far hand holds (two-hand items)
    art_angle: 180          # direction the item points in the art (180 = up, handle at the bottom)
    hold: one_hand          # one_hand | two_hand
    socket: grip_near
    hold_angle: -90         # item direction = hand direction + this (-90: forward when the hand hangs down)

Items are drawn right after their hand, so the fingers sit under the handle. In a two-hand hold the far arm is
solved with two-bone IK every frame so the far hand reaches ``grip2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ggs.rig.pose import bone_vector
from ggs.rig.spec import RigSpec


class ItemSpec(BaseModel):
    format: Literal["ggs.item"] = "ggs.item"
    version: Literal[1] = 1
    name: str
    image: str
    px_per_height: float = Field(gt=0)
    grip: tuple[float, float]
    grip2: tuple[float, float] | None = None
    art_angle: float = 180.0
    hold: Literal["one_hand", "two_hand"] = "one_hand"
    socket: str = "grip_near"
    hold_angle: float = -90.0


@dataclass
class Item:
    spec: ItemSpec
    base_dir: Path

    @property
    def image_path(self) -> Path:
        return self.base_dir / self.spec.image

    def load_image(self):
        from PIL import Image

        return Image.open(self.image_path).convert("RGBA")


def load_item(path: str | Path) -> Item:
    path = Path(path)
    if path.is_dir():
        path = path / "item.yaml"
    try:
        spec = ItemSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc
    item = Item(spec, path.parent)
    if not item.image_path.exists():
        raise ValueError(f"{path}: image {spec.image!r} not found")
    if spec.hold == "two_hand" and spec.grip2 is None:
        raise ValueError(f"{path}: a two_hand item needs grip2")
    return item


def make_item(png: str | Path, out_dir: str | Path, *, name: str, length: float, hold: str = "one_hand",
              grip_at: float = 0.12, grip2_at: float | None = None, hold_angle: float = -90.0) -> Path:
    """Item folder from a sprite drawn pointing up with the handle at the bottom (the ``held_item`` recipe draws
    them that way). ``length`` = the item's length in standing heights (a one-hand sword is about 0.45).
    The grip is on the vertical centre line, ``grip_at`` of the height up from the bottom; ``grip2`` (two-hand)
    at ``grip2_at`` (default: 20% of the item above the first grip)."""
    import numpy as np
    from PIL import Image

    img = Image.open(png).convert("RGBA")
    box = img.getchannel("A").getbbox()
    if box is None:
        raise ValueError(f"{png}: the image is empty")
    img = img.crop(box)
    a = np.asarray(img.getchannel("A")) > 127
    h = img.height

    def centre_at(frac: float) -> tuple[float, float]:
        y = min(h - 1, max(0, round(h * (1 - frac))))
        xs = np.flatnonzero(a[y])
        x = float(xs.mean()) if xs.size else img.width / 2
        return (round(x, 2), float(y))

    if grip2_at is None:
        grip2_at = min(0.95, grip_at + 0.20)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img.save(out / f"{name}.png")
    spec = ItemSpec(name=name, image=f"{name}.png", px_per_height=round(h / length, 3), grip=centre_at(grip_at),
                    grip2=centre_at(grip2_at) if hold == "two_hand" else None, hold=hold, hold_angle=hold_angle)
    path = out / "item.yaml"
    path.write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- geometry (standing heights, screen space)

def socket_point(rig: RigSpec, socket: str, pose: dict) -> tuple[tuple[float, float], str]:
    """Position of a socket in a posed skeleton ({bone: (start, end)}) and the bone it sits on."""
    s = next((x for x in rig.sockets if x.name == socket), None)
    if s is None:
        raise ValueError(f"rig {rig.id!r} has no socket {socket!r}")
    (x0, y0), (x1, y1) = pose[s.bone]
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy) or 1.0
    # offset is perpendicular, in bone lengths, along (-dy, dx): forward for the upright torso, the rig's convention
    px, py = -dy / n, dx / n
    length = n
    return ((x0 + s.t * dx + s.offset * length * px, y0 + s.t * dy + s.offset * length * py), s.bone)


def item_angle(item: ItemSpec, hand_angle: float) -> float:
    return hand_angle + item.hold_angle


def item_point(item: ItemSpec, pixel: tuple[float, float], grip_world: tuple[float, float], angle: float):
    """Where an item pixel lands (standing heights), the item's grip being at ``grip_world``."""
    th = math.radians(angle - item.art_angle)
    c, s = math.cos(th), math.sin(th)
    vx = (pixel[0] - item.grip[0]) / item.px_per_height
    vy = (pixel[1] - item.grip[1]) / item.px_per_height
    return (grip_world[0] + c * vx - s * vy, grip_world[1] + s * vx + c * vy)


def two_bone_ik(shoulder: tuple[float, float], target: tuple[float, float], l1: float, l2: float,
                bend: float = 1.0) -> tuple[float, float]:
    """Absolute angles (upper, lower) of a two-bone chain from ``shoulder`` reaching ``target`` (clamped to
    reach). ``bend`` +1 / -1 picks the elbow side (+1: elbow on the screen-clockwise side of the reach line)."""
    dx, dy = target[0] - shoulder[0], target[1] - shoulder[1]
    d = min(max(math.hypot(dx, dy), abs(l1 - l2) + 1e-6), l1 + l2 - 1e-6)
    base = math.degrees(math.atan2(-dx, dy))  # angle of the reach line, clockwise from +y
    cos_a = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    a = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
    upper = base + bend * a
    sx, sy = bone_vector(upper, l1)
    elbow = (shoulder[0] + sx, shoulder[1] + sy)
    lower = math.degrees(math.atan2(-(target[0] - elbow[0]), target[1] - elbow[1]))
    return upper, lower


def apply_two_hand(clip, rig: RigSpec, lengths: dict[str, float], item: ItemSpec, far_side: str = "far"):
    """A copy of ``clip`` whose far arm reaches the item's grip2 in every frame (two-bone IK; the far hand takes
    the near hand's angle, so both hands hold the handle the same way)."""
    from ggs.rig.master import wrap180
    from ggs.rig.pose import bone_defs, pose_bones

    if item.grip2 is None:
        return clip
    ua, fa, ha = f"upper_arm_{far_side}", f"forearm_{far_side}", f"hand_{far_side}"
    near_hand = "hand_near"
    angles = {b: list(v) for b, v in clip.angles.items()}
    bones = bone_defs(rig)
    for i in range(clip.frame_count):
        pose = pose_bones(bones, clip, i, lengths)
        grip, _ = socket_point(rig, item.socket, pose)
        hand_a = clip.angles[near_hand][i]
        ang = item_angle(item, hand_a)
        g2 = item_point(item, item.grip2, grip, ang)
        # the far hand's grip socket must land on g2: its wrist sits half a hand back along the hand direction
        hx, hy = bone_vector(hand_a, lengths[ha] * 0.5)
        wrist = (g2[0] - hx, g2[1] - hy)
        shoulder = pose[ua][0]
        up, lo = two_bone_ik(shoulder, wrist, lengths[ua], lengths[fa], bend=-1.0)
        angles[ua][i], angles[fa][i], angles[ha][i] = up, lo, hand_a
    parents = {b.name: b.parent for b in rig.bones}
    local = {b: [wrap180(angles[b][i] - (angles[parents[b]][i] if parents[b] else 0.0))
                 for i in range(clip.frame_count)] for b in clip.bones}
    return clip.model_copy(update={"angles": angles, "local_angles": local})


def attachment_for(item: Item, rig: RigSpec):
    """The renderer's view of an item (``ggs.rig.puppet.Attachment``)."""
    from ggs.rig.puppet import Attachment

    s = next((x for x in rig.sockets if x.name == item.spec.socket), None)
    if s is None:
        raise ValueError(f"rig {rig.id!r} has no socket {item.spec.socket!r}")
    return Attachment(name=item.spec.name, bone=s.bone, socket_t=s.t, socket_offset=s.offset,
                      image=item.load_image(), grip=item.spec.grip, art_angle=item.spec.art_angle,
                      px_per_height=item.spec.px_per_height, hold_angle=item.spec.hold_angle)
