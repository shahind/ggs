"""Cut-out puppet renderer: part images + bones + clip -> RGBA frames. Needs Pillow.

All resampling happens in premultiplied alpha (Pillow mode "RGBa"), so rotated and scaled part edges blend
cleanly without dark fringes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ggs.rig.clip import Clip
from ggs.rig.pose import BoneDef, pose_bones, root_scale
from ggs.rig.retarget import LimbGroup, swap_draw_order


class RenderError(ValueError):
    pass


@dataclass
class PartImage:
    """One bone's art. ``pivot`` is the bone start in image pixels; ``length_px`` is the drawn bone length."""

    image: object  # PIL.Image.Image, RGBA
    pivot: tuple[float, float]
    art_angle: float
    length_px: float
    tint: float = 1.0  # RGB multiplier, < 1 darkens (far limbs)


@dataclass
class Attachment:
    """An item held at a socket of ``bone`` (equipment). ``grip`` is the item pixel at the socket; the item points
    along bone angle + ``hold_angle``; ``px_per_height`` sets its size."""

    name: str
    bone: str
    socket_t: float
    socket_offset: float
    image: object  # PIL RGBA
    grip: tuple[float, float]
    art_angle: float
    px_per_height: float
    hold_angle: float


@dataclass
class Puppet:
    bones: list[BoneDef]
    lengths: dict[str, float]  # standing heights
    draw_order: list[str]
    groups: list[LimbGroup]
    parts: dict[str, PartImage]  # bone -> art
    attachments: list[Attachment] = field(default_factory=list)  # drawn right after their bone
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        names = {b.name for b in self.bones}
        missing = sorted(names - set(self.parts))
        if missing:
            raise RenderError(f"puppet has bones without art: {missing}")
        if sorted(self.draw_order) != sorted(names):
            raise RenderError("draw_order must list every bone exactly once")

    def _prepared(self, bone: str, height_px: float):
        """Part image tinted, premultiplied and pre-scaled close to the output size (LANCZOS)."""
        from PIL import Image

        key = (bone, round(height_px, 4))
        if key in self._cache:
            return self._cache[key]
        part = self.parts[bone]
        img = part.image.convert("RGBA")
        if part.tint != 1.0:
            r, g, b, a = img.split()
            r, g, b = (ch.point(lambda v, t=part.tint: round(v * t)) for ch in (r, g, b))
            img = Image.merge("RGBA", (r, g, b, a))
        img = img.convert("RGBa")
        target = self.lengths[bone] * height_px
        f = target / part.length_px if part.length_px > 0 else 1.0
        if abs(f - 1.0) > 1e-3:
            w, h = max(1, round(img.width * f)), max(1, round(img.height * f))
            fx, fy = w / img.width, h / img.height
            img = img.resize((w, h), Image.LANCZOS)
            pivot = (part.pivot[0] * fx, part.pivot[1] * fy)
            length = part.length_px * fx
        else:
            pivot, length = part.pivot, part.length_px
        self._cache[key] = (img, pivot, length)
        return self._cache[key]

    def render_frame(
        self,
        clip: Clip,
        frame: int,
        height_px: float,
        size: tuple[int, int],
        origin: tuple[float, float],
        in_place: bool = True,
    ):
        """One RGBA frame. ``origin`` is where the ground point under the root's start position lands."""
        from PIL import Image

        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        for _, layer in self.render_layers(clip, frame, height_px, size, origin, in_place):
            canvas.alpha_composite(layer)
        return canvas

    def render_layers(
        self,
        clip: Clip,
        frame: int,
        height_px: float,
        size: tuple[int, int],
        origin: tuple[float, float],
        in_place: bool = True,
    ) -> list[tuple[str, object]]:
        """(bone, RGBA layer) for every bone, back to front. Compositing them in order gives ``render_frame``."""
        from PIL import Image

        pose = pose_bones(
            self.bones, clip, frame, self.lengths, root_scale(self.bones, clip, self.lengths), in_place
        )
        swapped = {g: v[frame] for g, v in clip.swaps.items()}
        layers = []
        for name in swap_draw_order(self.draw_order, self.groups, swapped):
            img, pivot, art_len = self._prepared(name, height_px)
            (x0, y0), _ = pose[name]
            sx, sy = origin[0] + x0 * height_px, origin[1] + y0 * height_px
            s = (self.lengths[name] * height_px) / art_len if art_len > 0 else 1.0
            th = math.radians(clip.angles[name][frame] - self.parts[name].art_angle)
            c, sn = math.cos(th) / s, math.sin(th) / s
            # inverse map: output p -> art q = pivot + R(-th) (p - start) / s
            a, b, d, e = c, sn, -sn, c
            data = (a, b, pivot[0] - a * sx - b * sy, d, e, pivot[1] - d * sx - e * sy)
            layer = img.transform(size, Image.AFFINE, data, resample=Image.BICUBIC)
            layers.append((name, layer.convert("RGBA")))
            for att in self.attachments:
                if att.bone == name:
                    layers.append((f"item:{att.name}", self._attachment_layer(att, pose[name],
                                                                             clip.angles[name][frame],
                                                                             height_px, size, origin)))
        return layers

    def _attachment_layer(self, att: Attachment, bone_pose, bone_angle: float, height_px: float, size, origin):
        from PIL import Image

        (x0, y0), (x1, y1) = bone_pose
        dx, dy = x1 - x0, y1 - y0
        gx = x0 + att.socket_t * dx + att.socket_offset * -dy
        gy = y0 + att.socket_t * dy + att.socket_offset * dx
        sx, sy = origin[0] + gx * height_px, origin[1] + gy * height_px
        key = ("item", att.name, round(height_px, 4))
        if key not in self._cache:
            f = height_px / att.px_per_height
            img = att.image.convert("RGBa")
            w, h = max(1, round(img.width * f)), max(1, round(img.height * f))
            img = img.resize((w, h), Image.LANCZOS)
            self._cache[key] = (img, (att.grip[0] * w / att.image.width, att.grip[1] * h / att.image.height))
        img, pivot = self._cache[key]
        th = math.radians(bone_angle + att.hold_angle - att.art_angle)
        c, sn = math.cos(th), math.sin(th)
        a, b, d, e = c, sn, -sn, c
        data = (a, b, pivot[0] - a * sx - b * sy, d, e, pivot[1] - d * sx - e * sy)
        return img.transform(size, Image.AFFINE, data, resample=Image.BICUBIC).convert("RGBA")

    def render_clip(
        self, clip: Clip, height_px: int, padding: int = 2, in_place: bool = True
    ) -> tuple[list, tuple[int, int]]:
        """All frames cropped to one shared box. Returns (frames, anchor): anchor = sprite pivot in frame
        pixels, x under the root and y at the clip's ground contact."""
        reach = max((max(a.image.size) / a.px_per_height for a in self.attachments), default=0.0)
        w, h = int((2.0 + 2 * reach) * height_px), int((1.7 + reach) * height_px)
        origin = (w / 2, (1.35 + reach) * height_px)  # room for items held up high or out in front
        frames = [self.render_frame(clip, i, height_px, (w, h), origin, in_place)
                  for i in range(clip.frame_count)]
        box = None
        for im in frames:
            bb = im.getchannel("A").getbbox()
            if bb is None:
                continue
            if bb[0] == 0 or bb[1] == 0 or bb[2] == w or bb[3] == h:
                raise RenderError("character leaves the render canvas; use an in-place clip")
            box = bb if box is None else (min(box[0], bb[0]), min(box[1], bb[1]),
                                          max(box[2], bb[2]), max(box[3], bb[3]))
        if box is None:
            raise RenderError("every frame is empty")
        x0, y0 = max(0, box[0] - padding), max(0, box[1] - padding)
        x1, y1 = min(w, box[2] + padding), min(h, box[3] + padding)
        cropped = [im.crop((x0, y0, x1, y1)) for im in frames]
        self.last_root = (origin[0] - x0, origin[1] - y0)  # where the root sits in the cropped frames
        # Pivot: x under the root, y at the lowest visible pixel of the whole clip (ground contact). The
        # cut-out legs hang from the hips pivot, so the mocap ground line is a few pixels off.
        return cropped, (round(origin[0] - x0), box[3] - y0)


def frame_durations_ms(n: int, fps: float) -> list[int]:
    """Integer per-frame durations whose running total stays within 1 ms of the exact timing."""
    edges = [round(i * 1000.0 / fps) for i in range(n + 1)]
    return [max(1, edges[i + 1] - edges[i]) for i in range(n)]


def puppet_from_parts(
    bones: Sequence[BoneDef],
    lengths: dict[str, float],
    draw_order: list[str],
    groups: list[LimbGroup],
    parts: dict[str, PartImage],
) -> Puppet:
    return Puppet(list(bones), dict(lengths), list(draw_order), list(groups), dict(parts))
