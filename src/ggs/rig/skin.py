"""Warp animation: bend the whole drawing with its skeleton (mesh skinning), no cutting.

For characters without loose clothes or big props this looks better than the cut-out puppet: the drawing stays in
one piece, elbows and knees bend smoothly, and nothing has to be redrawn.

Built from a drawing whose joints are known (``ggs.rig.joints``):
- **Layers** back to front: the far arm and leg, the body (pelvis, torso, head), the near leg and arm. A layer's
  pixels are the ones its bones own; where something in front hid it (the torso behind the near arm, the far leg's
  top behind the skirt) the hidden area is filled from the layer's own colours, so it can show when the limb
  moves away.
- **Mesh**: a triangle grid over each layer. Every vertex is bound to the nearest bones of its layer and the bone
  the layer hangs from (the torso for an arm), weighted by distance scaled by bone width, so joints blend.
- **Frames**: forward kinematics of the clip with the drawing's own bone lengths; each bone's move from the
  drawing's pose to the frame's pose moves its vertices (linear blend skinning); triangles are drawn layer by layer
  (near and far swap when the clip says the limbs crossed).

Stored as ``skin.json`` + ``skin.png`` (+ one filled PNG per layer) so exports run without the drawing model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ggs.rig.spec import RigSpec

Point = tuple[float, float]
Joints = dict[str, tuple[Point, Point]]
GRID = 1 / 70  # mesh step in standing heights ("normal" detail)
DETAIL = {"coarse": 1 / 40, "normal": 1 / 70, "fine": 1 / 110}  # mesh subdivision levels
WEIGHT_POWER = 4.0
SKIRT_SHARE = 0.15  # cloth_share above this: the legs are drawn behind the body
SKIRT_SWAY = 0.4  # how much of a leg's swing the skirt takes
SMOOTH = 0.02  # owner smoothing window, standing heights
REGION_MAJORITY = 0.6  # a colour region this much in one layer goes wholly to that layer


class SkinError(ValueError):
    pass


@dataclass
class Layer:
    name: str
    bones: list[str]
    image: np.ndarray  # premultiplied RGBA float32, full drawing size
    verts: np.ndarray  # (N, 2) x, y pixels in the drawing
    tris: np.ndarray  # (M, 3) vertex indices
    weights: np.ndarray  # (N, B) over SkinRig.bones
    tri_bone: np.ndarray  # (M,) index into SkinRig.bones (draw order inside the layer)


def legs_behind(order: list[str]) -> list[str]:
    """A draw order with both legs behind everything else (a skirt or long coat hangs over them)."""
    legs = [b for b in order if b.split("_")[0] in ("thigh", "shin", "foot")]
    return legs + [b for b in order if b not in legs]


LOOSE = ("skirt", "hair_back", "cape")


def layer_bones(rig: RigSpec) -> list[tuple[str, list[str]]]:
    """(layer name, bones) for the side rig: far limbs, body, near limbs (rig draw order decides the order)."""
    from ggs.rig.retarget import limb_groups

    groups = limb_groups(rig)
    extras = [b.name for b in rig.bones if b.name in LOOSE]
    taken = {b for g in groups for b in (*g.near, *g.far)} | set(extras)
    out = [("body", [b.name for b in rig.bones if b.name not in taken])]
    for g in groups:
        out += [(f"{g.name}_far", list(g.far)), (f"{g.name}_near", list(g.near))]
    out += [(x, [x]) for x in extras]  # skirt, back hair, cape: their own layers, they bend on their own
    return out


def _seg_dist(xy: np.ndarray, p0: Point, p1: Point) -> np.ndarray:
    d = np.array(p1, np.float32) - np.array(p0, np.float32)
    n2 = float(d @ d)
    rel = xy - np.array(p0, np.float32)
    t = np.clip(rel @ d / n2, 0.0, 1.0) if n2 > 0 else np.zeros(len(xy), np.float32)
    return np.linalg.norm(rel - t[:, None] * d[None, :], axis=1)


def _capsule(shape, p0: Point, p1: Point, r: float) -> np.ndarray:
    import cv2

    m = np.zeros(shape, np.uint8)
    cv2.line(m, (round(p0[0]), round(p0[1])), (round(p1[0]), round(p1[1])), 1, max(1, round(2 * r)))
    cv2.circle(m, (round(p0[0]), round(p0[1])), max(1, round(r)), 1, -1)
    cv2.circle(m, (round(p1[0]), round(p1[1])), max(1, round(r)), 1, -1)
    return m.astype(bool)


class SkinRig:
    def __init__(self, rig: RigSpec, image: np.ndarray, joints: Joints, labels: np.ndarray, widths: dict[str, float],
                 height_px: float, grid: float = GRID):
        from ggs.rig.cutter import fill_from

        if image.shape[:2] != labels.shape:
            raise SkinError("drawing and labels differ in size")
        self.rig, self.joints, self.widths, self.height_px = rig, joints, widths, float(height_px)
        self.grid = grid
        self.source = image  # the drawing (uint8 RGBA) the rig was made from
        self.bones = [b.name for b in rig.bones]
        self.index = {n: i for i, n in enumerate(self.bones)}
        self.parent = {b.name: b.parent for b in rig.bones}
        self.size = (image.shape[1], image.shape[0])
        self._mips: dict[int, dict[float, np.ndarray]] = {}
        self._expanded: dict[int, tuple] = {}
        self.layout: dict | None = None  # the character's body layout (set by the loader)
        self.parse: np.ndarray | None = None  # body parsing of the picture (ggs.rig.parsing), when made
        from ggs.rig.joints import cloth_share

        # a skirt or long coat hangs over the legs: draw both legs behind the body (and fill their tops under it)
        self.skirt = "skirt" in self.bones or cloth_share(labels, rig, joints, widths) >= SKIRT_SHARE
        self.draw_order = self._base_order()
        order = {n: k for k, n in enumerate(self.draw_order)}
        rgba = image.astype(np.float32) / 255.0
        alpha = rgba[..., 3]
        from ggs.rig.joints import ink_mask

        ink = ink_mask(image[..., :3], image[..., 3] > 5)  # outlines: never a fill colour
        self.layers: list[Layer] = []
        for name, bones in layer_bones(rig):
            ids = [self.index[b] for b in bones]
            own = np.isin(labels, ids) & (alpha > 0.02)
            if not own.any():
                continue
            front = np.isin(labels, [self.index[b] for b in self.bones
                                     if order[b] > max(order[x] for x in bones)])
            # where layers in front hide this one: along its own bones, as wide as the bone
            hidden = np.zeros_like(own)
            for b in bones:
                hidden |= _capsule(own.shape, *joints[b], 0.9 * widths[b])
            fill = hidden & front & ~own
            known = own & ~ink if (own & ~ink).sum() > 50 else own
            if name == "body":  # blend in from the surrounding colours
                rgb = fill_from(rgba[..., :3] * 255.0, known, fill | (own & ~known)) / 255.0 if fill.any() \
                    else rgba[..., :3]
            else:  # a limb: the hidden part gets the limb's own typical colour (no shadow or outline streaks)
                rgb = rgba[..., :3].copy()
                for b in bones:
                    mine = known & (labels == self.index[b])
                    src = mine if mine.sum() > 30 else known
                    if src.any():
                        part = fill & _capsule(own.shape, *joints[b], 0.9 * widths[b])
                        rgb[part] = np.median(rgba[..., :3][src], axis=0)
            rgb = np.where(own[..., None], rgba[..., :3], rgb)
            a = np.where(own, alpha, np.where(fill, 1.0, 0.0)).astype(np.float32)
            layer_img = np.dstack([rgb * a[..., None], a]).astype(np.float32)
            self.layers.append(self._mesh(name, bones, layer_img, own | fill))

    # ------------------------------------------------------------------ building

    def _base_order(self) -> list[str]:
        return legs_behind(self.rig.draw_order) if self.skirt else list(self.rig.draw_order)

    def _candidates(self, bones: list[str]) -> list[str]:
        out = list(bones)
        root = next(b for b in bones if self.parent[b] not in bones)
        if self.parent[root] is not None:
            out.append(self.parent[root])
        skirt_layer = bones == ["skirt"] or (self.parent[root] is None and "skirt" not in self.bones)
        if self.skirt and skirt_layer:  # the skirt sways with the thighs too
            out += [b for b in self.bones if b.startswith("thigh")]
        return out

    def _mesh(self, name: str, bones: list[str], img: np.ndarray, mask: np.ndarray) -> Layer:
        import cv2

        step = max(4, round(self.grid * self.height_px))
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        ys, xs = np.nonzero(grown)
        x0, y0 = (xs.min() // step) * step, (ys.min() // step) * step
        nx, ny = (xs.max() - x0) // step + 2, (ys.max() - y0) // step + 2
        cells = np.zeros((ny - 1, nx - 1), bool)
        cy, cx = (ys - y0) // step, (xs - x0) // step
        cells[np.clip(cy, 0, ny - 2), np.clip(cx, 0, nx - 2)] = True
        vid = -np.ones((ny, nx), np.int64)
        verts, tris = [], []

        def v(i: int, j: int) -> int:
            if vid[i, j] < 0:
                vid[i, j] = len(verts)
                verts.append((x0 + j * step, y0 + i * step))
            return int(vid[i, j])

        for i, j in zip(*np.nonzero(cells)):
            a, b, c, d = v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j)
            tris += [(a, b, c), (a, c, d)]
        verts = np.array(verts, np.float32)
        tris = np.array(tris, np.int64)
        cand = self._candidates(bones)
        dist = np.stack([_seg_dist(verts, *self.joints[b]) / self.widths[b] for b in cand], axis=1)
        w = 1.0 / (dist + 0.15) ** WEIGHT_POWER
        # a vertex only blends between its nearest bone and that bone's parent/children (no arm-to-leg pulls)
        near = np.argmin(dist, axis=1)
        allowed = np.zeros_like(w, bool)
        for k, b in enumerate(cand):
            rel = [m for m, o in enumerate(cand) if o == b or self.parent.get(o) == b or self.parent.get(b) == o]
            allowed[np.ix_(near == k, rel)] = True
        w = np.where(allowed, w, 0.0)
        w /= w.sum(axis=1, keepdims=True)
        extra = [k for k, b in enumerate(cand) if b not in bones and self.parent.get(b) in bones]
        if extra:  # the skirt sways with the legs, but mostly hangs from the waist
            rigid = w.copy()
            rigid[:, extra] = 0.0
            ok = rigid.sum(axis=1, keepdims=True) > 1e-6
            rigid = np.where(ok, rigid / np.maximum(rigid.sum(axis=1, keepdims=True), 1e-6), w)
            w = SKIRT_SWAY * w + (1.0 - SKIRT_SWAY) * rigid
        full = np.zeros((len(verts), len(self.bones)), np.float32)
        full[:, [self.index[b] for b in cand]] = w
        centres = verts[tris].mean(axis=1)
        own_d = np.stack([_seg_dist(centres, *self.joints[b]) / self.widths[b] for b in bones], axis=1)
        tri_bone = np.array([self.index[bones[k]] for k in np.argmin(own_d, axis=1)], np.int64)
        return Layer(name, bones, img, verts, tris, full, tri_bone)

    # ------------------------------------------------------------------ posing

    def _rest(self) -> tuple[np.ndarray, np.ndarray]:
        s = np.array([self.joints[b][0] for b in self.bones], np.float32)
        e = np.array([self.joints[b][1] for b in self.bones], np.float32)
        return s, np.arctan2(e[:, 1] - s[:, 1], e[:, 0] - s[:, 0])

    def lengths(self) -> dict[str, float]:
        """Bone lengths of the drawing, in standing heights (what the clip is played with)."""
        return {b: max(math.dist(*self.joints[b]), 1e-3) / self.height_px for b in self.bones}

    def prepare(self, clip):
        """``clip`` with tracks for every bone of this rig (a layout's chest, neck, skirt, hair, cape), and the far
        arm and leg kept behind the body."""
        key = id(clip)
        if key not in self._expanded:
            from ggs.rig.layout import expand_clip, far_stays_back

            c = far_stays_back(clip)
            if not all(b in c.angles for b in self.bones) or (self.layout and self.layout.get("hands") ==
                                                                 "with_forearm"):
                c = expand_clip(c, self.rig, self.layout)
            self._expanded[key] = (clip, c)
        return self._expanded[key][1]

    def rest_angles(self) -> dict[str, float]:
        """World angles of the drawing's own pose (where a lead-in starts)."""
        from ggs.rig.master import bone_angle

        return {b: bone_angle(*self.joints[b]) for b in self.bones}

    def pose(self, clip, frame: int, in_place: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(start points, angles) of every bone at ``frame``, in drawing pixels (root at the origin)."""
        from ggs.rig.pose import bone_defs, pose_bones, root_scale

        bones = bone_defs(self.rig)
        lengths = self.lengths()
        clip = self.prepare(clip)
        p = pose_bones(bones, clip, frame, lengths, root_scale(bones, clip, lengths), in_place)
        s = np.array([p[b][0] for b in self.bones], np.float32) * self.height_px
        e = np.array([p[b][1] for b in self.bones], np.float32) * self.height_px
        return s, np.arctan2(e[:, 1] - s[:, 1], e[:, 0] - s[:, 0])

    def deform(self, layer: Layer, start: np.ndarray, angle: np.ndarray) -> np.ndarray:
        s0, a0 = self._rest()
        da = angle - a0
        c, sn = np.cos(da), np.sin(da)
        rel = layer.verts[:, None, :] - s0[None, :, :]  # (N, B, 2)
        x = c[None, :] * rel[..., 0] - sn[None, :] * rel[..., 1] + start[None, :, 0]
        y = sn[None, :] * rel[..., 0] + c[None, :] * rel[..., 1] + start[None, :, 1]
        return np.stack([(layer.weights * x).sum(1), (layer.weights * y).sum(1)], axis=1)

    def _order(self, clip, frame: int) -> list[Layer]:
        from ggs.rig.retarget import limb_groups, swap_draw_order

        swapped = {g: v[frame] for g, v in clip.swaps.items()} if clip.swaps else {}
        order = swap_draw_order(self.draw_order, limb_groups(self.rig), swapped)
        rank = {n: k for k, n in enumerate(order)}
        return sorted(self.layers, key=lambda L: min(rank[b] for b in L.bones))

    def _image(self, layer: Layer, scale: float) -> tuple[np.ndarray, float]:
        """The layer's picture at about the size it is drawn (halved until close): (image, factor)."""
        k = 1.0
        while k * 0.5 >= scale and min(layer.image.shape[:2]) * k > 64:  # never smaller than the output
            k *= 0.5
        if k == 1.0:
            return layer.image, 1.0
        cache = self._mips.setdefault(id(layer), {})
        if k not in cache:
            import cv2

            h, w = layer.image.shape[:2]
            cache[k] = cv2.resize(layer.image, (max(1, round(w * k)), max(1, round(h * k))),
                                  interpolation=cv2.INTER_AREA)
        return cache[k], k

    def render_pose(self, start: np.ndarray, ang: np.ndarray, swapped: dict[str, bool], scale: float,
                    size: tuple[int, int], offset: Point) -> np.ndarray:
        """Premultiplied RGBA float32: the drawing posed (bone starts and angles in drawing pixels) x ``scale``,
        shifted by ``offset``. Every layer's triangles are rasterised at once (numpy) and sampled with one remap,
        so a frame takes milliseconds."""
        import cv2

        from ggs.rig.retarget import limb_groups, swap_draw_order

        out = np.zeros((size[1], size[0], 4), np.float32)
        order = swap_draw_order(self.draw_order, limb_groups(self.rig), swapped)
        rank = {n: k for k, n in enumerate(order)}
        for layer in sorted(self.layers, key=lambda L: min(rank[b] for b in L.bones)):
            dst = self.deform(layer, start, ang) * scale + np.array(offset, np.float32)
            tri_rank = np.array([rank[self.bones[b]] for b in layer.tri_bone])
            img, k = self._image(layer, scale)
            mapx, mapy = rasterize(dst, layer.verts * k, layer.tris[np.argsort(tri_rank, kind="stable")], size,
                                   img[..., 3])
            if mapx is None:
                continue
            warped = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))
            out = warped + out * (1.0 - warped[..., 3:4])
        return out

    def render_frame(self, clip, frame: int, scale: float, size: tuple[int, int], offset: Point) -> np.ndarray:
        clip = self.prepare(clip)
        start, ang = self.pose(clip, frame)
        swapped = {g: v[frame] for g, v in clip.swaps.items()} if clip.swaps else {}
        return self.render_pose(start, ang, swapped, scale, size, offset)

    def render_clip(self, clip, height_px: int, padding: int = 2, in_place: bool = True) -> tuple[list, tuple[int, int]]:
        """Frames cropped to one shared box and the pivot (x under the root, y at the ground), like
        ``Puppet.render_clip``."""
        from PIL import Image

        scale = height_px / self.height_px
        lo, hi = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
        for f in range(clip.frame_count):  # shared canvas from the bounds of every deformed mesh
            start, ang = self.pose(clip, f, in_place)
            for layer in self.layers:
                d = self.deform(layer, start, ang) * scale
                lo, hi = np.minimum(lo, d.min(0)), np.maximum(hi, d.max(0))
        off = -lo + padding + 1
        size = (math.ceil(hi[0] - lo[0]) + 2 * padding + 3, math.ceil(hi[1] - lo[1]) + 2 * padding + 3)
        frames, box = [], None
        for f in range(clip.frame_count):
            pre = self.render_frame(clip, f, scale, size, tuple(off))
            a = np.clip(pre[..., 3], 0, 1)
            rgb = np.where(a[..., None] > 1e-4, pre[..., :3] / np.maximum(a[..., None], 1e-4), 0)
            im = Image.fromarray(np.dstack([np.clip(rgb, 0, 1) * 255, a * 255]).round().astype(np.uint8), "RGBA")
            frames.append(im)
            bb = im.getchannel("A").getbbox()
            if bb:
                box = bb if box is None else (min(box[0], bb[0]), min(box[1], bb[1]), max(box[2], bb[2]),
                                              max(box[3], bb[3]))
        if box is None:
            raise SkinError("every frame is empty")
        x0, y0 = max(0, box[0] - padding), max(0, box[1] - padding)
        x1, y1 = min(size[0], box[2] + padding), min(size[1], box[3] + padding)
        frames = [im.crop((x0, y0, x1, y1)) for im in frames]
        return frames, (round(off[0] - x0), box[3] - y0)

    # ------------------------------------------------------------------ files

    def save(self, out_dir: str | Path, image=None) -> Path:
        from PIL import Image

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (image or Image.fromarray(self.source, "RGBA")).save(out / "skin.png")
        if self.parse is not None:  # body parts found in the picture: rebuilds (moved joints) keep them
            from ggs.rig.parsing import save_parse

            save_parse(self.parse, out / "parse.png")
        data = {"format": "ggs.skin", "version": 1, "rig": self.rig.id, "image": "skin.png",
                "height_px": self.height_px, "joints": self.joints, "widths": self.widths, "grid": self.grid,
                "layout": self.layout, "confirmed": bool(getattr(self, "confirmed", False))}
        (out / "skin.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        return out / "skin.json"


def rasterize(dst: np.ndarray, src: np.ndarray, tris: np.ndarray, size: tuple[int, int],
              alpha: np.ndarray | None = None):
    """For every output pixel covered by a triangle, where to sample the source: (mapx, mapy) float32 arrays for
    ``cv2.remap`` (-1 outside), or (None, None) when nothing is covered. Where triangles overlap (a bent knee),
    the one with art at that spot wins (``alpha``: the source's opacity), then the later one."""
    w, h = size
    d = dst[tris]  # (M, 3, 2)
    x0 = np.clip(np.floor(d[..., 0].min(1)), 0, w - 1).astype(np.int64)
    x1 = np.clip(np.ceil(d[..., 0].max(1)), 0, w - 1).astype(np.int64)
    y0 = np.clip(np.floor(d[..., 1].min(1)), 0, h - 1).astype(np.int64)
    y1 = np.clip(np.ceil(d[..., 1].max(1)), 0, h - 1).astype(np.int64)
    e1, e2 = d[:, 1] - d[:, 0], d[:, 2] - d[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    keep = (np.abs(det) > 1e-6) & (x1 >= x0) & (y1 >= y0)
    idx = np.nonzero(keep)[0]
    if not len(idx):
        return None, None
    bw, bh = (x1 - x0 + 1)[idx], (y1 - y0 + 1)[idx]
    n = bw * bh
    tri = np.repeat(np.arange(len(idx)), n)
    off = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    px = x0[idx][tri] + off % bw[tri]
    py = y0[idx][tri] + off // bw[tri]
    t = idx[tri]
    rx, ry = px - d[t, 0, 0], py - d[t, 0, 1]
    inv = 1.0 / det[t]
    u = (rx * e2[t, 1] - ry * e2[t, 0]) * inv
    v = (ry * e1[t, 0] - rx * e1[t, 1]) * inv
    eps = 1e-3
    inside = (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps)
    px, py, t, u, v = px[inside], py[inside], t[inside], u[inside], v[inside]
    s = src[tris[t]]
    sx = s[:, 0, 0] + u * (s[:, 1, 0] - s[:, 0, 0]) + v * (s[:, 2, 0] - s[:, 0, 0])
    sy = s[:, 0, 1] + u * (s[:, 1, 1] - s[:, 0, 1]) + v * (s[:, 2, 1] - s[:, 0, 1])
    lin = py * w + px
    prio = np.arange(len(lin), dtype=np.int64)
    if alpha is not None:
        ax = np.clip(np.round(sx).astype(np.int64), 0, alpha.shape[1] - 1)
        ay = np.clip(np.round(sy).astype(np.int64), 0, alpha.shape[0] - 1)
        prio += (alpha[ay, ax] > 0.5).astype(np.int64) * len(lin)
    order = np.lexsort((prio, lin))
    lin_sorted = lin[order]
    ends = np.nonzero(np.r_[lin_sorted[1:] != lin_sorted[:-1], True])[0]
    last = order[ends]  # per pixel: the triangle with art there, else the last drawn
    mapx = np.full(h * w, -1.0, np.float32)
    mapy = np.full(h * w, -1.0, np.float32)
    mapx[lin[last]] = sx[last]
    mapy[lin[last]] = sy[last]
    return mapx.reshape(h, w), mapy.reshape(h, w)


def vote_regions(labels: np.ndarray, rgba: np.ndarray, rig: RigSpec, joints: Joints,
                 widths: dict[str, float]) -> np.ndarray:
    """Outlined colour regions (a sleeve, a boot) mostly in one layer go wholly to that layer; inside it every
    pixel takes the nearest of the layer's bones. Regions spread over several layers (painterly art without
    outlines) are left as they are."""
    from scipy import ndimage

    from ggs.rig.cutter import nearest_labels

    names = [b.name for b in rig.bones]
    layer_of = np.full(len(names) + 1, -1, np.int64)
    groups = layer_bones(rig)
    for k, (_, bones) in enumerate(groups):
        for b in bones:
            layer_of[names.index(b)] = k
    from ggs.rig.joints import colour_regions, ink_mask

    mask = labels >= 0
    ink = ink_mask(rgba[..., :3], mask)
    regions, _ = colour_regions(rgba[..., :3], mask, ink)
    out = labels.copy()
    for k, sl in enumerate(ndimage.find_objects(regions), start=1):
        if sl is None:
            continue
        r = regions[sl] == k
        lay = layer_of[labels[sl][r]]
        if len(lay) < 30:
            continue
        counts = np.bincount(lay[lay >= 0], minlength=len(groups))
        top = int(np.argmax(counts))
        if counts[top] < REGION_MAJORITY * len(lay) or counts[top] == len(lay) or groups[top][0] == "body":
            continue  # only limbs take regions over (a sleeve joined to the jacket stays split by distance)
        bones = groups[top][1]
        ys, xs = np.nonzero(r)
        pts = np.stack([xs + sl[1].start, ys + sl[0].start], axis=1).astype(np.float32)
        d = np.stack([_seg_dist(pts, *joints[b]) / widths[b] for b in bones], axis=1)
        sub = out[sl]
        sub[r] = np.array([names.index(b) for b in bones])[np.argmin(d, axis=1)]
    out = np.where(ink & mask, -1, out).astype(np.int16)
    return np.where(mask, nearest_labels(np.where(mask, out, -1).astype(np.int16), ink & mask), -1).astype(np.int16)


def smooth_labels(labels: np.ndarray, size: int) -> np.ndarray:
    """Each pixel takes the most common owner around it: thin strips (a sleeve's outline left with the torso)
    join what surrounds them."""
    from scipy.ndimage import uniform_filter

    ids = [i for i in np.unique(labels) if i >= 0]
    if len(ids) < 2:
        return labels
    counts = np.stack([uniform_filter((labels == i).astype(np.float32), size) for i in ids])
    best = np.array(ids, np.int16)[np.argmax(counts, axis=0)]
    return np.where(labels >= 0, best, -1).astype(np.int16)


def drawing_labels(rgba: np.ndarray, rig: RigSpec, joints: Joints, widths: dict[str, float],
                   height_px: float, parse: np.ndarray | None = None) -> np.ndarray:
    """Which bone owns each pixel of a drawing: outlined colour regions go whole to a limb or the trunk, limbs are
    cut straight at the joints, cloth hanging from the waist stays with the pelvis."""
    from ggs.rig.joints import bone_labels, region_labels

    labels = bone_labels(rgba[..., 3], rig, joints, widths, rgb=rgba[..., :3])
    return region_labels(labels, rgba, rig, joints, parse)


def v1_joints(joints: Joints) -> Joints:
    """A layout's joints folded back onto the plain rig: waist + chest = torso, neck + head = head."""
    j = {b: v for b, v in joints.items() if b not in ("chest", "neck", *LOOSE)}
    if "chest" in joints:
        j["torso"] = (joints["torso"][0], joints["chest"][1])
    if "neck" in joints:
        j["head"] = (joints["neck"][0], joints["head"][1])
    return j


def build_skin(rig: RigSpec, image, joints: Joints, widths: dict[str, float], height_px: float,
               max_height: float | None = 640.0, grid: float = GRID, layout: dict | None = None,
               parse: np.ndarray | None = None) -> SkinRig:
    """A SkinRig from a drawing (PIL RGBA). Large drawings are scaled to ``max_height`` px per standing height
    first (frames are made at game sizes; this keeps building and rendering quick)."""
    from PIL import Image

    k = min(1.0, max_height / height_px) if max_height else 1.0
    if k < 1.0:
        if parse is not None:
            parse = np.asarray(Image.fromarray(parse.astype(np.uint8), "L").resize(
                (round(image.width * k), round(image.height * k)), Image.NEAREST)).astype(np.int8)
        image = image.resize((round(image.width * k), round(image.height * k)), Image.LANCZOS)
        joints = {b: ((s[0] * k, s[1] * k), (e[0] * k, e[1] * k)) for b, (s, e) in joints.items()}
        widths = {b: w * k for b, w in widths.items()}
        height_px *= k
    arr = np.asarray(image.convert("RGBA"))
    if layout:  # label with the plain rig, then map through the layout (splits, skirt, hair, cape)
        from ggs.rig.layout import Layout, apply_layout, rig_for

        base = rig_for(rig.id, None)
        j1 = v1_joints(joints)
        w1 = {b: widths[b] for b in j1}
        labels1 = drawing_labels(arr, base, j1, w1, height_px, parse)
        rig, joints2, widths2, labels, lay = apply_layout(labels1, base, Layout(**layout), j1, w1, height_px)
        saved = Layout(**layout)
        if saved.anchors:  # a saved rig: its own joints and anchors win over the recomputed ones
            rig = rig_for(rig.id, saved)
            joints = {**joints2, **{b: joints[b] for b in joints if b in LOOSE}}
            widths = {**widths2, **{b: widths[b] for b in widths if b in LOOSE}}
            lay = saved
        else:
            joints, widths = joints2, widths2
        skin = SkinRig(rig, arr, joints, labels, widths, height_px, grid)
        skin.layout = lay.model_dump()
        skin.parse = parse
        return skin
    labels = drawing_labels(arr, rig, joints, widths, height_px, parse)
    skin = SkinRig(rig, arr, joints, labels, widths, height_px, grid)
    skin.parse = parse
    return skin


def load_skin(path: str | Path, rig: RigSpec | None = None) -> SkinRig:
    from PIL import Image

    from ggs.paths import rigs_dir
    from ggs.rig import load_rig_spec

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "ggs.skin":
        raise SkinError(f"{path} is not a skin.json")
    rig = rig or load_rig_spec(rigs_dir() / f"{data['rig']}.yaml")
    joints = {b: (tuple(s), tuple(e)) for b, (s, e) in data["joints"].items()}
    image = Image.open(path.parent / data["image"]).convert("RGBA")
    from ggs.rig.parsing import load_parse

    skin = build_skin(rig, image, joints, data["widths"], data["height_px"], max_height=None,
                      grid=data.get("grid", GRID), layout=data.get("layout"), parse=load_parse(path.parent / "parse.png"))
    skin.confirmed = bool(data.get("confirmed", False))  # the user checked the joints in the editor
    return skin
