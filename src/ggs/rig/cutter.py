"""Fallback cutter + part-sheet normalizer: master image + guide labels -> part sheet + character.yaml.

numpy + Pillow only (no segmentation model). Steps, see docs/PART_CUTTER.md:
  labels     every opaque master pixel takes the part of the nearest guide pixel
  cut        per sheet part: its own pixels + hidden areas (core parts) + joint caps, filled from its own colours
  normalize  rotate each part to its art_angle about its start joint, one shared scale, place at the cell pivot
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ggs.rig.master import MasterSpec, bone_angle
from ggs.rig.spec import Part, RigSpec

ALPHA_ON = 8  # master pixels at or above this alpha belong to the character
_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class CutError(ValueError):
    pass


# ---------------------------------------------------------------- image helpers

def _shift(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    out = np.full_like(a, fill)
    h, w = a.shape[:2]
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h - max(dy, 0),
                                                                   max(-dx, 0):w - max(dx, 0)]
    return out


def nearest_labels(labels: np.ndarray, need: np.ndarray) -> np.ndarray:
    """Spread labels (>= 0) into unlabelled pixels (-1) until every ``need`` pixel has one (8-neighbour growth,
    so roughly the nearest label)."""
    out = labels.copy()
    if not (out >= 0).any():
        raise CutError("guide labels are empty")
    while ((out < 0) & need).any():
        before = out.copy()
        for dy, dx in _N8:
            src = _shift(before, dy, dx, -1)
            take = (out < 0) & (src >= 0)
            out[take] = src[take]
    return out


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    from PIL import Image, ImageDraw

    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return mask.copy()
    pts = []
    for y in rows:
        xs = np.flatnonzero(mask[y])
        pts += [(int(xs[0]), int(y)), (int(xs[-1]), int(y))]
    pts = sorted(set(pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    img = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
    if len(hull) >= 3:
        ImageDraw.Draw(img).polygon(hull, fill=1, outline=1)
    else:
        ImageDraw.Draw(img).line(hull, fill=1)
    return np.asarray(img).astype(bool) | mask


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(px):
        for dy, dx in _N8:
            out |= _shift(mask, dy, dx, False)
        mask = out.copy()
    return out


def capsule_mask(shape: tuple[int, int], p0: tuple[float, float], p1: tuple[float, float], r: float) -> np.ndarray:
    """Pixels within ``r`` of the segment p0-p1."""
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n2 = dx * dx + dy * dy
    t = np.clip(((xx - p0[0]) * dx + (yy - p0[1]) * dy) / n2, 0, 1) if n2 > 0 else 0.0
    return (xx - p0[0] - t * dx) ** 2 + (yy - p0[1] - t * dy) ** 2 <= r * r


def disc_mask(shape: tuple[int, int], center: tuple[float, float], r: float) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= r * r


def component_sizes(mask: np.ndarray, weight: np.ndarray | None = None) -> list[int]:
    """Sizes of the 8-connected components of ``mask``, largest first (run-length union-find). With
    ``weight`` (bool, same shape) a component's size counts only its ``weight`` pixels."""
    parent: list[int] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    runs: list[tuple[int, int, int]] = []  # (start, end exclusive, id) of the previous row
    sizes: list[int] = []
    for y in range(mask.shape[0]):
        row = mask[y]
        if not row.any():
            runs = []
            continue
        d = np.diff(np.concatenate(([0], row.view(np.uint8), [0])).astype(np.int8))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        csum = None if weight is None else np.concatenate(([0], np.cumsum(weight[y])))
        cur = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            rid = len(parent)
            parent.append(rid)
            sizes.append(e - s if csum is None else int(csum[e] - csum[s]))
            for ps, pe, pid in runs:
                if ps <= e and pe >= s:  # touching diagonally counts (8-connected)
                    a, b = find(rid), find(pid)
                    if a != b:
                        parent[a] = b
            cur.append((s, e, rid))
        runs = cur
    total: dict[int, int] = {}
    for i, n in enumerate(sizes):
        r = find(i)
        total[r] = total.get(r, 0) + n
    return sorted(total.values(), reverse=True)


def fill_from(rgb: np.ndarray, known: np.ndarray, fill: np.ndarray, smooth: int = 4) -> np.ndarray:
    """Colour the ``fill`` pixels from the nearest ``known`` pixels (onion peel, then a light smoothing of the
    filled pixels only). Works inside the bounding box of known | fill."""
    out = rgb.astype(np.float32).copy()
    todo = fill & ~known
    if not todo.any():
        return out
    if not known.any():
        raise CutError("nothing to fill from")
    ys, xs = np.nonzero(known | fill)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    val, have, todo_c = out[y0:y1, x0:x1], known[y0:y1, x0:x1].copy(), todo[y0:y1, x0:x1].copy()
    filled = todo_c.copy()
    while todo_c.any():  # grow through every pixel, so fill areas that do not touch known pixels still get colour
        acc = np.zeros_like(val)
        cnt = np.zeros(have.shape, np.float32)
        for dy, dx in _N8:
            h = _shift(have, dy, dx, False)
            acc += _shift(val, dy, dx, 0.0) * h[..., None]
            cnt += h
        new = ~have & (cnt > 0)
        if not new.any():
            break
        val[new] = acc[new] / cnt[new][:, None]
        have |= new
        todo_c &= ~new
    if smooth:
        val[:] = _membrane(val, known[y0:y1, x0:x1], filled, iters=max(smooth, 1) * 10)
    out[y0:y1, x0:x1] = val
    return out


def _membrane(val: np.ndarray, known: np.ndarray, todo: np.ndarray, iters: int = 80, levels: int = 4) -> np.ndarray:
    """Smooth fill: each ``todo`` pixel becomes the average of its neighbours (a Laplace solve), so colours
    blend in from the whole border instead of streaking in from the nearest edge. Solved coarse to fine."""
    val = val.astype(np.float32)
    if levels > 0 and min(todo.shape) >= 16 and todo.any():
        h, w = (todo.shape[0] + 1) // 2 * 2, (todo.shape[1] + 1) // 2 * 2
        pv = np.zeros((h, w, val.shape[2]), np.float32)
        pk = np.zeros((h, w), np.float32)
        pv[:val.shape[0], :val.shape[1]] = val * ~todo[..., None]
        pk[:val.shape[0], :val.shape[1]] = ~todo
        sv = pv.reshape(h // 2, 2, w // 2, 2, -1).sum(axis=(1, 3))
        sk = pk.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
        small_todo = sk == 0
        small = np.where(small_todo[..., None], 0.0, sv / np.maximum(sk, 1)[..., None])
        small = _membrane(small, ~small_todo, small_todo, iters, levels - 1)
        up = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)[:val.shape[0], :val.shape[1]]
        val = np.where(todo[..., None], up, val)
    fixed = ~todo
    for _ in range(iters):
        acc = (_shift(val, 1, 0, 0.0) + _shift(val, -1, 0, 0.0) + _shift(val, 0, 1, 0.0) + _shift(val, 0, -1, 0.0))
        cnt = (_shift(np.ones(todo.shape, np.float32), 1, 0, 0.0) + _shift(np.ones(todo.shape, np.float32), -1, 0, 0.0)
               + _shift(np.ones(todo.shape, np.float32), 0, 1, 0.0) + _shift(np.ones(todo.shape, np.float32), 0, -1, 0.0))
        val = np.where(fixed[..., None], val, acc / cnt[..., None])
    return val


def part_width(own: np.ndarray, joint: tuple[float, float], angle: float, length: float, inward: float) -> float:
    """Width of ``own`` across the bone near ``joint``. ``inward`` = +1 when the part extends from the joint in
    the bone direction (a start joint), -1 when it extends backwards (an end joint)."""
    ys, xs = np.nonzero(own)
    if xs.size == 0:
        return 0.0
    a = math.radians(angle)
    ux, uy = -math.sin(a), math.cos(a)
    vx, vy = xs - joint[0], ys - joint[1]
    along = (vx * ux + vy * uy) * inward
    perp = vx * -uy + vy * ux
    # joints sit at the narrowest point (neck, wrist, ankle): take the narrowest 3 px slice near the joint
    reach = max(6.0, 0.25 * length)
    slices = [(along >= a0) & (along < a0 + 3.0) for a0 in np.arange(0.0, reach, 3.0)]
    counts = [int(sl.sum()) for sl in slices]
    full = 0.4 * max(counts, default=0)  # skip slices mostly hidden under a part drawn on top
    widths = [float(np.percentile(perp[sl], 97) - np.percentile(perp[sl], 3)) + 1.0
              for sl, n in zip(slices, counts) if n >= max(3, full)]
    if not widths:
        band = np.abs(along) <= max(6.0, 0.3 * length)
        if band.sum() < 3:
            return 0.0
        return float(np.percentile(perp[band], 97) - np.percentile(perp[band], 3))
    return min(widths)


# ---------------------------------------------------------------- cut

@dataclass
class CutPart:
    part: Part
    rgba: np.ndarray  # full master-size canvas, uint8
    own: np.ndarray  # master pixels labelled with this part
    filled_px: int
    angle: float  # bone angle in the master
    start: tuple[float, float]
    length_px: float
    widths: dict[str, float] = field(default_factory=dict)  # "start"/"end" -> px
    invented: np.ndarray | None = None  # pixels the cutter filled in (not seen in the master)


@dataclass
class CutResult:
    labels: np.ndarray  # master pixels -> rig bone index (-1 background)
    parts: dict[str, CutPart]
    moved_to_core: int = 0  # far-side accessory pixels given to the torso/pelvis/head
    moved: np.ndarray | None = None  # mask of those pixels


def master_labels(master: np.ndarray, guide_labels: np.ndarray) -> np.ndarray:
    on = master[..., 3] >= ALPHA_ON
    return np.where(on, nearest_labels(guide_labels, on), -1).astype(np.int16)


def segment_distance(shape: tuple[int, int], p0, p1) -> np.ndarray:
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n2 = dx * dx + dy * dy
    t = np.clip(((xx - p0[0]) * dx + (yy - p0[1]) * dy) / n2, 0, 1) if n2 > 0 else 0.0
    return np.sqrt((xx - p0[0] - t * dx) ** 2 + (yy - p0[1] - t * dy) ** 2)


def move_back_accessories(labels: np.ndarray, guide_labels: np.ndarray, spec: MasterSpec, rig: RigSpec,
                          margin: float = 0.04) -> tuple[np.ndarray, int]:
    """Far limbs are not cut (they reuse the near art), so master pixels labelled with a far limb but well
    outside that limb (a cape, a quiver, long hair behind the body) would be lost. Give them to the torso
    instead, so they move with the body. A pixel still counts as a far limb when it is within ``margin``
    standing heights of the guide's far-limb pixels (clothes and armour are wider than the guide)."""
    index = {b.name: i for i, b in enumerate(rig.bones)}
    far_ids = [index[p.bone_far] for p in rig.part_sheet.parts if p.bone_far]
    far_any = np.isin(labels, far_ids)
    out = labels.copy()
    if not far_any.any():
        return out, 0
    reach = max(1, round(margin * spec.height_px))
    ys, xs = np.nonzero(far_any)
    y0, y1 = max(0, ys.min() - reach), min(labels.shape[0], ys.max() + reach + 1)
    x0, x1 = max(0, xs.min() - reach), min(labels.shape[1], xs.max() + reach + 1)
    inside = np.zeros(labels.shape, bool)
    inside[y0:y1, x0:x1] = _grow(np.isin(guide_labels[y0:y1, x0:x1], far_ids), reach)
    moved = far_any & ~inside
    if moved.any():  # one part for all of it: split between torso and pelvis, a cape would crack when walking
        out[moved] = index["torso"] if "torso" in index else far_ids[0]
    return out, int(moved.sum())


def cut_parts(master: np.ndarray, guide_labels: np.ndarray, spec: MasterSpec, rig: RigSpec,
              draw_order: list[str] | None = None, layers: dict | None = None) -> CutResult:
    if master.shape[:2] != guide_labels.shape:
        raise CutError("master and guide labels differ in size")
    raw = master_labels(master, guide_labels)
    labels, moved = move_back_accessories(raw, guide_labels, spec, rig)
    index = {b.name: i for i, b in enumerate(rig.bones)}
    draw_order = list(draw_order or rig.draw_order)
    order = {n: k for k, n in enumerate(draw_order)}
    children: dict[str, list] = {}
    for b in rig.bones:
        if b.parent:
            children.setdefault(b.parent, []).append(b)
    rgb = master[..., :3]
    alpha = master[..., 3]
    shape = labels.shape
    from ggs.rig.joints import ink_mask

    ink = ink_mask(rgb, alpha >= 128)
    if spec.fit and layers:  # complete layers from See-through: pieces from whole art, split at the joints
        return CutResult(labels, _layered_parts(master, labels, layers, spec, rig, draw_order), moved, labels != raw)
    if spec.fit:  # joints found in the drawing: clean cuts (straight at the joints, real art overlapping)
        return CutResult(labels, _clean_parts(master, labels, spec, rig, draw_order, ink), moved, labels != raw)
    parts: dict[str, CutPart] = {}
    for p in rig.part_sheet.parts:
        (sx, sy), (ex, ey) = spec.joints[p.bone]
        length = math.hypot(ex - sx, ey - sy)
        angle = bone_angle((sx, sy), (ex, ey))
        own = labels == index[p.bone]
        if own.sum() < 20:
            raise CutError(f"part {p.name!r}: almost no master pixels were labelled with it")
        fill = np.zeros(shape, bool)
        above = np.isin(labels, [index[n] for n in draw_order if order[n] > order[p.bone]])
        # own pixels next to a part drawn on top carry that part's edge colour: refill them, never fill from them
        tainted = own & _grow(above, 2)
        if p.bone_far is None:  # core parts: fill what the near limbs drawn above them hide
            fill |= convex_hull_mask(own) & above & (alpha >= 128)
        widths = {}
        joints = []
        # start caps only on limbs (they bend a lot); head and torso turn little and their parent's end capsule
        # covers the joint
        if p.bone_far is not None or any(c.attach == 0.0 for c in children.get(p.bone, [])):
            joints.append(("start", (sx, sy), 1.0))
        if any(c.attach == 1.0 for c in children.get(p.bone, [])):
            joints.append(("end", (ex, ey), -1.0))
        for key, j, inward in joints:
            w = part_width(own, j, angle, length, inward)
            widths[key] = w
            if w > 2:
                if key == "start":  # a full disc: the part is drawn over its parent and rounds the joint
                    fill |= disc_mask(shape, j, w / 2)
                else:  # the limb continues under the part drawn on top, up to and round the joint; only where
                    # the master is opaque, so it never bulges past the silhouette
                    fill |= capsule_mask(shape, (sx, sy), j, w / 2) & (alpha >= 128)
        fill &= ~own
        known = own & (alpha >= 128) & ~tainted
        if known.sum() < 20:  # tiny part: keep its edge pixels as they are
            known, tainted = own & (alpha >= 128), np.zeros(shape, bool)
        inked = own & ink  # outlines: kept as drawn, never filled from
        if (known & ~inked).sum() >= 50:
            known, tainted = known & ~inked, tainted & ~inked
        colour = fill_from(rgb, known, fill | tainted, smooth=8)
        colour = np.where((own & ~(fill | tainted))[..., None], rgb, colour)  # the part's own pixels stay as drawn
        out = np.zeros((*shape, 4), np.uint8)
        out[..., :3] = np.clip(np.where((own | fill)[..., None], colour, 0), 0, 255).astype(np.uint8)
        out[..., 3] = np.where(own, alpha, np.where(fill, 255, 0)).astype(np.uint8)
        parts[p.name] = CutPart(p, out, own, int(fill.sum()), angle, (sx, sy), length, widths, fill | tainted)
    return CutResult(labels, parts, moved, labels != raw)


OVERLAP = 0.6  # how far (x the part's half-width) a part continues under its neighbour across a joint
TRUNK_PARTS = ("torso", "pelvis", "chest", "waist")
NO_HIDDEN = ("head", "neck", "hair_back", "skirt", "cape")


def hidden_shape(own: np.ndarray, front: np.ndarray, joints, bone: str, width: float, trunk: bool,
                 opaque: np.ndarray) -> np.ndarray:
    """Where a part continues under the parts drawn in front of it (the chest under the near arm, the hip under
    the near thigh): a limb along its bone at its own width, a trunk within its own outline. Only pixels the
    front parts cover, and only inside the drawing."""
    from ggs.rig.skin import _capsule

    (sx, sy), (ex, ey) = joints[bone]
    if trunk:
        if own.sum() < 50:
            return np.zeros_like(own)
        shape = convex_hull_mask(own) & _capsule(own.shape, (sx, sy), (ex, ey), 1.4 * width)
    else:
        shape = _capsule(own.shape, (sx, sy), (ex, ey), width)
    return shape & front & opaque & ~own


def _clean_parts(master: np.ndarray, labels: np.ndarray, spec: MasterSpec, rig: RigSpec, draw_order: list[str],
                 ink: np.ndarray) -> dict[str, CutPart]:
    """Parts from region/joint ownership (``ggs.rig.joints.region_labels``), no invented shapes:
    - across each joint, the part drawn *below* continues a little under its neighbour, using the neighbour's
      own art, but only where it is the same material (the same colour area: skin into skin, cloth into cloth),
      so a bent elbow shows no gap and no piece carries a wedge of another part;
    - where a part drawn in front covers this one (the chest under the near arm), the hidden area is added to the
      part in its own flat colour and marked as invented, for the refiner to paint (``ggs.rig.refine``)."""
    index = {b.name: i for i, b in enumerate(rig.bones)}
    order = {n: k for k, n in enumerate(draw_order)}
    parent = {b.name: b.parent for b in rig.bones}
    rgb, alpha = master[..., :3], master[..., 3]
    on = alpha >= 128
    widths = spec.fit.get("widths") or {}
    yy, xx = np.mgrid[:labels.shape[0], :labels.shape[1]].astype(np.float32)
    from ggs.rig.joints import colour_regions

    regions, _ = colour_regions(rgb, on, ink)

    def band(bone: str, joint, toward, same_as: str) -> np.ndarray:
        """Pixels of ``bone`` within OVERLAP x its half-width of ``joint``, on the side ``toward`` points to, in a
        colour area that ``same_as`` also has near the joint (the same material continuing)."""
        w = max(widths.get(bone, 8.0), 3.0)
        d = np.hypot(xx - joint[0], yy - joint[1])
        side = (xx - joint[0]) * toward[0] + (yy - joint[1]) * toward[1]
        near = d <= (1.0 + OVERLAP) * w
        shared = np.unique(regions[near & (labels == index[same_as])])
        shared = shared[shared > 0]
        return ((labels == index[bone]) & on & near & (side >= -1) & (side <= OVERLAP * w)
                & np.isin(regions, shared))

    extra: dict[str, np.ndarray] = {b.name: np.zeros(labels.shape, bool) for b in rig.bones}
    for b in rig.bones:
        p = parent[b.name]
        if p is None:
            continue
        j = spec.joints[b.name][0]
        (s0, e0) = spec.joints[b.name]
        dirc = np.array(e0, np.float32) - np.array(s0, np.float32)
        dirc /= max(float(np.linalg.norm(dirc)), 1e-6)
        if order[p] < order[b.name]:  # parent below: it continues into the child's start
            extra[p] |= band(b.name, j, dirc, p)
        else:  # child below: it continues back into the parent
            extra[b.name] |= band(p, j, -dirc, b.name)
    parts: dict[str, CutPart] = {}
    for p in rig.part_sheet.parts:
        (sx, sy), (ex, ey) = spec.joints[p.bone]
        length = math.hypot(ex - sx, ey - sy)
        angle = bone_angle((sx, sy), (ex, ey))
        own = labels == index[p.bone]
        if own.sum() < 20:
            raise CutError(f"part {p.name!r}: almost no master pixels were labelled with it")
        front = np.isin(labels, [index[n] for n in draw_order if n in index and order[n] > order[p.bone]])
        hidden = np.zeros(labels.shape, bool)
        if p.bone not in NO_HIDDEN:  # the head's width is its hair or hat: it has no hidden continuation
            hidden = hidden_shape(own, front, spec.joints, p.bone, max(widths.get(p.bone, 8.0), 3.0),
                                  p.bone in TRUNK_PARTS, on) & ~extra[p.bone]
        body = own | extra[p.bone]
        out = np.zeros((*labels.shape, 4), np.uint8)
        out[..., :3] = np.where(body[..., None], rgb, 0).astype(np.uint8)
        out[..., 3] = np.where(body, alpha, 0).astype(np.uint8)
        if hidden.any():  # placeholder: the part's typical colour (not its outline), painted over by the refiner
            known = own & on & ~ink
            src = known if known.sum() >= 30 else own & on
            out[hidden, :3] = np.median(rgb[src], axis=0).astype(np.uint8)
            out[hidden, 3] = 255
        w = 2.0 * widths.get(p.bone, 8.0)
        parts[p.name] = CutPart(p, out, own, int(hidden.sum()), angle, (sx, sy), length,
                                {"start": w, "end": w}, hidden)
    return parts


# See-through layer tag -> the bones its pixels may go to (by name prefix); a layer whose bones are missing
# from the rig falls back to every bone
LAYER_BONES = {
    "front hair": ("head", "neck", "hair_back"), "back hair": ("head", "neck", "hair_back"),
    "head": ("head", "neck"), "headwear": ("head",), "face": ("head",), "irides": ("head",),
    "eyebrow": ("head",), "eyewhite": ("head",), "eyelash": ("head",), "eyewear": ("head",), "ears": ("head",),
    "earwear": ("head",), "nose": ("head",), "mouth": ("head",), "neck": ("neck", "head"),
    "neckwear": ("chest", "torso", "neck", "head"),
    "topwear": ("chest", "torso", "waist", "cape", "skirt"),  # sleeves come in See-through's arm layer
    "handwear": ("upper_arm", "forearm", "hand"),
    "bottomwear": ("pelvis", "waist", "skirt", "thigh", "shin"), "legwear": ("pelvis", "thigh", "shin", "skirt"),
    "footwear": ("foot", "shin"),
}
FAR_PENALTY = 1.35
# back to front inside one piece (See-through's layers overlap: the face sits on the head, the hair on the face)
LAYER_ORDER = ["back hair", "tail", "wings", "neck", "head", "face", "ears", "earwear", "nose", "mouth", "eyewhite",
               "irides", "eyelash", "eyebrow", "eyewear", "front hair", "headwear", "bottomwear", "legwear",
               "footwear", "topwear", "neckwear", "handwear", "objects"]


def _limb_cuts(chain: list[str], joints) -> list[tuple[np.ndarray, np.ndarray]]:
    """(point, normal to the child side) of the straight cut at each joint of ``chain``: the bisector of the
    bend. A limb that hangs from the *start* of its parent (legs from the pelvis, which points up) is cut against
    the parent reversed, so the hip cut runs across the leg, not along it."""
    import itertools

    cuts = []
    for a, b in itertools.pairwise(chain):
        p = np.array(joints[b][0], np.float32)
        a0, a1 = np.array(joints[a][0], np.float32), np.array(joints[a][1], np.float32)
        da = a1 - a0
        if np.linalg.norm(p - a0) < np.linalg.norm(p - a1):  # attached at the parent's start
            da = -da
        db = np.array(joints[b][1], np.float32) - p
        n = da / max(np.linalg.norm(da), 1e-6) + db / max(np.linalg.norm(db), 1e-6)
        if np.linalg.norm(n) < 1e-3:
            n = db
        cuts.append((p, n / max(np.linalg.norm(n), 1e-6)))
    return cuts


def _layered_parts(master: np.ndarray, labels: np.ndarray, layers: dict, spec: MasterSpec, rig: RigSpec,
                   draw_order: list[str]) -> dict[str, CutPart]:
    """Pieces from See-through's complete layers (``ggs.pipeline.layers``).

    - Each layer's pixels first go to a limb (or the trunk): the nearest of the bones the layer may belong to
      (``LAYER_BONES``), by the bone's width. Along that limb they are cut straight at every joint, on the
      bisector of the bend, as a rigger cuts: a whole arm layer at the elbow and wrist, the trousers at the hip
      and knee. The shirt layer only goes to the trunk (See-through draws the sleeves with the arms).
    - A piece is all its layers composited back to front. The layers are whole, so the chest under the arm and
      the trousers under the shirt are already drawn; nothing is invented.
    - Across a joint, the piece drawn underneath continues in a straight strip past the cut (OVERLAP x the
      narrower bone's width), with the same layer's art, so a bend shows no gap.
    - Pixels of the drawing the layers do not reproduce (See-through can miss a hat) keep their drawn colour and
      go with the nearest pixel the layers do explain (a hat with the hair, a collar with the shirt)."""
    from ggs.rig.skin import _seg_dist

    index = {b.name: i for i, b in enumerate(rig.bones)}
    parent = {b.name: b.parent for b in rig.bones}
    rank = {n: k for k, n in enumerate(draw_order)}
    widths = spec.fit.get("widths") or {}
    h, w = labels.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    pts_all = np.stack([xx.ravel(), yy.ravel()], axis=1)
    names = [b.name for b in rig.bones if b.name in spec.joints]
    # far limbs reuse the near limb's art, so their own pixels are never shown: pixels in between (the crotch, the
    # shoulder) go to the near limb or the trunk
    dist = {b: (_seg_dist(pts_all, *spec.joints[b]) / max(widths.get(b, 8.0), 3.0)).reshape(h, w)
            * (FAR_PENALTY if b.endswith("_far") else 1.0) for b in names}
    order = [t for t in LAYER_ORDER if t in layers] + [t for t in layers if t not in LAYER_ORDER]
    from ggs.rig.joints import chains

    # every chain a layer can be cut along: the trunk, and each limb with the bone it hangs from (pelvis + leg)
    paths = []
    for c in chains(rig):
        c = [b for b in c if b in index]
        if c and parent.get(c[0]) and parent[c[0]] in index:
            c = [parent[c[0]], *c]
        if c:
            paths.append(c)
    owner: dict[str, np.ndarray] = {}  # tag -> bone index per pixel (-1: not this layer)
    cut_pos: dict[str, tuple] = {}  # tag -> (path id per pixel, signed distance to the cut at each joint)
    for tag in order:
        a = layers[tag][..., 3] > 16
        allowed = [b for b in names if any(b == p or b.startswith(p + "_") for p in LAYER_BONES.get(tag, ()))]
        allowed = allowed or names
        nearest = np.array([index[b] for b in allowed])[np.argmin(np.stack([dist[b] for b in allowed]), axis=0)]
        own = np.where(a, nearest, -1)
        pid = -np.ones((h, w), np.int64)
        # which limb (or the trunk) a pixel belongs to: its nearest allowed bone that is not a shared parent
        # (a pixel nearest the pelvis goes with the nearer leg's path, then the hip cut decides)
        cands = [(k, [b for b in pth if b in allowed]) for k, pth in enumerate(paths)]
        cands = [(k, bs) for k, bs in cands if bs]
        if len(cands) > 1:
            own_of = {}
            for k, bs in cands:
                tips = [b for b in bs if b != paths[k][0]] or bs
                # which limb: plain distance (the far-side penalty only settles pixels between a bone and its
                # neighbour, never which leg a pixel is)
                own_of[k] = np.min(np.stack([dist[b] / (FAR_PENALTY if b.endswith("_far") else 1.0)
                                             for b in tips]), axis=0)
            ks = list(own_of)
            pid = np.where(a, np.array(ks)[np.argmin(np.stack([own_of[k] for k in ks]), axis=0)], -1)
        elif cands:
            pid = np.where(a, cands[0][0], -1)
        signed = {}
        ys_, xs_ = np.nonzero(pid >= 0)
        for k, bs in cands:
            sel = pid[ys_, xs_] == k
            if not sel.any():
                continue
            chain = [b for b in paths[k] if b in bs]
            if len(chain) == 1:
                own[ys_[sel], xs_[sel]] = index[chain[0]]
                continue
            # straight cuts along the bisector at each joint, as a rigger cuts (no curved nearest-bone edges)
            pts = np.stack([xs_[sel], ys_[sel]], axis=1).astype(np.float32)
            kk = np.zeros(len(pts), np.int64)
            for i, (pp, nn) in enumerate(_limb_cuts(chain, spec.joints)):
                d_ = (pts - pp) @ nn
                kk = np.where((d_ > 0) & (kk == i), i + 1, kk)
                plane = np.full((h, w), np.nan, np.float32)
                plane[ys_[sel], xs_[sel]] = d_
                signed[(chain[i], chain[i + 1])] = plane
            own[ys_[sel], xs_[sel]] = np.array([index[b] for b in chain])[kk]
        owner[tag] = own
        cut_pos[tag] = signed
    # what the layers show, stacked back to front, and which bone shows at each pixel
    comp = np.zeros((h, w, 4), np.float32)
    shown = -np.ones((h, w), np.int64)
    for tag in order:
        lay = layers[tag].astype(np.float32) / 255.0
        a = lay[..., 3:4]
        comp[..., :3] = lay[..., :3] * a + comp[..., :3] * (1 - a)
        comp[..., 3:] = a + comp[..., 3:] * (1 - a)
        shown = np.where(lay[..., 3] > 0.5, owner[tag], shown)
    opaque = master[..., 3] >= 128
    # pixels of the drawing the layers do not reproduce: none covers them, or one shows something else there
    # (See-through can draw the whole hair under a hat it missed)
    diff = np.abs(comp[..., :3] * 255.0 - master[..., :3].astype(np.float32)).max(axis=2)
    explained = opaque & (comp[..., 3] > 0.5) & (diff < 70) & (shown >= 0)
    residual = opaque & ~explained
    res_bone = -np.ones((h, w), np.int64)
    if residual.any() and explained.any():  # each goes with the nearest pixel the layers do explain
        from scipy import ndimage

        _, (iy, ix) = ndimage.distance_transform_edt(~explained, return_indices=True)
        res_bone = np.where(residual, shown[iy, ix], -1)
        # a bone no layer drew at all (boots See-through missed): what lies nearest to it, measured by its width
        missing = [b for b in names if not (shown == index[b]).any()]
        if missing:
            near_bone = np.array([index[b] for b in names])[np.argmin(np.stack([dist[b] for b in names]), axis=0)]
            for b in missing:
                res_bone = np.where(residual & (near_bone == index[b]), index[b], res_bone)
    parts: dict[str, CutPart] = {}
    for p in rig.part_sheet.parts:
        b = p.bone
        (sx, sy), (ex, ey) = spec.joints[b]
        length = math.hypot(ex - sx, ey - sy)
        angle = bone_angle((sx, sy), (ex, ey))
        wb = max(widths.get(b, 8.0), 3.0)
        near = [(parent[b], (sx, sy))] if parent[b] else []
        near += [(c.name, spec.joints[c.name][0]) for c in rig.bones if c.parent == b and c.name in spec.joints]
        out = np.zeros((h, w, 4), np.float32)
        mine = np.zeros((h, w), bool)
        for tag in order:
            take = owner[tag] == index[b]
            for nb, _j in near:  # a strip past the joint's straight cut, into the neighbour's pixels of the same
                if nb not in index or rank.get(b, 0) >= rank.get(nb, 0):  # layer, only under a piece drawn over
                    continue
                wj = min(wb, max(widths.get(nb, 8.0), 3.0))
                if (nb, b) in cut_pos[tag]:  # this piece is the child: it continues back into its parent
                    d_ = cut_pos[tag][(nb, b)]
                    band = (owner[tag] == index[nb]) & (d_ > -OVERLAP * wj)
                elif (b, nb) in cut_pos[tag]:  # this piece is the parent: it continues into its child
                    d_ = cut_pos[tag][(b, nb)]
                    band = (owner[tag] == index[nb]) & (d_ < OVERLAP * wj)
                else:
                    continue
                take |= np.nan_to_num(band, nan=0).astype(bool)
            if not take.any():
                continue
            lay = layers[tag].astype(np.float32) / 255.0
            a = np.where(take, lay[..., 3], 0.0)
            out[..., :3] = lay[..., :3] * a[..., None] + out[..., :3] * (1 - a[..., None])
            out[..., 3] = a + out[..., 3] * (1 - a)
            mine |= owner[tag] == index[b]
        res = residual & (res_bone == index[b])
        if res.any():  # what no layer drew (a hat): the drawing's own pixels, on top
            out[res, :3] = master[res, :3] / 255.0
            out[res, 3] = 1.0
            mine |= res
        if mine.sum() < 20:
            raise CutError(f"part {p.name!r}: the layers gave it almost no pixels")
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 3] = np.clip(out[..., 3] * 255, 0, 255).round().astype(np.uint8)
        rgba[..., :3] = np.clip(out[..., :3] * 255, 0, 255).round().astype(np.uint8)
        parts[p.name] = CutPart(p, rgba, mine, 0, angle, (sx, sy), length, {"start": 2 * wb, "end": 2 * wb}, None)
    return parts


# ---------------------------------------------------------------- normalize

@dataclass
class SheetResult:
    sheet: object  # PIL RGBA
    px_per_height: float
    bone_lengths: dict[str, float]
    scale: float  # sheet px per master px
    placements: dict[str, dict]
    pivots: dict[str, tuple[float, float]] = field(default_factory=dict)
    invented: object = None  # PIL "L" image, sheet size: 255 where the cutter filled pixels in


def _to_art(cp: CutPart, art_angle: float, k: float) -> tuple[float, float, float, float]:
    """Bounding box of the part's pixels in art space (pivot at 0,0), at sheet scale k."""
    ys, xs = np.nonzero(cp.rgba[..., 3] > 0)
    th = math.radians(cp.angle - art_angle)
    c, s = math.cos(th), math.sin(th)
    vx, vy = xs - cp.start[0], ys - cp.start[1]
    qx, qy = c * vx + s * vy, -s * vx + c * vy  # R(-th) (p - start)
    return float(qx.min()) * k, float(qx.max()) * k, float(qy.min()) * k, float(qy.max()) * k


def normalize(cut: CutResult, spec: MasterSpec, rig: RigSpec, margin_px: int = 6) -> SheetResult:
    from PIL import Image

    ps = rig.part_sheet
    cell = ps.cell_px
    room = cell - 2 * margin_px
    k = 1.0  # never upscale the master's pixels
    boxes = {}
    for name, cp in cut.parts.items():
        x0, x1, y0, y1 = boxes[name] = _to_art(cp, cp.part.art_angle, 1.0)
        k = min(k, room / max(x1 - x0, 1.0), room / max(y1 - y0, 1.0))
    if not (k > 0.05):
        raise CutError(f"parts do not fit the {cell}px cells (scale {k:.3f})")
    # each part's pivot centres its box in the cell (stored per character, so big capes or hats keep detail)
    pivots = {}
    for name, (x0, x1, y0, y1) in boxes.items():
        pivots[name] = (round(((cell - k * (x1 - x0)) / 2 - k * x0) / cell, 5),
                        round(((cell - k * (y1 - y0)) / 2 - k * y0) / cell, 5))
    sheet = Image.new("RGBA", (ps.cols * cell, ps.rows * cell), (0, 0, 0, 0))
    invented = Image.new("L", sheet.size, 0)
    placements = {}
    for name, cp in cut.parts.items():
        img = Image.fromarray(cp.rgba, "RGBA").convert("RGBa")
        w, h = img.size
        small = img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
        fx, fy = small.width / w, small.height / h
        jx, jy = cp.start[0] * fx, cp.start[1] * fy
        th = math.radians(cp.angle - cp.part.art_angle)
        c, s = math.cos(th), math.sin(th)
        px, py = pivots[name][0] * cell, pivots[name][1] * cell
        # inverse map: art q -> master p = start + R(th) (q - pivot)
        data = (c, -s, jx - c * px + s * py, s, c, jy - s * px - c * py)
        art = small.transform((cell, cell), Image.AFFINE, data, resample=Image.BICUBIC).convert("RGBA")
        col, row = cp.part.cell
        sheet.alpha_composite(art, (col * cell, row * cell))
        if cp.invented is not None and cp.invented.any():
            m = Image.fromarray((cp.invented * 255).astype(np.uint8), "L")
            m = m.resize(small.size, Image.BILINEAR).transform((cell, cell), Image.AFFINE, data,
                                                               resample=Image.BILINEAR)
            invented.paste(m, (col * cell, row * cell))
        placements[name] = {"cell": [col, row], "master_angle": round(cp.angle, 2),
                            "length_px": round(cp.length_px * k, 2), "filled_px": cp.filled_px}
        if name == "head":
            for v in ps.head_variants:  # copies until an edit model draws the expressions
                sheet.alpha_composite(art, (v.cell[0] * cell, v.cell[1] * cell))
    height = spec.height_px
    lengths = {}
    for b in rig.bones:
        (sx, sy), (ex, ey) = spec.joints[b.name]
        lengths[b.name] = math.hypot(ex - sx, ey - sy) / height
    for p in ps.parts:  # far bones reuse the near art, so they get the near length
        if p.bone_far:
            lengths[p.bone_far] = lengths[p.bone]
    return SheetResult(sheet, k * height, {n: round(v, 5) for n, v in lengths.items()}, k, placements, pivots,
                       invented)


def write_character(sheet: SheetResult, rig: RigSpec, out_dir: str | Path, name: str, stature: float = 1.0,
                    draw_order: list[str] | None = None, layout: dict | None = None) -> Path:
    import yaml

    from ggs.rig.character import CharacterSpec

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet.sheet.save(out / "parts.png")
    if sheet.invented is not None:  # for `ggs character refine`: which pixels were filled in, not seen
        sheet.invented.save(out / "parts_filled.png")
    spec = CharacterSpec(name=name, rig=rig.id, sheet="parts.png", px_per_height=round(sheet.px_per_height, 3),
                         bone_lengths=sheet.bone_lengths, pivots=sheet.pivots, stature=stature,
                         draw_order=draw_order, layout=layout)
    path = out / "character.yaml"
    path.write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False), encoding="utf-8")
    return path
