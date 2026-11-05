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
              draw_order: list[str] | None = None) -> CutResult:
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


def _clean_parts(master: np.ndarray, labels: np.ndarray, spec: MasterSpec, rig: RigSpec, draw_order: list[str],
                 ink: np.ndarray) -> dict[str, CutPart]:
    """Parts from region/joint ownership (``ggs.rig.joints.region_labels``), no invented shapes:
    - across each joint, the part drawn *below* continues a little under its neighbour, using the neighbour's
      own art (what a rigger does by hand), so a bent elbow shows no gap;
    - areas a part drawn above hides (the torso behind the near arm) get the part's main colour, flat;
    - nothing else is added: no discs, no blur."""
    index = {b.name: i for i, b in enumerate(rig.bones)}
    order = {n: k for k, n in enumerate(draw_order)}
    parent = {b.name: b.parent for b in rig.bones}
    rgb, alpha = master[..., :3], master[..., 3]
    on = alpha >= 128
    widths = spec.fit.get("widths") or {}
    yy, xx = np.mgrid[:labels.shape[0], :labels.shape[1]].astype(np.float32)

    def band(bone: str, joint, toward) -> np.ndarray:
        """Pixels of ``bone`` within OVERLAP x its half-width of ``joint``, on the side ``toward`` points to."""
        w = max(widths.get(bone, 8.0), 3.0)
        d = np.hypot(xx - joint[0], yy - joint[1])
        side = (xx - joint[0]) * toward[0] + (yy - joint[1]) * toward[1]
        return (labels == index[bone]) & on & (d <= (1.0 + OVERLAP) * w) & (side >= -1) & (side <= OVERLAP * w)

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
            extra[p] |= band(b.name, j, dirc)
        else:  # child below: it continues back into the parent
            extra[b.name] |= band(p, j, -dirc)
    parts: dict[str, CutPart] = {}
    for p in rig.part_sheet.parts:
        (sx, sy), (ex, ey) = spec.joints[p.bone]
        length = math.hypot(ex - sx, ey - sy)
        angle = bone_angle((sx, sy), (ex, ey))
        own = labels == index[p.bone]
        if own.sum() < 20:
            raise CutError(f"part {p.name!r}: almost no master pixels were labelled with it")
        above = np.isin(labels, [index[n] for n in draw_order if order[n] > order[p.bone]])
        hidden = np.zeros(labels.shape, bool)
        if p.bone_far is None:  # trunk parts: what the near limbs drawn above them hide, inside their outline
            hidden = convex_hull_mask(own) & above & on
        hidden &= ~own
        body = own | extra[p.bone]
        paint = own & on & ~ink
        flat = np.median(rgb[paint], axis=0) if paint.sum() > 10 else np.array([128, 128, 128])
        out = np.zeros((*labels.shape, 4), np.uint8)
        out[..., :3] = np.where(body[..., None], rgb, np.where(hidden[..., None], flat, 0)).astype(np.uint8)
        out[..., 3] = np.where(body, alpha, np.where(hidden, 255, 0)).astype(np.uint8)
        w = 2.0 * widths.get(p.bone, 8.0)
        parts[p.name] = CutPart(p, out, own, int(hidden.sum()), angle, (sx, sy), length,
                                {"start": w, "end": w}, hidden)
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
                    draw_order: list[str] | None = None) -> Path:
    import yaml

    from ggs.rig.character import CharacterSpec

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet.sheet.save(out / "parts.png")
    if sheet.invented is not None:  # for `ggs character refine`: which pixels were filled in, not seen
        sheet.invented.save(out / "parts_filled.png")
    spec = CharacterSpec(name=name, rig=rig.id, sheet="parts.png", px_per_height=round(sheet.px_per_height, 3),
                         bone_lengths=sheet.bone_lengths, pivots=sheet.pivots, stature=stature,
                         draw_order=draw_order)
    path = out / "character.yaml"
    path.write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False), encoding="utf-8")
    return path
