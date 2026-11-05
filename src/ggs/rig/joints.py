"""Find where the joints really are in a drawing, and label its pixels by the bones that run through them.

The drawing model follows the pose guide well, but not exactly: a forearm drawn a little lower, a longer leg, the
far arm at another angle. Cutting by the guide's own shapes then gives sleeves to the torso and skirts to the legs.
Here a small pose detector (ViTPose, Apache-2.0, ~0.2 GB of VRAM) finds shoulders, elbows, wrists, hips, knees and
ankles in the drawing; each detection replaces the guide's joint when it is confident and close to where the guide
put that joint (the drawing was made over the guide, so a far-off detection is a mistake, not a different pose).
Near and far are matched to the guide limb by limb (detectors mix up left and right in side views).

``fit_joints`` returns joints that chain (a child starts where its parent ends), as the rig expects.
``bone_labels`` labels every opaque pixel with the bone whose segment, scaled by that bone's width, is closest;
where a near limb lies over the body the limb wins (it is drawn on top).
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from ggs.rig.spec import RigSpec

Point = tuple[float, float]
Joints = dict[str, tuple[Point, Point]]

POSE_MODEL = "vitpose-base-simple"
COCO = {"l_sho": 5, "r_sho": 6, "l_elb": 7, "r_elb": 8, "l_wri": 9, "r_wri": 10,
        "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14, "l_ank": 15, "r_ank": 16}
MIN_SCORE = 0.45
MAX_SHIFT = 0.15  # standing heights a detection may sit from the guide's joint
TOP_CORE = 1.0  # a bone drawn on top keeps the pixels this deep inside its width; elsewhere the nearest wins
CLOTH_WIDTH = 1.35  # thigh pixels farther out than this x the thigh's half-width count as cloth on the waist


def detect_keypoints(images: list, files: dict[str, Path] | None = None, stages=None) -> list[dict]:
    """COCO keypoints for each RGBA drawing: {name: (x, y, score)}. One model load for all images."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    from ggs.pipeline.models import resolve_pose
    from ggs.pipeline.vram import Stages

    src = str(files["pose"]) if files and files.get("pose") else str(resolve_pose(POSE_MODEL))
    stages = stages or Stages(enforce=False)
    out = []
    with stages.stage("joints"):
        proc = AutoProcessor.from_pretrained(src)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.float32
        model = VitPoseForPoseEstimation.from_pretrained(src, torch_dtype=dtype).to(dev).eval()
        for rgba in images:
            rgba = rgba.convert("RGBA")
            bb = rgba.getchannel("A").getbbox()
            if bb is None:
                out.append({})
                continue
            im = Image.new("RGB", rgba.size, "white")
            im.paste(rgba, mask=rgba.getchannel("A"))
            box = [[bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]]]
            inputs = proc(im, boxes=[box], return_tensors="pt").to(dev)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            with torch.no_grad():
                o = model(**inputs)
            o.heatmaps = o.heatmaps.float()
            res = proc.post_process_pose_estimation(o, boxes=[box])[0][0]
            kp, sc = res["keypoints"].tolist(), res["scores"].tolist()
            out.append({n: (kp[i][0], kp[i][1], sc[i]) for n, i in COCO.items()})
        del model, proc
    return out


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def _rot(v: Point, deg: float) -> Point:
    a = math.radians(deg)
    return (v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a))


def _ang(v: Point) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fit_joints(guide: Joints, kp: dict, height_px: float) -> tuple[Joints, dict]:
    """Guide joints moved onto confident detections. Returns (joints, report) where the report lists, per joint,
    how far it moved (standing heights) or why the detection was not used."""
    lim = MAX_SHIFT * height_px
    report: dict[str, str | float] = {}

    def pick(names: tuple[str, str], targets: tuple[Point, Point]) -> tuple[Point | None, Point | None]:
        """Detections for (near, far) target points, matched by distance (left/right is unreliable)."""
        cands = [kp.get(n) for n in names]
        best, best_cost = (None, None), math.inf
        for order in ((0, 1), (1, 0)):
            got, cost = [], 0.0
            for k, t in zip(order, targets):
                c = cands[k]
                if c is None or c[2] < MIN_SCORE or _dist(c[:2], t) > lim:
                    got.append(None)
                    cost += lim
                else:
                    got.append((c[0], c[1]))
                    cost += _dist(c[:2], t)
            if cost < best_cost:
                best, best_cost = (got[0], got[1]), cost
        return best

    g = {n: (tuple(s), tuple(e)) for n, (s, e) in guide.items()}
    pts: dict[str, Point] = {}

    def take(key: str, found: Point | None, default: Point) -> None:
        pts[key] = found if found is not None else default
        report[key] = round(_dist(pts[key], default) / height_px, 3) if found is not None else "guide"

    sh, sh_far = pick(("r_sho", "l_sho"), (g["upper_arm_near"][0], g["upper_arm_far"][0]))
    shoulders = [p for p in (sh, sh_far) if p is not None]
    take("shoulder", ((sum(p[0] for p in shoulders) / len(shoulders), sum(p[1] for p in shoulders) / len(shoulders))
                      if shoulders else None), g["torso"][1])
    hp, hp_far = pick(("r_hip", "l_hip"), (g["thigh_near"][0], g["thigh_far"][0]))
    hips = [p for p in (hp, hp_far) if p is not None]
    take("hip", ((sum(p[0] for p in hips) / len(hips), sum(p[1] for p in hips) / len(hips)) if hips else None),
         g["pelvis"][0])
    for side in ("near", "far"):
        i = 0 if side == "near" else 1
        e = pick(("r_elb", "l_elb"), (g["forearm_near"][0], g["forearm_far"][0]))[i]
        w = pick(("r_wri", "l_wri"), (g["hand_near"][0], g["hand_far"][0]))[i]
        k = pick(("r_knee", "l_knee"), (g["shin_near"][0], g["shin_far"][0]))[i]
        a = pick(("r_ank", "l_ank"), (g["foot_near"][0], g["foot_far"][0]))[i]
        take(f"elbow_{side}", e, g[f"forearm_{side}"][0])
        take(f"wrist_{side}", w, g[f"hand_{side}"][0])
        take(f"knee_{side}", k, g[f"shin_{side}"][0])
        take(f"ankle_{side}", a, g[f"foot_{side}"][0])

    out: Joints = {}

    def bone(name: str, start: Point, end: Point) -> None:
        out[name] = (start, end)

    def carried(name: str, start: Point, parent_old: str, parent_new: tuple[Point, Point]) -> None:
        """A bone without its own detection keeps its guide shape, turned with its (moved) parent."""
        s0, e0 = g[name]
        ps, pe = g[parent_old]
        turn = _ang(_sub(parent_new[1], parent_new[0])) - _ang(_sub(pe, ps))
        bone(name, start, _add(start, _rot(_sub(e0, s0), turn)))

    hip, shoulder = pts["hip"], pts["shoulder"]
    pel_vec = _sub(g["pelvis"][1], g["pelvis"][0])
    bone("pelvis", hip, _add(hip, pel_vec))
    bone("torso", out["pelvis"][1], shoulder)
    carried("head", shoulder, "torso", out["torso"])
    for side in ("near", "far"):
        bone(f"upper_arm_{side}", shoulder, pts[f"elbow_{side}"])
        bone(f"forearm_{side}", pts[f"elbow_{side}"], pts[f"wrist_{side}"])
        carried(f"hand_{side}", pts[f"wrist_{side}"], f"forearm_{side}", out[f"forearm_{side}"])
        bone(f"thigh_{side}", hip, pts[f"knee_{side}"])
        bone(f"shin_{side}", pts[f"knee_{side}"], pts[f"ankle_{side}"])
        s0, e0 = g[f"foot_{side}"]  # feet keep their direction on the ground
        bone(f"foot_{side}", pts[f"ankle_{side}"], _add(pts[f"ankle_{side}"], _sub(e0, s0)))
    for n in guide:
        out.setdefault(n, g[n])
    return out, report


def bone_widths(guide_labels: np.ndarray, rig: RigSpec, guide: Joints) -> dict[str, float]:
    """Half-width of every bone's part in the guide (as a capsule round the bone), in pixels."""
    out = {}
    for i, b in enumerate(rig.bones):
        area = float((guide_labels == i).sum())
        length = max(_dist(*guide[b.name]), 1.0)
        out[b.name] = max(area / length / 2.0, 3.0)
    for name, w in list(out.items()):  # far limbs are mostly hidden in the guide: as wide as the near ones
        if name.endswith("_far") and name[:-4] + "_near" in out:
            out[name] = max(w, out[name[:-4] + "_near"])
    return out


def _segment_distance(shape: tuple[int, int], p0: Point, p1: Point, box) -> np.ndarray:
    y0, y1, x0, x1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n2 = dx * dx + dy * dy
    t = np.clip(((xx - p0[0]) * dx + (yy - p0[1]) * dy) / n2, 0, 1) if n2 > 0 else 0.0
    return np.sqrt((xx - p0[0] - t * dx) ** 2 + (yy - p0[1] - t * dy) ** 2)


def bone_labels(alpha: np.ndarray, rig: RigSpec, joints: Joints, widths: dict[str, float],
                on: int = 128, cloth: bool = True, rgb: np.ndarray | None = None) -> np.ndarray:
    """-1 for transparent pixels, else the index (into rig.bones) of the bone that owns the pixel."""
    h, w = alpha.shape
    mask = alpha >= on
    ys, xs = np.nonzero(mask)
    labels = np.full((h, w), -1, np.int16)
    if not len(ys):
        return labels
    box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    order = {n: k for k, n in enumerate(rig.draw_order)}
    best = np.full((box[1] - box[0], box[3] - box[2]), np.inf, np.float32)
    lab = np.full(best.shape, -1, np.int16)
    # top-most bones first: a later (lower) bone only wins where it is clearly closer, so a limb drawn over the
    # body keeps its pixels inside its own width
    for b in sorted(rig.bones, key=lambda b: -order.get(b.name, 0)):
        d = _segment_distance((h, w), *joints[b.name], box) / widths[b.name]
        i = rig.bones.index(b)
        take = (best > TOP_CORE) & (d < best)
        best = np.where(take, d, best)
        lab = np.where(take, i, lab)
    if cloth and rgb is not None:
        lab = _cloth_regions(lab, mask[box[0]:box[1], box[2]:box[3]], rgb[box[0]:box[1], box[2]:box[3]], rig,
                             joints, widths, box)
    sub = mask[box[0]:box[1], box[2]:box[3]]
    labels[box[0]:box[1], box[2]:box[3]] = np.where(sub, lab, -1)
    return labels


def ink_mask(rgb: np.ndarray, mask: np.ndarray | None = None, width: int = 7) -> np.ndarray:
    """Outline pixels: dark and thin. Dark areas wider than ``width`` px (black boots, navy trousers) are paint,
    not outlines."""
    from scipy import ndimage

    dark = rgb[..., :3].max(axis=2) < 80
    if mask is not None:
        dark &= mask
    thick = ndimage.binary_opening(dark, structure=np.ones((width, width), bool))
    return dark & ~ndimage.binary_dilation(thick, iterations=1)


def colour_regions(rgb: np.ndarray, mask: np.ndarray, ink: np.ndarray, step: int = 40):
    """Connected areas of one colour: split at outlines and wherever the colour jumps (art without outlines)."""
    from scipy import ndimage

    c = rgb[..., :3].astype(np.int16)
    edge = np.zeros(mask.shape, bool)
    jump_x = np.abs(c[:, 1:] - c[:, :-1]).max(axis=2) > step
    jump_y = np.abs(c[1:] - c[:-1]).max(axis=2) > step
    edge[:, 1:] |= jump_x
    edge[1:] |= jump_y
    return ndimage.label(mask & ~ink & ~edge)


def _cloth_regions(lab: np.ndarray, mask: np.ndarray, rgb: np.ndarray, rig: RigSpec, joints: Joints,
                   widths: dict[str, float], box) -> np.ndarray:
    """Skirts and coat tails: in flat game art the outlines enclose colour regions (a skirt panel, a stocking).
    A region on the thighs that mostly reaches past the thighs' width hangs from the
    waist: the whole region goes to the pelvis. Outline pixels then take the label of the nearest region."""
    from scipy import ndimage

    from ggs.rig.cutter import nearest_labels

    pelvis = next((i for i, b in enumerate(rig.bones) if b.parent is None), None)
    thighs = [i for i, b in enumerate(rig.bones) if b.name.startswith("thigh")]
    trunk = [i for i, b in enumerate(rig.bones) if b.parent is None or b.parent == rig.bones[pelvis].name
             and not b.name.startswith("thigh")] if pelvis is not None else []
    if pelvis is None or not thighs:
        return lab
    ink = ink_mask(rgb, mask)
    regions, _n = colour_regions(rgb, mask, ink)
    yy, xx = np.mgrid[box[0]:box[1], box[2]:box[3]].astype(np.float32)
    dist, along = [], []
    for i in thighs:
        p0, p1 = joints[rig.bones[i].name]
        dist.append(_segment_distance(mask.shape, p0, p1, box) / widths[rig.bones[i].name])
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        along.append(((xx - p0[0]) * dx + (yy - p0[1]) * dy) / max(dx * dx + dy * dy, 1e-6))
    dmin = np.minimum.reduce(dist)
    tmin = np.minimum.reduce(along)
    out = lab.copy()
    idx = ndimage.find_objects(regions)
    for k, sl in enumerate(idx, start=1):
        if sl is None:
            continue
        r = regions[sl] == k
        owners = lab[sl][r]
        on_thigh = np.isin(owners, thighs)
        if on_thigh.sum() < 0.3 * len(owners) or len(owners) < 30:
            continue
        if (~np.isin(owners, [*thighs, *trunk])).mean() > 0.2:  # a skirt sits on the trunk and thighs only
            continue
        wide = (dmin[sl][r] > CLOTH_WIDTH).mean() > 0.35
        upper = np.median(tmin[sl][r]) < 0.75
        if upper and wide:  # straddling both legs alone is just trousers without an outline between them
            out[sl][r] = pelvis
    out = np.where(ink & mask, -1, out)
    return np.where(mask, nearest_labels(np.where(mask, out, -1).astype(np.int16), ink & mask), -1).astype(np.int16)


def cloth_share(labels: np.ndarray, rig: RigSpec, joints: Joints, widths: dict[str, float]) -> float:
    """How much of the legs' top is cloth hanging from the waist (0 = trousers, ~0.3+ = a skirt or long coat):
    pelvis pixels below the hip, compared with the thighs' own pixels."""
    pelvis = next(i for i, b in enumerate(rig.bones) if b.parent is None)
    hip_y = joints[rig.bones[pelvis].name][0][1]
    ys = np.nonzero(labels == pelvis)[0]
    hanging = float((ys > hip_y + 0.3 * widths[rig.bones[pelvis].name]).sum())
    thighs = float(sum((labels == i).sum() for i, b in enumerate(rig.bones) if b.name.startswith("thigh")))
    return hanging / max(hanging + thighs, 1.0)


def fit_master(master_json: str | Path, kp: dict, rig: RigSpec) -> dict:
    """Move a master's skeleton onto the detected joints and label its pixels from that skeleton.

    Rewrites master.json (joints, angles, origin) and guide_labels.png (the labels the cutter reads); the guide's
    own labels stay in guide_labels_guide.png. Returns the fit report (how far each joint moved)."""
    import json

    from PIL import Image

    from ggs.rig.master import bone_angle, load_master
    from ggs.rig.skin import drawing_labels

    path = Path(master_json)
    spec, master, glabels = load_master(path)
    joints, report = fit_joints(spec.joints, kp, spec.height_px)
    widths = bone_widths(glabels, rig, spec.joints)
    labels = drawing_labels(master, rig, joints, widths, spec.height_px)
    guide_copy = path.parent / "guide_labels_guide.png"
    if not guide_copy.exists():
        Image.fromarray((glabels + 1).astype(np.uint8), "L").save(guide_copy)
    Image.fromarray((labels + 1).astype(np.uint8), "L").save(path.parent / spec.labels)
    root = next(b.name for b in rig.bones if b.parent is None)
    spec = spec.model_copy(update={"joints": joints, "angles": {b: bone_angle(*joints[b]) for b in joints},
                                   "origin": joints[root][0]})
    data = spec.model_dump()
    from ggs.rig.skin import SKIRT_SHARE, legs_behind

    skirt = cloth_share(labels, rig, joints, widths) >= SKIRT_SHARE
    data["fit"] = {"model": POSE_MODEL, "moved": report, "widths": widths, "skirt": skirt,
                   "draw_order": legs_behind(rig.draw_order) if skirt else None}
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return report


def chains(rig: RigSpec) -> list[list[str]]:
    """Bone chains, parent to child: the trunk (root up to the head) and every near/far limb."""
    from ggs.rig.retarget import limb_groups

    groups = limb_groups(rig)
    limbs = [list(g.near) for g in groups] + [list(g.far) for g in groups]
    taken = {b for c in limbs for b in c}
    trunk = [b.name for b in rig.bones if b.name not in taken]
    return [trunk, *limbs]


def _joint_cuts(chain: list[str], joints: Joints) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each joint between consecutive bones of a chain: (point, normal pointing to the child side). The cut
    is the line through the joint along the bisector of the bend, as a rigger would cut."""
    cuts = []
    for a, b in itertools.pairwise(chain):
        p = np.array(joints[b][0], np.float32)
        da = np.array(joints[a][1], np.float32) - np.array(joints[a][0], np.float32)
        db = np.array(joints[b][1], np.float32) - np.array(joints[b][0], np.float32)
        n = da / max(np.linalg.norm(da), 1e-6) + db / max(np.linalg.norm(db), 1e-6)
        if np.linalg.norm(n) < 1e-6:  # folded back on itself: cut across the child
            n = db
        cuts.append((p, n / max(np.linalg.norm(n), 1e-6)))
    return cuts


def split_along_chain(xs: np.ndarray, ys: np.ndarray, chain: list[str], joints: Joints) -> np.ndarray:
    """Index into ``chain`` for each pixel: the bone between the joint cuts it lies."""
    k = np.zeros(len(xs), np.int64)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    for i, (p, n) in enumerate(_joint_cuts(chain, joints)):
        k = np.where(((pts - p) @ n > 0) & (k == i), i + 1, k)
    return k


def region_labels(labels: np.ndarray, rgba: np.ndarray, rig: RigSpec, joints: Joints) -> np.ndarray:
    """Clean ownership for flat game art: every outlined colour region goes to one limb (or the trunk), whole;
    inside a limb the bones are split by straight cuts at the joints. ``labels`` (nearest bone by width) only
    votes. A region shared by several limbs (no outline between them) is split by those votes first. Outline
    pixels take the owner of the region next to them."""
    from ggs.rig.cutter import nearest_labels

    names = [b.name for b in rig.bones]
    index = {n: i for i, n in enumerate(names)}
    chain_list = chains(rig)
    chain_of = np.full(len(names), -1, np.int64)
    for c, ch in enumerate(chain_list):
        for b in ch:
            chain_of[index[b]] = c
    mask = labels >= 0
    ink = ink_mask(rgba[..., :3], mask)
    regions, _n = colour_regions(rgba[..., :3], mask, ink)
    from scipy import ndimage

    out = np.full(labels.shape, -1, np.int16)
    for k, sl in enumerate(ndimage.find_objects(regions), start=1):
        if sl is None:
            continue
        r = regions[sl] == k
        votes = chain_of[labels[sl][r]]
        ys, xs = np.nonzero(r)
        xs, ys = xs + sl[1].start, ys + sl[0].start
        counts = np.bincount(votes[votes >= 0], minlength=len(chain_list))
        top = int(np.argmax(counts))
        whole = counts[top] >= 0.8 * len(votes)
        per_px = np.full(len(xs), top) if whole else np.where(votes >= 0, votes, top)
        lab = np.empty(len(xs), np.int16)
        for c in np.unique(per_px):
            sel = per_px == c
            ch = chain_list[c]
            lab[sel] = np.array([index[b] for b in ch])[split_along_chain(xs[sel], ys[sel], ch, joints)]
        sub = out[sl]
        sub[r] = lab
    # cloth hanging from the waist (labelled by bone_labels) stays with the pelvis
    root = next(i for i, b in enumerate(rig.bones) if b.parent is None)
    out = np.where((labels == root) & ~ink & (out >= 0) & (chain_of[np.maximum(out, 0)] != chain_of[root]),
                   root, out)
    filled = nearest_labels(np.where(mask & ~ink, out, -1).astype(np.int16), mask)
    return np.where(mask, filled, -1).astype(np.int16)
