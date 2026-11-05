"""Body layouts: what a character is made of (docs/BODY_LAYOUTS.md).

A layout is a few switches stored with the character:
  neck    none (the head sits on the body) | head (the neck is drawn with the head) | separate
  trunk   one | two (waist + chest: the back can curve)
  hands   separate | with_forearm (mitten hands: the hand is part of the forearm piece)
  extras  any of hair_back, skirt, cape (loose parts with follow-through motion)

``layout_rig`` turns the side rig (humanoid_side_v1) into the layout's rig: extra bones and parts, the layout's
drawing order, a part sheet with room for every piece. It keeps the rig id, so every library clip still fits:
``expand_clip`` adds the tracks the extra bones need (chest and neck share the trunk's and head's turn; skirt,
hair and cape get follow-through, computed once and stored in the clip so engines just play bone tracks).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ggs.rig.spec import Bone, HeadVariant, Part, PartSheet, RigSpec

Extra = Literal["hair_back", "skirt", "cape"]


class Layout(BaseModel):
    neck: Literal["none", "head", "separate"] = "head"
    trunk: Literal["one", "two"] = "one"
    hands: Literal["separate", "with_forearm"] = "separate"
    extras: list[Extra] = Field(default_factory=list)
    # where each loose part hangs on its parent bone, measured on the character's drawing: (t along the parent,
    # offset to the parent's back in standing heights)
    anchors: dict[str, tuple[float, float]] = Field(default_factory=dict)

    def preset(self) -> str | None:
        return next((n for n, p in PRESETS.items() if (p.neck, p.trunk, p.hands) ==
                     (self.neck, self.trunk, self.hands)), None)

    def label(self) -> str:
        base = self.preset()
        words = [PRESET_LABELS.get(base, "Custom")]
        words += [EXTRA_LABELS[e] for e in self.extras]
        return " + ".join(words)


PRESETS: dict[str, Layout] = {
    "chibi": Layout(neck="none", trunk="one", hands="with_forearm"),
    "standard": Layout(neck="head", trunk="one", hands="separate"),
    "tall": Layout(neck="separate", trunk="two", hands="separate"),
}
PRESET_LABELS = {"chibi": "Chibi", "standard": "Standard", "tall": "Tall"}
EXTRA_LABELS = {"hair_back": "long hair", "skirt": "skirt", "cape": "cape"}

# share of the v1 torso the waist keeps when the trunk is two pieces
WAIST_SHARE = 0.45
EXTRA_BONES = {  # parent, attach, rest length (standing heights)
    "skirt": ("pelvis", 1.0, 0.22),
    "hair_back": ("head", 0.35, 0.22),
    "cape": ("chest", 1.0, 0.40),
}
EXTRA_PIVOTS = {"skirt": (0.5, 0.12), "hair_back": (0.5, 0.12), "cape": (0.5, 0.10)}


def layout_rig(base: RigSpec, layout: Layout) -> RigSpec:
    """The side rig with the layout's bones, parts and drawing order."""
    if base.view != "side":
        return base
    bones = [b.model_copy() for b in base.bones]
    by = {b.name: b for b in bones}
    parts = [p.model_copy() for p in base.part_sheet.parts]
    order = list(base.draw_order)
    trunk_top = "torso"
    if layout.trunk == "two":
        torso = by["torso"]
        chest = Bone(name="chest", parent="torso", rest_length=round(torso.rest_length * (1 - WAIST_SHARE), 4))
        torso.rest_length = round(torso.rest_length * WAIST_SHARE, 4)
        for b in bones:
            if b.parent == "torso":
                b.parent = "chest"
        bones.insert(bones.index(torso) + 1, chest)
        by["chest"] = chest
        parts.append(Part(name="chest", bone="chest", cell=(0, 0), pivot=(0.5, 0.83), art_angle=180))
        order.insert(order.index("torso") + 1, "chest")
        trunk_top = "chest"
    if layout.neck == "separate":
        neck = Bone(name="neck", parent=trunk_top, rest_length=0.05)
        head = by["head"]
        head.parent = "neck"
        bones.insert(bones.index(head), neck)
        by["neck"] = neck
        parts.append(Part(name="neck", bone="neck", cell=(0, 0), pivot=(0.5, 0.85), art_angle=180))
        order.insert(order.index("head"), "neck")
    if "skirt" in layout.extras:  # the legs go behind the body, the skirt over them
        legs = [b for b in order if b.split("_")[0] in ("thigh", "shin", "foot")]
        order = legs + [b for b in order if b not in legs]
    for extra in ("skirt", "hair_back", "cape"):
        if extra not in layout.extras:
            continue
        parent, attach, length = EXTRA_BONES[extra]
        if parent == "chest" and "chest" not in by:
            parent = "torso"
        t, side = layout.anchors.get(extra, (attach, 0.0))
        b = Bone(name=extra, parent=parent, rest_length=length, attach=t, attach_side=side)
        bones.append(b)
        by[extra] = b
        parts.append(Part(name=extra, bone=extra, cell=(0, 0), pivot=EXTRA_PIVOTS[extra], art_angle=0))
        if extra == "skirt":
            order.insert(order.index("pelvis") + 1, "skirt")
        elif extra == "hair_back":
            order.insert(0, "hair_back")
        else:
            order.insert(0, "cape")  # the cape behind the hair
    cols = base.part_sheet.cols
    n = len(parts) + len(base.part_sheet.head_variants)
    rows = -(-n // cols)
    for k, p in enumerate(parts):
        p.cell = (k % cols, k // cols)
    variants = [HeadVariant(name=v.name, cell=((len(parts) + k) % cols, (len(parts) + k) // cols))
                for k, v in enumerate(base.part_sheet.head_variants)]
    sheet = PartSheet(cell_px=base.part_sheet.cell_px, cols=cols, rows=rows, parts=parts, head_variants=variants)
    sockets = [s.model_copy(update={"bone": "chest"}) if s.bone == "torso" and "chest" in by else s
               for s in base.sockets]
    return RigSpec(id=base.id, version=base.version, view=base.view, facing=base.facing, near_side=base.near_side,
                   bones=bones, draw_order=order, part_sheet=sheet, sockets=sockets, mocap_profiles={})


def rig_for(rig_id: str, layout: Layout | dict | None):
    """The rig a character uses: its base rig with its layout (None = the plain base rig)."""
    from ggs.paths import rigs_dir
    from ggs.rig import load_rig_spec

    base = load_rig_spec(rigs_dir() / f"{rig_id}.yaml")
    if not layout:
        return base
    return layout_rig(base, layout if isinstance(layout, Layout) else Layout(**layout))


# ---------------------------------------------------------------- animation of the extra bones


def _unwrap_mean(a: float, b: float, w: float = 0.5) -> float:
    d = (b - a + 180.0) % 360.0 - 180.0
    return a + w * d


def derive_angles(angles: dict[str, float], rig: RigSpec) -> dict[str, float]:
    """World angles for chest and neck from a v1 pose (one frame). The v1 torso is split: the waist turns a
    quarter of the bend less, the chest a quarter more (a curved back); the neck takes half of the head's turn."""
    names = {b.name for b in rig.bones}
    out = dict(angles)
    if "chest" in names and "chest" not in angles:
        t, p = angles["torso"], angles["pelvis"]
        bend = (t - p + 180.0) % 360.0 - 180.0
        out["torso"] = t - 0.25 * bend
        out["chest"] = t + 0.25 * bend
    if "neck" in names and "neck" not in angles:
        top = out.get("chest", out["torso"])
        out["neck"] = _unwrap_mean(top, out["head"])
    return out


def _spring(targets: list[float], fps: float, stiffness: float = 60.0, damping: float = 9.0, loop: bool = False
            ) -> list[float]:
    """A damped spring following ``targets`` (degrees per frame): follow-through and settle. A looping clip runs
    twice and keeps the second pass, so the loop joins."""
    passes = 2 if loop else 1
    dt = 1.0 / max(fps, 1.0)
    sub = 4
    th, vel = targets[0], 0.0
    out: list[float] = []
    for _ in range(passes):  # the second pass starts where the first ended: the loop joins
        out = []
        for t in targets:
            for _ in range(sub):
                acc = stiffness * (t - th) - damping * vel
                vel += acc * dt / sub
                th += vel * dt / sub
            out.append(th)
    return out


def extra_targets(clip, extra: str, frame: int, angles: dict[str, float], vx: float) -> float:
    """Where a loose part would hang this frame (world angle, 0 = straight down), before follow-through."""
    drag = max(-35.0, min(35.0, 10.0 * vx))  # moving right (forward) pushes it back (+ angle = towards -x)
    if extra == "skirt":  # sways with 40 % of the legs' mean swing, trails a little when moving
        legs = [angles[b] for b in ("thigh_near", "thigh_far") if b in angles]
        swing = sum(((a + 180.0) % 360.0 - 180.0) for a in legs) / max(len(legs), 1)
        return 0.4 * swing + 0.3 * drag
    lean = ((angles.get("chest", angles["torso"]) - 180.0 + 180.0) % 360.0) - 180.0
    if extra == "hair_back":
        head = ((angles["head"] - 180.0 + 180.0) % 360.0) - 180.0
        return 0.5 * head + 0.6 * drag
    return 0.5 * lean + drag  # cape


def expand_clip(clip, rig: RigSpec, layout: Layout | dict | None = None):
    """``clip`` with tracks for every bone of ``rig`` (a layout rig): derived chest/neck, follow-through for
    skirt, back hair and cape. Mitten hands (``hands: with_forearm``) keep the forearm's angle (no wrist bend).
    A clip that already has every track (and no mitten hands) is returned unchanged."""
    if isinstance(layout, dict):
        layout = Layout(**layout)
    names = [b.name for b in rig.bones]
    missing = [n for n in names if n not in clip.angles]
    if layout is not None and layout.hands == "with_forearm" and "hand_near" in clip.angles:
        angles0 = {b: list(v) for b, v in clip.angles.items()}
        for side in ("near", "far"):
            angles0[f"hand_{side}"] = list(angles0[f"forearm_{side}"])
        local0 = {b: list(v) for b, v in clip.local_angles.items()}
        for side in ("near", "far"):
            local0[f"hand_{side}"] = [0.0] * clip.frame_count
        clip = clip.model_copy(update={"angles": angles0, "local_angles": local0})
    if not missing:
        return clip
    n = clip.frame_count
    angles = {b: list(v) for b, v in clip.angles.items()}
    split = "chest" in missing
    per_frame = [derive_angles({b: v[f] for b, v in clip.angles.items()}, rig) for f in range(n)]
    for b in ("torso", "chest", "neck"):
        if b in names and (b in missing or (b == "torso" and split)):
            angles[b] = [pf[b] for pf in per_frame]
    xs = list(clip.root.x)
    for extra in ("skirt", "hair_back", "cape"):
        if extra not in missing:
            continue
        targets = []
        for f in range(n):
            vx = (xs[min(f + 1, n - 1)] - xs[max(f - 1, 0)]) * clip.fps / 2 if n > 1 else 0.0
            if clip.in_place:
                vx = clip.loop_travel * clip.fps / n if clip.loop_travel else vx
            targets.append(extra_targets(clip, extra, f, per_frame[f], vx))
        angles[extra] = _spring(targets, clip.fps, loop=clip.loop)
    parents = {b.name: b.parent for b in rig.bones}
    local = {}
    for b in names:
        p = parents[b]
        local[b] = [((angles[b][f] - (angles[p][f] if p else 0.0) + 180.0) % 360.0) - 180.0 for f in range(n)]
    lengths = dict(clip.bone_lengths)
    defaults = {b.name: b.rest_length for b in rig.bones}
    if split:
        lengths["chest"] = lengths["torso"] * (1 - WAIST_SHARE)
        lengths["torso"] = lengths["torso"] * WAIST_SHARE
    for b in names:
        lengths.setdefault(b, defaults[b])
    return clip.model_copy(update={"bones": names, "angles": angles, "local_angles": local, "bone_lengths": lengths})


# ---------------------------------------------------------------- the pose GGS draws a character in for animation


def capture_pose(layout: Layout) -> dict[str, float]:
    """The pose of the animation-ready drawing for ``layout`` (side view, facing right): the near arm forward and
    clear of the chest, the far arm back, the legs in a stride. Adjusted per layout (docs/BODY_LAYOUTS.md)."""
    from ggs.rig.master import SPREAD_POSE

    a = dict(SPREAD_POSE)
    a.update({"upper_arm_near": -70.0, "forearm_near": -85.0, "hand_near": -85.0,
              "upper_arm_far": 45.0, "forearm_far": 35.0, "hand_far": 35.0,
              "thigh_near": -25.0, "shin_near": -10.0, "thigh_far": 25.0, "shin_far": 32.0})
    if layout.neck == "none" or layout.hands == "with_forearm":  # chibi: short arms clear of the big head
        a.update({"upper_arm_near": -55.0, "forearm_near": -65.0, "hand_near": -65.0,
                  "upper_arm_far": 45.0, "forearm_far": 38.0, "hand_far": 38.0,
                  "thigh_near": -18.0, "shin_near": -6.0, "thigh_far": 18.0, "shin_far": 24.0})
    if "skirt" in layout.extras:  # a smaller stride keeps the feet showing; the arm clears the skirt
        a.update({"thigh_near": -14.0, "shin_near": -6.0, "thigh_far": 14.0, "shin_far": 20.0,
                  "upper_arm_near": -80.0, "forearm_near": -90.0, "hand_near": -90.0})
    if layout.neck == "separate":  # chin up a little: the neck shows between chin and collar
        a["head"] = 175.0
    return a


# ---------------------------------------------------------------- a drawing measured for a layout

NECK_SHARE = 0.25  # of the v1 head bone (neck base to skull top) that is neck when it is separate


def _bone_angle(s, e) -> float:
    import math

    return math.degrees(math.atan2(-(e[0] - s[0]), e[1] - s[1]))


def _anchor(parent: tuple, start, height_px: float) -> tuple[float, float]:
    """(t along the parent bone, offset along the parent's local x in standing heights) for a point."""
    import math

    (sx, sy), (ex, ey) = parent
    dx, dy = ex - sx, ey - sy
    n2 = max(dx * dx + dy * dy, 1e-6)
    t = ((start[0] - sx) * dx + (start[1] - sy) * dy) / n2
    a = math.radians(_bone_angle(parent[0], parent[1]))
    px, py = sx + t * dx, sy + t * dy
    side = ((start[0] - px) * math.cos(a) + (start[1] - py) * math.sin(a)) / height_px
    return round(t, 4), round(side, 4)


def measure_extras(labels, base: RigSpec, joints: dict, widths: dict, height_px: float) -> dict[str, object]:
    """Masks of the loose parts in a drawing labelled with the base (v1) rig: cloth below the hips (skirt), head
    pixels hanging below the neck behind the body (long hair), trunk pixels far behind the back (cape)."""
    import numpy as np

    idx = {b.name: i for i, b in enumerate(base.bones)}
    yy, xx = np.mgrid[:labels.shape[0], :labels.shape[1]]
    hip = joints["pelvis"][0]
    waist = joints["pelvis"][1]
    neck = joints["torso"][1]
    back_x = min(waist[0], neck[0])  # facing right: the back is towards -x
    tw = widths["torso"]
    skirt = (labels == idx["pelvis"]) & (yy > hip[1] - 0.3 * (hip[1] - waist[1]))
    hair = (labels == idx["head"]) & (yy > neck[1] + 0.02 * height_px) & (xx < back_x - 0.2 * tw)
    cape = (labels == idx["torso"]) & (xx < back_x - 1.5 * tw) & (yy > waist[1])  # hangs behind, below the waist
    return {"skirt": skirt, "hair_back": hair, "cape": cape}


def between_legs(alpha, joints: dict, widths: dict) -> float | None:
    """How much of the space between the thighs is covered, halfway to the knees (the capture pose keeps the legs
    apart): ~0 for trousers and shorts, ~1 for a skirt or dress. None when the legs are too close to tell."""
    import numpy as np

    hip = np.array(joints["thigh_near"][0], np.float32)
    kn, kf = np.array(joints["thigh_near"][1], np.float32), np.array(joints["thigh_far"][1], np.float32)
    wn, wf = widths.get("thigh_near", 8.0), widths.get("thigh_far", 8.0)
    covered, total = 0, 0
    for t in np.linspace(0.5, 0.85, 8):
        a, b = hip + t * (kn - hip), hip + t * (kf - hip)
        d = b - a
        gap = float(np.linalg.norm(d)) - 0.9 * (wn + wf)
        if gap < 3:
            continue
        u = d / np.linalg.norm(d)
        for k in np.linspace(0.0, 1.0, max(int(gap), 2)):
            p = a + u * (0.9 * wn + k * gap)
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= y < alpha.shape[0] and 0 <= x < alpha.shape[1]:
                total += 1
                covered += int(alpha[y, x] >= 128)
    return covered / total if total >= 10 else None


def suggest(labels, base: RigSpec, joints: dict, widths: dict, height_px: float, body_preset: str = "standard"
            ) -> tuple[Layout, dict]:
    """The layout GGS proposes for a drawing, and the measures behind it."""
    import numpy as np

    from ggs.rig.joints import cloth_share

    preset = {"chibi": "chibi", "heroic": "tall", "lanky": "tall"}.get(body_preset, "standard")
    lay = PRESETS[preset].model_copy(deep=True)
    masks = measure_extras(labels, base, joints, widths, height_px)
    idx = {b.name: i for i, b in enumerate(base.bones)}
    head_px = max(int((labels == idx["head"]).sum()), 1)
    torso_px = max(int((labels == idx["torso"]).sum()), 1)
    legs = between_legs(np.where(labels >= 0, 255, 0), joints, widths)
    measures = {
        "body_preset": body_preset,
        "between_legs": None if legs is None else round(legs, 3),
        "cloth_share": round(cloth_share(labels, base, joints, widths), 3),
        "hair_share": round(int(masks["hair_back"].sum()) / head_px, 3),
        "cape_share": round(int(masks["cape"].sum()) / torso_px, 3),
    }
    extras = []
    if (legs is not None and legs >= 0.6) or (legs is None and measures["cloth_share"] >= 0.15):
        extras.append("skirt")
    if measures["hair_share"] >= 0.12:
        extras.append("hair_back")
    if measures["cape_share"] >= 0.15:
        extras.append("cape")
    lay.extras = extras
    return lay, measures


def apply_layout(labels, base: RigSpec, layout: Layout, joints: dict, widths: dict, height_px: float):
    """A drawing labelled with the base rig, adapted to ``layout``: (layout rig, joints, widths, labels as indices
    into the layout rig's bones, layout with its anchors). Splits the torso (waist / chest) and the head (neck /
    head) at straight lines; gives the skirt, back hair and cape their own pixels and bones."""
    import numpy as np

    from ggs.rig.joints import split_along_chain

    j = {b: (tuple(s), tuple(e)) for b, (s, e) in joints.items()}
    w = dict(widths)
    if layout.trunk == "two":
        s, e = j["torso"]
        p = (s[0] + WAIST_SHARE * (e[0] - s[0]), s[1] + WAIST_SHARE * (e[1] - s[1]))
        j["torso"], j["chest"] = (s, p), (p, e)
        w["chest"] = w["torso"]
    if layout.neck == "separate":
        s, e = j["head"]
        p = (s[0] + NECK_SHARE * (e[0] - s[0]), s[1] + NECK_SHARE * (e[1] - s[1]))
        j["neck"], j["head"] = (s, p), (p, e)
        w["neck"] = 0.45 * w["torso"]
    masks = measure_extras(labels, base, joints, widths, height_px)
    anchors = {}
    lay = layout.model_copy(deep=True)
    for extra in [x for x in ("skirt", "hair_back", "cape") if x in layout.extras]:
        m = masks[extra]
        if m.sum() < 30:  # not in this drawing after all
            lay.extras = [x for x in lay.extras if x != extra]
            continue
        ys, xs = np.nonzero(m)
        if extra == "skirt":
            hip, waist = j["pelvis"]
            top_y = hip[1] - 0.3 * (hip[1] - waist[1])
            start = (hip[0], top_y)
            parent = "pelvis"
        elif extra == "hair_back":
            parent = "head"
            hs, he = j["head"]
            base_pt = (hs[0] + 0.35 * (he[0] - hs[0]), hs[1] + 0.35 * (he[1] - hs[1]))
            start = (float(xs.mean()) * 0.3 + base_pt[0] * 0.7, base_pt[1])
        else:
            parent = "chest" if "chest" in j else "torso"
            top = j[parent][1]
            start = (float(xs.max()) * 0.5 + top[0] * 0.5, top[1])
        far = int(np.argmax((xs - start[0]) ** 2 + (ys - start[1]) ** 2))
        bottom = (float(xs[ys >= np.percentile(ys, 90)].mean()), float(ys.max()))
        end = bottom if extra != "hair_back" else (float(xs[far]), float(ys[far]))
        j[extra] = (start, end)
        span = float(xs.max() - xs.min() + 1)
        w[extra] = max(span / 2.0, 4.0)
        anchors[extra] = _anchor(j[parent], start, height_px)
    lay.anchors = anchors
    rig = layout_rig(base, lay)
    names = [b.name for b in rig.bones]
    out = np.full(labels.shape, -1, np.int16)
    base_names = [b.name for b in base.bones]
    for i, n in enumerate(base_names):
        out[labels == i] = names.index(n)
    ys, xs = np.nonzero(out >= 0)
    if "chest" in names:
        sel = out[ys, xs] == names.index("torso")
        k = split_along_chain(xs[sel], ys[sel], ["torso", "chest"], j)
        out[ys[sel], xs[sel]] = np.where(k == 1, names.index("chest"), names.index("torso"))
    if "neck" in names:
        sel = out[ys, xs] == names.index("head")
        k = split_along_chain(xs[sel], ys[sel], ["neck", "head"], j)
        out[ys[sel], xs[sel]] = np.where(k == 1, names.index("head"), names.index("neck"))
    for extra in lay.extras:
        out[masks[extra] & (out >= 0)] = names.index(extra)
    return rig, j, w, out, lay
