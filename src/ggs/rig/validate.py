"""Reassembly validator: does a part sheet rebuild its master image, and does it hold together when posed?

Checks and limits are listed in docs/PART_CUTTER.md. ``validate_character`` returns a report; ``passed`` is
False if any check fails. The report and a debug image are written by the CLI next to the character.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ggs.rig.cutter import CutResult, component_sizes, convex_hull_mask
from ggs.rig.master import MASTER_POSE, TEST_POSES, MasterSpec, bone_angle, pose_clip
from ggs.rig.spec import RigSpec


@dataclass
class Limits:
    min_iou: float = 0.90
    max_colour_error: float = 0.10  # mean |RGB difference| / 255 over near/core part pixels
    min_guide_agreement: float = 0.40
    joint_on_part: float = 0.30  # joint distance to its own part, in part widths
    joint_to_parent: float = 0.80  # joint distance to the part it attaches to, in part widths
    min_largest_piece: float = 0.85
    max_pose_pieces: int = 1
    piece_min_fraction: float = 0.003  # pieces smaller than this share of the silhouette are ignored


@dataclass
class Check:
    name: str
    passed: bool
    value: float
    limit: float
    detail: str = ""


@dataclass
class Report:
    passed: bool
    score: float  # reassembly IoU
    checks: list[Check] = field(default_factory=list)
    images: dict = field(default_factory=dict)  # name -> PIL image (reassembly, test poses)

    def failures(self) -> list[str]:
        return [f"{c.name}: {c.detail or c.value}" for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "score": round(self.score, 4),
                "checks": [{"name": c.name, "passed": c.passed, "value": round(float(c.value), 4),
                            "limit": c.limit, "detail": c.detail} for c in self.checks]}


def _min_dist(mask: np.ndarray, p: tuple[float, float]) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return math.inf
    return float(np.sqrt(((xs - p[0]) ** 2 + (ys - p[1]) ** 2).min()))


def _subtree(rig: RigSpec, bone: str) -> set[str]:
    out, todo = set(), [bone]
    while todo:
        cur = todo.pop()
        out.add(cur)
        todo += [b.name for b in rig.bones if b.parent == cur]
    return out


def _above(labels: np.ndarray, rig: RigSpec, bone: str, index: dict[str, int], exclude: set[str]) -> np.ndarray:
    """Pixels of parts drawn after ``bone``, except ``exclude``."""
    order = rig.draw_order
    return np.isin(labels, [index[n] for n in order[order.index(bone) + 1:] if n not in exclude])


def master_angles(spec: MasterSpec, rig: RigSpec) -> dict[str, float]:
    return {b.name: bone_angle(*spec.joints[b.name]) for b in rig.bones}


def validate_character(character, rig: RigSpec, master: np.ndarray, spec: MasterSpec, cut: CutResult,
                       guide_labels: np.ndarray, limits: Limits | None = None) -> Report:
    from PIL import Image

    lim = limits or Limits()
    checks: list[Check] = []
    puppet = character.puppet(rig)
    lengths = dict(character.spec.bone_lengths)
    size = tuple(spec.size)
    index = {b.name: i for i, b in enumerate(rig.bones)}
    sheet_parts = rig.part_sheet.parts

    # 1-2. reassembly in the master pose
    clip = pose_clip(rig, master_angles(spec, rig), lengths, "master")
    rebuilt = puppet.render_frame(clip, 0, spec.height_px, size, spec.origin)
    r = np.asarray(rebuilt)
    m_on, r_on = master[..., 3] >= 128, r[..., 3] >= 128
    union = (m_on | r_on).sum()
    iou = float((m_on & r_on).sum() / union) if union else 0.0
    checks.append(Check("reassembly_iou", iou >= lim.min_iou, iou, lim.min_iou,
                        f"IoU {iou:.3f} of the rebuilt silhouette with the master"))
    core = np.isin(cut.labels, [index[p.bone] for p in sheet_parts]) & m_on & r_on
    err = float(np.abs(master[..., :3].astype(np.float32) - r[..., :3])[core].mean() / 255) if core.any() else 1.0
    checks.append(Check("reassembly_colour", err <= lim.max_colour_error, err, lim.max_colour_error,
                        f"mean colour difference {err:.3f}"))

    # 3-5. per part: guide agreement, connectedness, joints
    for p in sheet_parts:
        i = index[p.bone]
        own = (cut.labels == i) & m_on
        g = guide_labels == i
        mine = own if cut.moved is None else own & ~cut.moved  # accessories moved to a core part don't count
        u = (mine | g).sum()
        agree = float((mine & g).sum() / u) if u else 0.0
        checks.append(Check(f"guide_agreement:{p.name}", agree >= lim.min_guide_agreement, agree,
                            lim.min_guide_agreement, f"{p.name}: IoU {agree:.2f} with the guide (did the master "
                                                     "follow the pose?)"))
        # a wide torso crossed by the near arm shows in two pieces: count what parts drawn on top hide inside it
        hidden = _above(cut.labels, rig, p.bone, index, {p.bone}) & m_on & convex_hull_mask(own)
        sizes = component_sizes(own | hidden, weight=own)  # pieces joined through hidden pixels, own pixels counted
        frac = sizes[0] / sum(sizes) if sizes and sum(sizes) else 0.0
        checks.append(Check(f"connected:{p.name}", frac >= lim.min_largest_piece, frac, lim.min_largest_piece,
                            f"{p.name}: largest piece holds {frac:.0%} of its pixels"))
        bone = rig.bone(p.bone)
        if bone.parent is None:
            continue
        cp = cut.parts[p.name]
        par_cut = next((q for q in cut.parts.values() if q.part.bone == bone.parent), None)
        width = max(cp.widths.get("start") or cp.widths.get("end")  # core parts have no start cap
                    or (par_cut.widths.get("end") if par_cut else None) or 0.03 * spec.height_px, 4.0)
        start = spec.joints[p.bone][0]
        # a joint may be hidden under a part drawn above (in this rig the arms attach where the neck starts)
        # ... but never under the part itself or its own children: that would hide a misplaced joint
        mine = _subtree(rig, p.bone)
        d_own = _min_dist(own | _above(cut.labels, rig, p.bone, index, mine) & m_on, start) / width
        parent_mask = (cut.labels == index[bone.parent]) & m_on
        d_par = _min_dist(parent_mask | _above(cut.labels, rig, bone.parent, index, mine) & m_on, start) / width
        ok = d_own <= lim.joint_on_part and d_par <= lim.joint_to_parent
        checks.append(Check(f"joint:{p.name}", ok, max(d_own, d_par), lim.joint_to_parent,
                            f"{p.name} start joint is {d_own:.2f} widths from its part and {d_par:.2f} from "
                            f"{bone.parent}"))

    # 6. cell fit: nothing touches a cell border
    sheet = np.asarray(character.load_sheet())
    cell = rig.part_sheet.cell_px
    for p in sheet_parts:
        c0, r0 = p.cell[0] * cell, p.cell[1] * cell
        a = sheet[r0:r0 + cell, c0:c0 + cell, 3]
        edge = int(a[0].max() > 0 or a[-1].max() > 0 or a[:, 0].max() > 0 or a[:, -1].max() > 0)
        checks.append(Check(f"cell_fit:{p.name}", edge == 0, edge, 0, f"{p.name} touches its cell border"))

    # 7. test poses: the puppet stays one piece
    images = {"reassembly": rebuilt}
    if rig.view == "front":
        from ggs.rig.poses import FRONT_REST

        tests, base = FRONT_TEST_POSES, FRONT_REST
    else:
        tests, base = TEST_POSES, MASTER_POSE
    for name, overrides in tests.items():
        angles = {**base, **overrides}
        im = puppet.render_frame(pose_clip(rig, angles, lengths, name), 0, spec.height_px, size, spec.origin)
        on = np.asarray(im)[..., 3] >= 128
        sizes = component_sizes(on)
        total = sum(sizes) or 1
        pieces = sum(1 for s in sizes if s / total >= lim.piece_min_fraction)
        checks.append(Check(f"pose:{name}", pieces <= lim.max_pose_pieces, pieces, lim.max_pose_pieces,
                            f"pose {name!r} falls into {pieces} pieces"))
        images[f"pose_{name}"] = im
    passed = all(c.passed for c in checks)
    diff = np.zeros((*m_on.shape, 4), np.uint8)
    diff[m_on & ~r_on] = (220, 40, 40, 255)  # master only
    diff[r_on & ~m_on] = (40, 90, 220, 255)  # rebuilt only
    diff[m_on & r_on] = (200, 200, 200, 255)
    images["diff"] = Image.fromarray(diff, "RGBA")
    return Report(passed, iou, checks, images)


# front-view rigs: arms up and bent, a squat (overrides of the front rest pose)
FRONT_TEST_POSES: dict[str, dict[str, float]] = {
    "reach": {"upper_arm_l": 150.0, "forearm_l": 165.0, "hand_l": 165.0,
              "upper_arm_r": -150.0, "forearm_r": -165.0, "hand_r": -165.0},
    "bend": {"upper_arm_l": 60.0, "forearm_l": -40.0, "hand_l": -40.0, "upper_arm_r": -30.0, "forearm_r": 70.0,
             "hand_r": 70.0},
    "squat": {"thigh_l": 60.0, "shin_l": -10.0, "thigh_r": -60.0, "shin_r": 10.0},
}


def debug_sheet(master_img, report: Report, tile: int = 384):
    """master | reassembly | diff | test poses, on white."""
    from PIL import Image

    ims = [master_img, report.images["reassembly"], report.images["diff"]] + [
        v for k, v in report.images.items() if k.startswith("pose_")]
    out = Image.new("RGB", (tile * len(ims), tile), "white")
    for i, im in enumerate(ims):
        t = im.convert("RGBA").resize((tile, tile), Image.LANCZOS)
        bg = Image.new("RGBA", t.size, (255, 255, 255, 255))
        bg.alpha_composite(t)
        out.paste(bg.convert("RGB"), (i * tile, 0))
    return out
