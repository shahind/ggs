"""Warp animation straight from a character's design: no redrawing, no cutting.

The accepted design (``<item>/design/design.png``) was drawn over a known skeleton (``design_skeleton.json``).
The joint finder moves that skeleton onto the drawing; a skin rig (``ggs.rig.skin``: layers, triangle mesh,
skinning) is built over the picture and saved in ``<item>/warp/``. Animations bend the design with it; frames
are rendered only for export.

Edits made in the app (joints moved, a pose touched up in one frame) are saved as part of the rig
(``skin.json``) or as an edited copy of the clip (``<item>/warp/clips/``), so the CLI exports exactly what the
user saw.
"""

from __future__ import annotations

import json
from pathlib import Path

from ggs.rig.skin import DETAIL, SkinRig, build_skin, load_skin

Point = tuple[float, float]


class WarpError(ValueError):
    pass


def warp_dir(item_dir: Path) -> Path:
    return Path(item_dir) / "warp"


def warp_skin(item_dir: Path) -> Path | None:
    p = warp_dir(item_dir) / "skin.json"
    return p if p.exists() else None


def warp_source(item_dir: Path) -> Path | None:
    """The picture a warp is built from by default: the animation-ready drawing (made with the sliced version)
    when there is one, else the accepted design."""
    from ggs.items import design_dir
    from ggs.rig.slices import rigdraw_dir

    rd = rigdraw_dir(item_dir)
    if (rd / "master.json").exists() and (rd / "master.png").exists():
        return rd / "master.png"
    p = design_dir(item_dir) / "design.png"
    return p if p.exists() else None


def picture_sig(p: Path) -> str:
    import hashlib

    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def warp_stale(item_dir: Path) -> bool:
    """True when the warp was built from another picture than the one it would be built from now (a new design
    was accepted, or the sliced version drew a new animation-ready drawing)."""
    if warp_skin(item_dir) is None:
        return False
    src = warp_source(item_dir)
    if src is None:
        return False
    info_file = warp_dir(item_dir) / "warp.json"
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    if "source_sig" in info:
        return info["source_sig"] != picture_sig(src)
    # warps made before the fingerprint: out of date when the picture is newer than the warp
    return src.stat().st_mtime > (info_file if info_file.exists() else warp_skin(item_dir)).stat().st_mtime


def warp_detail(item_dir: Path) -> str:
    info_file = warp_dir(item_dir) / "warp.json"
    if info_file.exists():
        d = json.loads(info_file.read_text(encoding="utf-8")).get("detail")
        if d in DETAIL:
            return d
    return "normal"


def design_skeleton(item_dir: Path) -> tuple[dict, dict[str, tuple[Point, Point]], float]:
    """(skeleton.json data, joints in design pixels, pixels per standing height)."""
    from ggs.items import design_dir

    f = design_dir(item_dir) / "design_skeleton.json"
    if not f.exists():
        raise WarpError("no design yet: press 'Use this design' first")
    data = json.loads(f.read_text(encoding="utf-8"))
    joints = {b["name"]: (tuple(b["start"]), tuple(b["end"])) for b in data["bones"]}
    return data, joints, float(data["px_per_height"])


def body_widths(rig, cs, joints: dict, height_px: float) -> dict[str, float]:
    """Part half-widths for the design's body shape, in design pixels (from the programmer-art puppet)."""
    from ggs.rig.body import PRESETS, saved_body
    from ggs.rig.joints import bone_widths
    from ggs.rig.make import guide_puppet
    from ggs.rig.master import render_guide

    body = saved_body(cs.body) if cs.body else PRESETS.get(cs.body_preset, PRESETS["standard"])
    puppet, lengths = guide_puppet(rig, body)
    guide = render_guide(rig, puppet, lengths, size=1024)
    k = height_px / guide.spec.height_px
    return {b: w * k for b, w in bone_widths(guide.labels, rig, guide.spec.joints).items()}


def make_design_warp(project: Path, ref: str, detail: str = "normal", *, fit: bool = True, stages=None,
                     source: str = "auto") -> Path:
    """Build the warp rig and save it in ``<item>/warp/``. ``source``: ``drawing`` = the animation-ready drawing
    (made with the sliced version, in the layout's capture pose, with the layout's loose parts), ``design`` = the
    accepted design as it is (advanced: arms over the body bend less well), ``auto`` = the drawing when there is
    one."""
    from ggs.items import load as _load
    from ggs.rig.slices import rigdraw_dir

    _, d0 = _load(Path(project), ref)
    rd = rigdraw_dir(d0) / "master.json"
    if source == "drawing" or (source == "auto" and rd.exists()):
        if not rd.exists():
            raise WarpError("no animation-ready drawing yet: make the sliced version first")
        return _warp_from_drawing(d0, rd, detail)
    from PIL import Image

    from ggs.items import CharacterSettings, design_dir, load
    from ggs.paths import rigs_dir
    from ggs.rig import load_rig_spec

    if detail not in DETAIL:
        raise WarpError(f"detail must be one of {sorted(DETAIL)}")
    rec, d = load(Path(project), ref)
    data, joints, height_px = design_skeleton(d)
    rig = load_rig_spec(rigs_dir() / f"{data['rig']}.yaml")
    if rig.view != "side":
        raise WarpError("bending works on side-view characters (the animations are side view)")
    info = json.loads((design_dir(d) / "design.json").read_text(encoding="utf-8"))
    cs = CharacterSettings(**(info["drawing"].get("character") or {}))
    image = Image.open(design_dir(d) / data["image"]).convert("RGBA")
    report: dict = {}
    parse = None
    if fit:
        from ggs.pipeline.models import MissingModelError
        from ggs.rig.joints import detect_keypoints, fit_joints

        try:
            kp = detect_keypoints([image], stages=stages)[0]
            joints, report = fit_joints(joints, kp, height_px)
        except (MissingModelError, OSError) as exc:
            print(f"joint finder not used ({exc}); the design's own skeleton is used")
        try:
            from ggs.rig.parsing import parse_bodies

            parse = parse_bodies([image], stages=stages)[0]
        except (MissingModelError, OSError):
            parse = None
    widths = body_widths(rig, cs, joints, height_px)
    skin = build_skin(rig, image, joints, widths, height_px, grid=DETAIL[detail], parse=parse if fit else None)
    out = skin.save(warp_dir(d))
    (warp_dir(d) / "warp.json").write_text(json.dumps({
        "format": "ggs.warp", "version": 1, "design": rec.accepted, "detail": detail, "fitted": report,
        "source": "design", "source_sig": picture_sig(design_dir(d) / data["image"])}, indent=1), encoding="utf-8")
    return out


def _warp_from_drawing(d: Path, master_json: Path, detail: str) -> Path:
    from PIL import Image

    from ggs.rig.layout import rig_for
    from ggs.rig.master import load_master

    spec, master, _labels = load_master(master_json)
    fit = spec.fit or {}
    base = rig_for(spec.rig, None)
    widths = fit.get("widths") or {}
    layout = fit.get("layout")
    from ggs.rig.parsing import load_parse

    skin = build_skin(base, Image.fromarray(master, "RGBA"), spec.joints, widths, spec.height_px,
                      grid=DETAIL[detail], layout=layout, parse=load_parse(Path(master_json).parent / "parse.png"))
    out = skin.save(warp_dir(d))
    (warp_dir(d) / "warp.json").write_text(json.dumps({
        "format": "ggs.warp", "version": 1, "source": "drawing", "detail": detail, "layout": layout,
        "source_sig": picture_sig(Path(master_json).parent / "master.png")}, indent=1), encoding="utf-8")
    return out


def move_joints(skin_json: Path, moved: dict[str, Point]) -> SkinRig:
    """Move joints of a saved rig (``moved``: joint name -> new position in rig pixels, where a joint is a bone's
    start, e.g. ``forearm_near`` = the near elbow, or ``<bone>:end`` for a chain's tip) and rebuild it."""
    skin = load_skin(skin_json)
    joints = {b: [list(s), list(e)] for b, (s, e) in skin.joints.items()}
    parent = {b.name: b.parent for b in skin.rig.bones}
    for key, (x, y) in moved.items():
        bone, _, tip = key.partition(":")
        if bone not in joints:
            raise WarpError(f"unknown joint {key!r}")
        if tip == "end":
            joints[bone][1] = [x, y]
            continue
        joints[bone][0] = [x, y]
        p = parent[bone]
        if p is not None and _close(skin.joints[p][1], skin.joints[bone][0]):
            joints[p][1] = [x, y]  # the parent ends where this bone starts
        for b, q in parent.items():  # siblings starting at the same point move too
            if q == p and b != bone and _close(skin.joints[b][0], skin.joints[bone][0]):
                joints[b][0] = [x, y]
    new = {b: (tuple(s), tuple(e)) for b, (s, e) in joints.items()}
    from ggs.rig.layout import rig_for

    rebuilt = build_skin(rig_for(skin.rig.id, None) if skin.layout else skin.rig, _image(skin), new, skin.widths,
                         skin.height_px, max_height=None, grid=skin.grid, layout=skin.layout, parse=skin.parse,
                         mode=skin.mode)
    rebuilt.save(Path(skin_json).parent)
    return rebuilt


def _close(a, b, tol: float = 0.5) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _image(skin: SkinRig):
    from PIL import Image

    return Image.fromarray(skin.source, "RGBA")
