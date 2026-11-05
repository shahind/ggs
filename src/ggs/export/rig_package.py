"""Rig package: ``parts.png`` atlas + ``rig.json`` (bones, parts, draw order, sockets) + ``clips/*.json``.

Pure data, in output pixels: an engine importer (or ``load_rig_package``) rebuilds the puppet from it.
See docs/EXPORT_FORMATS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from ggs import __version__
from ggs.export.pack import paste_extruded, shelf_pack
from ggs.rig.character import Character
from ggs.rig.clip import Clip, RootTrack
from ggs.rig.pose import BoneDef, bone_defs, root_scale
from ggs.rig.puppet import PartImage, Puppet
from ggs.rig.retarget import LimbGroup, limb_groups
from ggs.rig.spec import RigSpec

COORDS = ("screen pixels, +x right, +y down; angles in degrees clockwise from +y (down = 0, right = -90); "
          "root positions are relative to the ground point under the character")


def _scaled_part(img, pivot, factor: float):
    from PIL import Image

    rgba = img.convert("RGBa")
    w, h = max(1, round(img.width * factor)), max(1, round(img.height * factor))
    rgba = rgba.resize((w, h), Image.LANCZOS).convert("RGBA")
    fx, fy = w / img.width, h / img.height
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("empty part image")
    x0, y0 = max(0, bbox[0] - 1), max(0, bbox[1] - 1)
    x1, y1 = min(w, bbox[2] + 1), min(h, bbox[3] + 1)
    return rgba.crop((x0, y0, x1, y1)), (pivot[0] * fx - x0, pivot[1] * fy - y0), fx


def write_rig_package(rig: RigSpec, character: Character, clips: dict[str, Clip], out_dir: str | Path,
                      height_px: int, padding: int = 2, items: list | None = None) -> Path:
    """``items``: held equipment (``ggs.rig.equipment.Item``), packed into the atlas and listed in rig.json
    with the bone, socket and hold angle an engine needs to place them."""
    from PIL import Image

    character.check(rig)
    out_dir = Path(out_dir)
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)
    sheet = Image.open(character.sheet_path).convert("RGBA")
    cell = rig.part_sheet.cell_px
    ch = character.spec
    lengths = ch.bone_lengths

    entries = []  # (name, kind, part or variant, image, pivot)
    for p in rig.part_sheet.parts:
        entries.append((p.name, "part", p, p.cell, character.pivot(p), lengths[p.bone]))
    head = next((p for p in rig.part_sheet.parts if p.name == "head"), None)
    if head is not None:
        for v in rig.part_sheet.head_variants:
            entries.append((v.name, "head_variant", head, v.cell, character.pivot(head), lengths[head.bone]))

    scaled = []
    for it in items or []:  # held items: scaled to the package's pixels per standing height
        img = it.load_image()
        f = height_px / it.spec.px_per_height
        crop, piv, _ = _scaled_part(img, it.spec.grip, f)
        scaled.append((it.spec.name, "item", it, crop, piv, 0.0))
    for name, kind, part, (col, row), pivot, length in entries:
        img = sheet.crop((col * cell, row * cell, (col + 1) * cell, (row + 1) * cell))
        art_len = length * ch.px_per_height
        factor = (length * height_px) / art_len
        crop, piv, fx = _scaled_part(img, (pivot[0] * cell, pivot[1] * cell), factor)
        scaled.append((name, kind, part, crop, piv, art_len * fx))

    (sw, sh), positions = shelf_pack([c[3].size for c in scaled], padding)
    atlas = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    parts_json, variants_json, items_json = [], [], []
    for (name, kind, part, crop, piv, length_px), (x, y) in zip(scaled, positions):
        paste_extruded(atlas, crop, (x, y), min(1, padding))
        rect = {"x": x, "y": y, "w": crop.width, "h": crop.height}
        pivot = {"x": round(piv[0], 3), "y": round(piv[1], 3)}
        if kind == "item":
            sock = next(s for s in rig.sockets if s.name == part.spec.socket)
            items_json.append({"name": name, "bone": sock.bone, "socket": sock.name, "socket_t": sock.t,
                               "socket_offset": sock.offset, "rect": rect, "pivot": pivot,
                               "art_angle": part.spec.art_angle, "hold_angle": part.spec.hold_angle,
                               "hold": part.spec.hold})
        elif kind == "part":
            bones = [part.bone] + ([part.bone_far] if part.bone_far else [])
            parts_json.append({"name": name, "bones": bones, "rect": rect, "pivot": pivot,
                               "art_angle": part.art_angle, "length_px": round(length_px, 3),
                               "far_tint": part.far_tint})
        else:
            variants_json.append({"name": name, "part": part.name, "rect": rect, "pivot": pivot})
    atlas.save(out_dir / "parts.png")

    by_bone = {b: p for p in rig.part_sheet.parts for b in (p.bone, p.bone_far) if b}
    rest_angles = {b.name: by_bone[b.name].art_angle for b in rig.bones}
    root = next(b.name for b in rig.bones if b.parent is None)
    leg = [g for g in limb_groups(rig) if rig.bone(g.near[0]).parent == root]
    hip = sum(lengths[b] for b in leg[0].near[:2]) if leg else 0.5
    doc = {
        "format": "ggs.rig_package",
        "version": 1,
        "generator": f"ggs {__version__}",
        "name": ch.name,
        "rig": rig.id,
        "view": rig.view,
        "facing": rig.facing,
        "coordinates": COORDS,
        "height_px": height_px,
        "image": "parts.png",
        "image_size": {"w": sw, "h": sh},
        "bones": [{"name": b.name, "parent": b.parent, "length": round(lengths[b.name] * height_px, 3),
                   "attach": b.attach, "attach_side_px": round(b.attach_side * height_px, 3)} for b in rig.bones],
        "rest_pose": {"root": {"x": 0.0, "y": round(-hip * height_px, 3)},
                      "angles": rest_angles},
        "parts": parts_json,
        "head_variants": variants_json,
        "items": items_json,
        "draw_order": list(character.spec.draw_order or rig.draw_order),
        "limb_groups": [{"name": g.name, "near": list(g.near), "far": list(g.far)}
                        for g in limb_groups(rig)],
        "sockets": [s.model_dump() for s in rig.sockets],
        "clips": sorted(clips),
    }
    (out_dir / "rig.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")

    bones = bone_defs(rig)
    for cname, clip in clips.items():
        k = root_scale(bones, clip, lengths) * height_px
        xs = list(clip.root.x)
        if clip.loop:  # remove the steady travel so the loop is seamless in place; keep the natural sway
            n = clip.frame_count
            xs = [x - clip.loop_travel * i / n for i, x in enumerate(xs)]
        cdoc = {
            "format": "ggs.rig_clip",
            "version": 1,
            "name": cname,
            "fps": clip.fps,
            "frame_count": clip.frame_count,
            "loop": clip.loop,
            "loop_travel_px": round(clip.loop_travel * k, 3),
            "root": {"x": [round(v * k, 3) for v in xs], "y": [round(v * k, 3) for v in clip.root.y]},
            "angles": clip.angles,
            "local_angles": clip.local_angles,
            "swaps": clip.swaps,
        }
        (out_dir / "clips" / f"{cname}.json").write_text(json.dumps(cdoc, separators=(",", ":")),
                                                        encoding="utf-8")
    return out_dir


def load_rig_package(path: str | Path) -> tuple[Puppet, dict[str, Clip], dict]:
    """Rebuild the puppet and clips from a package (the reference for engine importers)."""
    from PIL import Image

    path = Path(path)
    doc = json.loads((path / "rig.json").read_text(encoding="utf-8"))
    if doc.get("format") != "ggs.rig_package" or doc.get("version") != 1:
        raise ValueError(f"{path}: not a ggs.rig_package v1")
    H = float(doc["height_px"])
    atlas = Image.open(path / doc["image"]).convert("RGBA")
    h = doc.get("height_px") or 1.0
    bones = [BoneDef(b["name"], b["parent"], b["attach"], b.get("attach_side_px", 0.0) / h) for b in doc["bones"]]
    lengths = {b["name"]: b["length"] / H for b in doc["bones"]}
    parts: dict[str, PartImage] = {}
    for p in doc["parts"]:
        r = p["rect"]
        img = atlas.crop((r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]))
        for i, bone in enumerate(p["bones"]):
            parts[bone] = PartImage(img, (p["pivot"]["x"], p["pivot"]["y"]), p["art_angle"],
                                    p["length_px"], 1.0 if i == 0 else p["far_tint"])
    groups = [LimbGroup(g["name"], tuple(g["near"]), tuple(g["far"])) for g in doc["limb_groups"]]
    puppet = Puppet(bones, lengths, doc["draw_order"], groups, parts)
    clips = {}
    for cname in doc["clips"]:
        c = json.loads((path / "clips" / f"{cname}.json").read_text(encoding="utf-8"))
        clips[cname] = Clip(
            rig=doc["rig"], source=str(path), profile="rig_package", fps=c["fps"],
            frame_count=c["frame_count"], loop=c["loop"], loop_travel=c["loop_travel_px"] / H,
            bones=[b.name for b in bones], bone_lengths=lengths, angles=c["angles"],
            local_angles=c["local_angles"],
            root=RootTrack(x=[v / H for v in c["root"]["x"]], y=[v / H for v in c["root"]["y"]]),
            swaps=c["swaps"],
        )
    return puppet, clips, doc
