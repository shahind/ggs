"""Godot 4 exporter: a plain-text ``.tscn`` scene next to the rig package. No editor plugin needed.

Scene layout (Godot 2D is +y down, rotation clockwise, so package values map 1:1, in radians):

    <Name> (Node2D)                 origin = ground point under the character
      pelvis (Node2D)               position animated (root track)
        art (Sprite2D)              region of parts.png, pivot at the node origin
        torso (Node2D)              position = (0, attach * parent length), rotation = local angle
          ...
      AnimationPlayer               one Animation per clip, autoplays the first

A bone node's local +y axis points along the bone. Each ``art`` sprite is rotated by -art_angle so the drawn
bone lines up with that axis. Sprites use absolute z_index = draw order; clips with near/far swaps animate it.
"""

from __future__ import annotations

import math
from pathlib import Path

from ggs.export.engines.common import (
    Package,
    base_draw_index,
    draw_index_tracks,
    ident,
    load_package,
    local_unwrapped,
)

GODOT_MIN = "4.2"


def _f(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _v2(x: float, y: float) -> str:
    return f"Vector2({_f(x)}, {_f(y)})"


def _node_path(pkg: Package, bone: str) -> str:
    return "/".join(pkg.bone_path(bone))


def _parent_attr(pkg: Package, bone: str) -> str:
    chain = pkg.bone_path(bone)[:-1]
    return "." if not chain else "/".join(chain)


def _sprite_lines(name: str, parent: str, part: dict, rect: dict, pivot: dict, length: float,
                  tint: float, z: int, visible: bool = True) -> list[str]:
    s = length / part["length_px"] if part["length_px"] > 0 else 1.0
    lines = [
        f'[node name="{name}" type="Sprite2D" parent="{parent}"]',
        f"z_index = {z}",
        "z_as_relative = false",
        f"rotation = {_f(-math.radians(part['art_angle']))}",
    ]
    if abs(s - 1.0) > 1e-6:
        lines.append(f"scale = {_v2(s, s)}")
    if tint != 1.0:
        lines.append(f"modulate = Color({_f(tint)}, {_f(tint)}, {_f(tint)}, 1)")
    if not visible:
        lines.append("visible = false")
    lines += [
        'texture = ExtResource("1_parts")',
        "centered = false",
        f"offset = {_v2(-pivot['x'], -pivot['y'])}",
        "region_enabled = true",
        f"region_rect = Rect2({rect['x']}, {rect['y']}, {rect['w']}, {rect['h']})",
        "",
    ]
    return lines


def _animation(pkg: Package, cname: str, clip: dict, rid: str) -> list[str]:
    n, fps = clip["frame_count"], clip["fps"]
    times = "PackedFloat32Array(" + ", ".join(_f(i / fps) for i in range(n)) + ")"
    ones = "PackedFloat32Array(" + ", ".join("1" for _ in range(n)) + ")"
    local = local_unwrapped(pkg, clip)
    tracks: list[tuple[str, int, int, str]] = []  # path, interp, update, values
    root = pkg.root
    pos = ", ".join(_v2(x, y) for x, y in zip(clip["root"]["x"], clip["root"]["y"]))
    tracks.append((f"{_node_path(pkg, root)}:position", 1, 0, f"[{pos}]"))
    for b in pkg.ordered_bones():
        vals = ", ".join(_f(math.radians(a)) for a in local[b])
        tracks.append((f"{_node_path(pkg, b)}:rotation", 3, 0, f"[{vals}]"))  # 3 = linear angle
    for b, idx in draw_index_tracks(pkg, clip).items():
        vals = ", ".join(str(i) for i in idx)
        tracks.append((f"{_node_path(pkg, b)}/art:z_index", 0, 1, f"[{vals}]"))  # discrete
    lines = [
        f'[sub_resource type="Animation" id="{rid}"]',
        f'resource_name = "{cname}"',
        f"length = {_f(n / fps)}",
        f"loop_mode = {1 if clip['loop'] else 0}",
        f"step = {_f(1 / fps)}",
    ]
    for i, (path, interp, update, values) in enumerate(tracks):
        lines += [
            f'tracks/{i}/type = "value"',
            f"tracks/{i}/imported = false",
            f"tracks/{i}/enabled = true",
            f'tracks/{i}/path = NodePath("{path}")',
            f"tracks/{i}/interp = {interp}",
            f"tracks/{i}/loop_wrap = true",
            f"tracks/{i}/keys = {{",
            f'"times": {times},',
            f'"transitions": {ones},',
            f'"update": {update},',
            f'"values": {values}',
            "}",
        ]
    lines.append("")
    return lines


def write_godot(package_dir: str | Path) -> Path:
    """Write ``<name>.tscn`` into the rig package folder. Copy the whole folder into the Godot project."""
    pkg = load_package(package_dir)
    name = ident(pkg.name)
    z0 = base_draw_index(pkg)
    clip_names = list(pkg.clips)
    anim_ids = {c: f"Animation_{ident(c)}" for c in clip_names}
    load_steps = 1 + 1 + len(clip_names) + 1  # ext texture + animations + library + 1

    out = [f"[gd_scene load_steps={load_steps} format=3]", "",
           '[ext_resource type="Texture2D" path="parts.png" id="1_parts"]', ""]
    for c in clip_names:
        out += _animation(pkg, c, pkg.clips[c], anim_ids[c])
    out += ['[sub_resource type="AnimationLibrary" id="AnimationLibrary_ggs"]', "_data = {"]
    out += [",\n".join(f'"{c}": SubResource("{anim_ids[c]}")' for c in clip_names), "}", ""]

    out += [f'[node name="{name}" type="Node2D"]', ""]
    first = pkg.clips[clip_names[0]] if clip_names else None
    first_local = local_unwrapped(pkg, first) if first else {}
    for b in pkg.ordered_bones():
        bone = pkg.bone(b)
        parent = bone["parent"]
        if parent is None:
            pos = (first["root"]["x"][0], first["root"]["y"][0]) if first else (
                pkg.doc["rest_pose"]["root"]["x"], pkg.doc["rest_pose"]["root"]["y"])
            rot = first_local[b][0] if first else pkg.doc["rest_pose"]["angles"][b]
        else:
            pos = (bone.get("attach_side_px", 0.0), bone["attach"] * pkg.bone(parent)["length"])
            rot = first_local[b][0] if first else (
                pkg.doc["rest_pose"]["angles"][b] - pkg.doc["rest_pose"]["angles"][parent])
        out += [f'[node name="{b}" type="Node2D" parent="{_parent_attr(pkg, b)}"]',
                f"position = {_v2(*pos)}", f"rotation = {_f(math.radians(rot))}", ""]
        part, far = pkg.part_for(b)
        out += _sprite_lines("art", _node_path(pkg, b), part, part["rect"], part["pivot"], bone["length"],
                             part["far_tint"] if far else 1.0, z0[b])
        if part["name"] == "head" and not far:
            for v in pkg.doc.get("head_variants", []):
                out += _sprite_lines(f"art_{v['name']}", _node_path(pkg, b), part, v["rect"], v["pivot"],
                                     bone["length"], 1.0, z0[b], visible=False)
        for it in pkg.doc.get("items", []):
            if it["bone"] == b:  # held item: fixed on the hand at its socket, on top of every part
                length = bone["length"]
                out += [f'[node name="item_{ident(it["name"])}" type="Sprite2D" parent="{_node_path(pkg, b)}"]',
                        f"z_index = {len(pkg.doc['draw_order'])}", "z_as_relative = false",
                        f"position = {_v2(-it['socket_offset'] * length, it['socket_t'] * length)}",
                        f"rotation = {_f(math.radians(it['hold_angle'] - it['art_angle']))}",
                        'texture = ExtResource("1_parts")', "centered = false",
                        f"offset = {_v2(-it['pivot']['x'], -it['pivot']['y'])}", "region_enabled = true",
                        (f"region_rect = Rect2({it['rect']['x']}, {it['rect']['y']}, {it['rect']['w']}, "
                         f"{it['rect']['h']})"), ""]
    if clip_names:
        out += ['[node name="AnimationPlayer" type="AnimationPlayer" parent="."]',
                'libraries = {', '"": SubResource("AnimationLibrary_ggs")', "}",
                f'autoplay = "{clip_names[0]}"', ""]

    path = Path(package_dir) / f"{name}.tscn"
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")
    (Path(package_dir) / "README_GODOT.txt").write_text(
        f"Godot {GODOT_MIN}+ : copy this whole folder into your project (anywhere under res://) and open\n"
        f"{name}.tscn. The scene origin is the ground point under the character; it autoplays\n"
        f"'{clip_names[0] if clip_names else ''}'. Animations: {', '.join(clip_names)}.\n"
        "Head variants (blink, mouth_open, hurt) are hidden sprites under the head node: toggle 'visible'.\n"
        "Root motion: move the scene by loop_travel_px (see clips/*.json) per loop.\n",
        encoding="utf-8", newline="\n")
    return path
