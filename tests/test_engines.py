"""Engine exporters, checked with small "virtual engines": parse the generated files, apply each engine's
transform rules (Godot: +y down, clockwise; Unity/Defold: +y up, counter-clockwise), and compare every bone
with the in-app pose. Real-engine smoke tests live in docs/EXPORT_FORMATS.md (manual)."""

import cmath
import json
import math
import re

import pytest

pytest.importorskip("PIL")

from bvh_synth import walk_bvh

from ggs.cli import main
from ggs.export.engines.common import load_package, local_unwrapped
from ggs.export.engines.defold import write_defold
from ggs.export.engines.godot import write_godot
from ggs.export.engines.unity import write_unity
from ggs.export.rig_package import load_rig_package, write_rig_package
from ggs.paths import rigs_dir
from ggs.rig import load_rig_spec, parse_bvh, retarget
from ggs.rig.character import load_character, make_programmer_art
from ggs.rig.clip import RootTrack
from ggs.rig.cliptools import cut_loop, find_loop
from ggs.rig.pose import pose_bones

H = 128
TOL_PX = 0.05
TOL_DEG = 0.05


@pytest.fixture(scope="module")
def pkg_dir(tmp_path_factory):
    rig = load_rig_spec(rigs_dir() / "humanoid_side_v1.yaml")
    tmp = tmp_path_factory.mktemp("eng")
    character = load_character(make_programmer_art(rig, tmp / "pa"))
    take = retarget(parse_bvh(walk_bvh(frames=90, fps=30.0)), rig, "cmu_bvh", drop_tpose_frame=False)
    s, e, _ = find_loop(take, 0.6, 1.6)
    walk = cut_loop(take, s, e)
    # a second, non-looping clip whose far leg crosses in front (exercises draw-order swaps)
    n = 6
    cross = walk.model_copy(update={
        "frame_count": n, "loop": False, "loop_travel": 0.0,
        "angles": {b: v[:n] for b, v in walk.angles.items()},
        "local_angles": {b: v[:n] for b, v in walk.local_angles.items()},
        "root": RootTrack(x=walk.root.x[:n], y=walk.root.y[:n]),
        "swaps": {"thigh": [False, False, True, True, True, False], "upper_arm": [False] * n},
    })
    out = write_rig_package(rig, character, {"walk": walk, "cross": cross}, tmp / "pkg", H)
    write_godot(out)
    write_unity(out)
    write_defold(out, "/chars/hero")
    return out


def expected(pkg_dir):
    """In-app pose per clip/frame: bone -> (start complex px, y-down; absolute angle deg)."""
    puppet, clips, _ = load_rig_package(pkg_dir)
    out = {}
    for name, clip in clips.items():
        frames = []
        for f in range(clip.frame_count):
            pose = pose_bones(puppet.bones, clip, f, puppet.lengths)
            frames.append({b: (complex(s[0] * H, s[1] * H), clip.angles[b][f]) for b, (s, _) in pose.items()})
        out[name] = frames
    return out


def same_angle(a, b):
    return abs((a - b + 180) % 360 - 180) < TOL_DEG


def compose(parent, local):
    """2D similarity transforms as (origin complex, rotation unit complex, scale)."""
    p0, r0, s0 = parent
    p1, r1, s1 = local
    return p0 + r0 * s0 * p1, r0 * r1, s0 * s1


# ------------------------------------------------------------------ Godot


def parse_tscn(text):
    nodes, anims = {}, {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^\[(\w+) (.*)\]$", line)
        if m:
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(2)))
            if m.group(1) == "node":
                path = attrs["name"] if attrs.get("parent") in (None, ".") else f"{attrs['parent']}/{attrs['name']}"
                if "parent" not in attrs:
                    path = ""
                cur = nodes.setdefault(path, {"type": attrs["type"], "props": {}})
            elif m.group(1) == "sub_resource" and attrs["type"] == "Animation":
                cur = anims.setdefault(attrs["id"], {"paths": [], "values": [], "props": {}})
            else:
                cur = {"props": {}}
            continue
        if cur is None:
            continue
        m = re.match(r'^tracks/\d+/path = NodePath\("(.*)"\)$', line)
        if m:
            cur["paths"].append(m.group(1))
            continue
        if line.startswith('"values": '):
            cur["values"].append(line[len('"values": '):])
            continue
        m = re.match(r"^([\w/]+) = (.*)$", line)
        if m:
            cur["props"][m.group(1)] = m.group(2)
    return nodes, anims


def vec2(s):
    x, y = re.match(r"Vector2\(([^,]+), ([^)]+)\)", s).groups()
    return complex(float(x), float(y))


def test_godot_scene_matches_in_app_pose(pkg_dir):
    pkg = load_package(pkg_dir)
    text = (pkg_dir / "programmer_art.tscn").read_text(encoding="utf-8")
    assert text.startswith("[gd_scene") and 'path="parts.png"' in text
    nodes, anims = parse_tscn(text)
    exp = expected(pkg_dir)
    for anim in anims.values():
        name = anim["props"]["resource_name"].strip('"')
        tracks = dict(zip(anim["paths"], anim["values"]))
        for f, want in enumerate(exp[name]):
            world = {}
            for b in pkg.ordered_bones():
                path = "/".join(pkg.bone_path(b))
                props = nodes[path]["props"]
                pos = vec2(props["position"])
                if f"{path}:position" in tracks:
                    pos = [vec2(v) for v in re.findall(r"Vector2\([^)]*\)", tracks[f"{path}:position"])][f]
                rot = float(json.loads(tracks[f"{path}:rotation"])[f])
                parent = pkg.bone(b)["parent"]
                base = world[parent] if parent else (0j, 1 + 0j, 1.0)
                world[b] = compose(base, (pos, cmath.exp(1j * rot), 1.0))
                origin, r, _ = world[b]
                # Godot: +y down, rotation clockwise, bone axis = local +y -> angle = rotation
                assert abs(origin - want[b][0]) < TOL_PX, (name, f, b)
                assert same_angle(math.degrees(cmath.phase(r)), want[b][1]), (name, f, b)
                art = nodes[f"{path}/art"]["props"]
                part, far = pkg.part_for(b)
                assert float(art["rotation"]) == pytest.approx(-math.radians(part["art_angle"]), abs=1e-5)
                assert vec2(art["offset"]) == pytest.approx(complex(-part["pivot"]["x"], -part["pivot"]["y"]))
                assert ("modulate" in art) == far


def test_godot_swaps_animate_z_index(pkg_dir):
    nodes, anims = parse_tscn((pkg_dir / "programmer_art.tscn").read_text(encoding="utf-8"))
    cross = next(a for a in anims.values() if a["props"]["resource_name"] == '"cross"')
    tracks = dict(zip(cross["paths"], cross["values"]))
    near = json.loads(tracks["pelvis/thigh_near/art:z_index"])
    far = json.loads(tracks["pelvis/thigh_far/art:z_index"])
    assert [f > n for f, n in zip(far, near)] == [False, False, True, True, True, False]
    walk = next(a for a in anims.values() if a["props"]["resource_name"] == '"walk"')
    assert not any(p.endswith(":z_index") for p in walk["paths"])
    assert walk["props"]["loop_mode"] == "1" and cross["props"]["loop_mode"] == "0"
    assert nodes["pelvis/torso/head/art_blink"]["props"]["visible"] == "false"


# ------------------------------------------------------------------ Unity


def test_unity_rig_matches_in_app_pose(pkg_dir):
    data = json.loads((pkg_dir / "programmer_art.ggsrig.json").read_text(encoding="utf-8"))
    assert data["format"] == "ggs.unity_rig"
    src = (pkg_dir / "GgsRigPlayer.cs").read_text(encoding="utf-8")
    assert "class GgsRigPlayer : MonoBehaviour" in src and src.count("{") == src.count("}")
    bones = {b["name"]: b for b in data["bones"]}
    assert data["bones"][0]["parent"] == ""
    exp = expected(pkg_dir)
    ppu = 100.0
    for clip in data["clips"]:
        local = {t["bone"]: t["values"] for t in clip["local"]}
        for f, want in enumerate(exp[clip["name"]]):
            world = {}
            for b in data["bones"]:  # mirrors GgsRigPlayer.Build/Apply (y up, CCW degrees)
                name = b["name"]
                if b["parent"]:
                    pos = complex(0, -b["attach"] * bones[b["parent"]]["length"] / ppu)
                    base = world[b["parent"]]
                else:
                    pos = complex(clip["rootX"][f] / ppu, -clip["rootY"][f] / ppu)
                    base = (0j, 1 + 0j, 1.0)
                world[name] = compose(base, (pos, cmath.exp(1j * math.radians(-local[name][f])), 1.0))
                origin, r, _ = world[name]
                got = complex(origin.real * ppu, -origin.imag * ppu)  # back to package pixels, y down
                assert abs(got - want[name][0]) < TOL_PX, (clip["name"], f, name)
                assert same_angle(-math.degrees(cmath.phase(r)), want[name][1]), (clip["name"], f, name)
    cross = next(c for c in data["clips"] if c["name"] == "cross")
    idx = {t["bone"]: t["values"] for t in cross["drawIndex"]}
    assert [a > b for a, b in zip(idx["thigh_far"], idx["thigh_near"])] == [False, False, True, True, True, False]


def test_unity_sprite_pivots(pkg_dir):
    data = json.loads((pkg_dir / "programmer_art.ggsrig.json").read_text(encoding="utf-8"))
    for p in data["parts"]:
        r = p["rect"]
        # Unity: rect from bottom-left, pivot normalised from bottom-left (GgsRigPlayer.MakeSprite)
        uy = data["atlasH"] - r["y"] - r["h"]
        assert 0 <= uy and uy + r["h"] <= data["atlasH"]
        piv = (p["pivotX"] / r["w"], 1 - p["pivotY"] / r["h"])
        assert 0 <= piv[0] <= 1 and 0 <= piv[1] <= 1


# ------------------------------------------------------------------ Defold


def parse_collection(text):
    inst, cur, block = {}, None, None
    for line in text.splitlines():
        if line == "embedded_instances {":
            cur = {"children": [], "position": {}, "rotation": {}, "scale3": {}}
            continue
        if cur is None:
            continue
        m = re.match(r'^  id: "(.*)"$', line)
        if m:
            cur["id"] = m.group(1)
            inst[m.group(1)] = cur
            continue
        m = re.match(r'^  children: "(.*)"$', line)
        if m:
            cur["children"].append(m.group(1))
            continue
        if line.startswith("  data: "):
            cur["data"] = line[len("  data: "):]
            continue
        m = re.match(r"^  (position|rotation|scale3) \{$", line)
        if m:
            block = m.group(1)
            continue
        m = re.match(r"^    ([xyzw]): (.*)$", line)
        if m and block:
            cur[block][m.group(1)] = float(m.group(2))
            continue
        if line == "}":
            cur, block = None, None
    return inst


def test_defold_collection_matches_in_app_pose(pkg_dir):
    pkg = load_package(pkg_dir)
    text = (pkg_dir / "programmer_art.collection").read_text(encoding="utf-8")
    inst = parse_collection(text)
    parent = {c: i["id"] for i in inst.values() for c in i["children"]}

    def local(iid):
        i = inst[iid]
        p = complex(i["position"]["x"], i["position"]["y"])
        r = complex(i["rotation"]["w"], i["rotation"]["z"]) ** 2  # unit quaternion about z -> e^(i angle)
        return p, r, i["scale3"]["x"]

    def world(iid):
        t = local(iid)
        while iid in parent:
            iid = parent[iid]
            t = compose(local(iid), t)
        return t

    want = expected(pkg_dir)["walk"][0]  # the collection holds frame 0 of the first clip
    atlas_names = re.findall(r'image: "/chars/hero/parts/(\w+)\.png"', (pkg_dir / "programmer_art.atlas").read_text())
    for b in pkg.ordered_bones():
        origin, r, _ = world(b)
        assert abs(complex(origin.real, -origin.imag) - want[b][0]) < TOL_PX, b
        assert same_angle(-math.degrees(cmath.phase(r)), want[b][1]), b
        # the sprite (centred on <bone>_art) must put its pivot pixel on the bone origin, art along the bone
        part, _ = pkg.part_for(b)
        ap, ar, s = world(f"{b}_art")
        rect, piv = part["rect"], part["pivot"]
        pivot_local = complex(piv["x"] - rect["w"] / 2, -(piv["y"] - rect["h"] / 2))
        assert abs(ap + ar * s * pivot_local - origin) < TOL_PX, b
        a0 = math.radians(part["art_angle"])
        art_dir = ar * complex(-math.sin(a0), -math.cos(a0))  # art direction in y-up sprite space
        bone_dir = complex(-math.sin(math.radians(want[b][1])), -math.cos(math.radians(want[b][1])))
        assert abs(art_dir - bone_dir) < 1e-3, b
        assert f'default_animation: \\\\\\"{part["name"]}\\\\\\"' in inst[f"{b}_art"]["data"]
        assert part["name"] in atlas_names
    # z follows the draw order
    zs = {b: inst[f"{b}_art"]["position"]["z"] for b in pkg.ordered_bones()}
    assert sorted(zs, key=zs.get) == pkg.doc["draw_order"]


def test_defold_lua_and_script(pkg_dir):
    pkg = load_package(pkg_dir)
    lua = (pkg_dir / "programmer_art_rig.lua").read_text(encoding="utf-8")
    script = (pkg_dir / "programmer_art_rig.script").read_text(encoding="utf-8")
    assert 'require "chars.hero.programmer_art_rig"' in script
    assert lua.count("{") == lua.count("}") and lua.lstrip().startswith("--")
    local = local_unwrapped(pkg, pkg.clips["walk"])
    m = re.search(r"pelvis = \{([^}]*)\}", lua.split("angles = ")[1])
    assert [float(v) for v in m.group(1).split(",")][:3] == pytest.approx(local["pelvis"][:3], abs=1e-3)
    for f in ("thigh", "head_blink"):
        assert (pkg_dir / "parts" / f"{f}.png").exists()
    coll = (pkg_dir / "programmer_art.collection").read_text(encoding="utf-8")
    assert '/chars/hero/programmer_art_rig.script' in coll and "/chars/hero/programmer_art.atlas" in coll


def test_cli_engine_exports(tmp_path):
    bvh = tmp_path / "walk.bvh"
    bvh.write_text(walk_bvh(frames=60, fps=30.0), encoding="utf-8")
    take, loop = tmp_path / "take.json", tmp_path / "walk.json"
    assert main(["mocap", "retarget", str(bvh), "--profile", "cmu_bvh", "--out", str(take)]) == 0
    assert main(["clip", "loop", str(take), "--out", str(loop)]) == 0
    assert main(["character", "programmer-art", "--out", str(tmp_path / "pa")]) == 0
    base = ["export", str(loop), "--character", str(tmp_path / "pa" / "character.yaml"), "--height", "96"]
    for fmt, marker in (("godot", "programmer_art.tscn"), ("unity", "GgsRigPlayer.cs"),
                        ("defold", "programmer_art.collection")):
        out = tmp_path / fmt
        assert main([*base, "--format", fmt, "--out", str(out)]) == 0
        assert (out / marker).exists() and (out / f"README_{fmt.upper()}.txt").exists()
        assert (out / "rig.json").exists()
    assert main([*base, "--format", "defold", "--defold-path", "relative/path", "--out", str(tmp_path / "d2")]) == 1


def test_loop_root_is_in_place(pkg_dir):
    walk = json.loads((pkg_dir / "clips" / "walk.json").read_text(encoding="utf-8"))
    xs = walk["root"]["x"]
    assert walk["loop_travel_px"] > 0
    assert max(xs) - min(xs) < 0.1 * walk["loop_travel_px"]  # steady travel removed, small sway left


# ------------------------------------------------------------------ held items (M7)


@pytest.fixture(scope="module")
def item_pkg(tmp_path_factory):
    from PIL import Image, ImageDraw

    from ggs.rig.equipment import load_item, make_item

    rig = load_rig_spec(rigs_dir() / "humanoid_side_v1.yaml")
    tmp = tmp_path_factory.mktemp("item")
    bar = Image.new("RGBA", (24, 200), (0, 0, 0, 0))
    ImageDraw.Draw(bar).rectangle([8, 0, 15, 199], fill=(150, 150, 160, 255))
    bar.save(tmp / "bar.png")
    item = load_item(make_item(tmp / "bar.png", tmp / "sword", name="sword", length=0.45))
    character = load_character(make_programmer_art(rig, tmp / "pa"))
    take = retarget(parse_bvh(walk_bvh(frames=60, fps=30.0)), rig, "cmu_bvh", drop_tpose_frame=False)
    out = write_rig_package(rig, character, {"walk": take}, tmp / "pkg", H, items=[item])
    write_godot(out)
    write_unity(out)
    write_defold(out, "/chars/hero")
    return out, rig, item


def test_items_in_package_and_godot(item_pkg):
    from ggs.rig.equipment import socket_point

    pkg_dir, rig, _ = item_pkg
    pkg = load_package(pkg_dir)
    (it,) = pkg.doc["items"]
    assert it["bone"] == "hand_near" and it["hold_angle"] == -90
    nodes, anims = parse_tscn((pkg_dir / "programmer_art.tscn").read_text(encoding="utf-8"))
    anim = next(iter(anims.values()))
    tracks = dict(zip(anim["paths"], anim["values"]))
    puppet, clips, _ = load_rig_package(pkg_dir)
    clip = clips["walk"]
    for f in (0, 5, 17):
        world = {}
        for b in pkg.ordered_bones():
            path = "/".join(pkg.bone_path(b))
            pos = vec2(nodes[path]["props"]["position"])
            if f"{path}:position" in tracks:
                pos = [vec2(v) for v in re.findall(r"Vector2\([^)]*\)", tracks[f"{path}:position"])][f]
            rot = float(json.loads(tracks[f"{path}:rotation"])[f])
            parent = pkg.bone(b)["parent"]
            world[b] = compose(world[parent] if parent else (0j, 1 + 0j, 1.0), (pos, cmath.exp(1j * rot), 1.0))
        hand_path = "/".join(pkg.bone_path("hand_near"))
        node = nodes[f"{hand_path}/item_sword"]["props"]
        origin, r, _ = compose(world["hand_near"], (vec2(node["position"]), cmath.exp(1j * float(node["rotation"])),
                                                     1.0))
        pose = pose_bones(puppet.bones, clip, f, puppet.lengths)
        (gx, gy), _ = socket_point(rig, "grip_near", pose)
        assert abs(origin - complex(gx * H, gy * H)) < TOL_PX
        want = clip.angles["hand_near"][f] + it["hold_angle"]
        assert same_angle(math.degrees(cmath.phase(r)) + it["art_angle"], want)


def test_items_in_unity_and_defold(item_pkg):
    pkg_dir, _, _ = item_pkg
    unity = json.loads(next(pkg_dir.glob("*.ggsrig.json")).read_text(encoding="utf-8"))
    (u,) = unity["items"]
    assert u["bone"] == "hand_near" and u["socketT"] == 0.5 and u["holdAngle"] == -90
    assert "GgsItem" in (pkg_dir / "GgsRigPlayer.cs").read_text(encoding="utf-8")
    coll = (pkg_dir / "programmer_art.collection").read_text(encoding="utf-8")
    assert 'id: "hand_near_item_sword"' in coll and 'children: "hand_near_item_sword"' in coll
    assert (pkg_dir / "parts" / "item_sword.png").exists()
    assert "item_sword.png" in (pkg_dir / "programmer_art.atlas").read_text(encoding="utf-8")


def test_front_rig_side_attachments_in_godot(tmp_path):
    """Front-view rig: shoulders and hips sit beside their parent (attach_side); Godot must place them the same."""
    from ggs.rig.master import pose_clip
    from ggs.rig.poses import pose_angles

    rig = load_rig_spec(rigs_dir() / "humanoid_front_v1.yaml")
    character = load_character(make_programmer_art(rig, tmp_path / "pa", px_per_height=400))
    clip = pose_clip(rig, pose_angles("front", "t_pose"), character.spec.bone_lengths)
    clip.root.y = [-0.5]
    out = write_rig_package(rig, character, {"tpose": clip}, tmp_path / "pkg", H)
    write_godot(out)
    pkg = load_package(out)
    nodes, anims = parse_tscn(next(out.glob("*.tscn")).read_text(encoding="utf-8"))
    anim = next(iter(anims.values()))
    tracks = dict(zip(anim["paths"], anim["values"]))
    want = expected(out)["tpose"][0]
    world = {}
    for b in pkg.ordered_bones():
        path = "/".join(pkg.bone_path(b))
        pos = vec2(nodes[path]["props"]["position"])
        if f"{path}:position" in tracks:
            pos = next(vec2(v) for v in re.findall(r"Vector2\([^)]*\)", tracks[f"{path}:position"]))
        rot = float(json.loads(tracks[f"{path}:rotation"])[0])
        parent = pkg.bone(b)["parent"]
        world[b] = compose(world[parent] if parent else (0j, 1 + 0j, 1.0), (pos, cmath.exp(1j * rot), 1.0))
        assert abs(world[b][0] - want[b][0]) < TOL_PX, b
    # T-pose: the shoulders are mirror images about the spine
    l, r = want["upper_arm_l"][0], want["upper_arm_r"][0]
    spine_x = want["head"][0].real
    assert abs((spine_x - l.real) - (r.real - spine_x)) < 0.5 and l.real < spine_x < r.real
