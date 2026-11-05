import json
import math

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from bvh_synth import walk_bvh
from PIL import Image

from ggs.cli import main
from ggs.export.animated import save_apng, save_gif, save_webp
from ggs.export.pack import grid_layout, next_pot, shelf_pack
from ggs.export.rig_package import load_rig_package, write_rig_package
from ggs.export.sheet import save_sprite_sheet
from ggs.paths import rigs_dir
from ggs.rig import load_rig_spec, parse_bvh, retarget
from ggs.rig.character import PANTS, SKIN, load_character, make_programmer_art
from ggs.rig.clip import Clip, RootTrack
from ggs.rig.cliptools import cut_loop, find_loop
from ggs.rig.pose import BoneDef, pose_bones
from ggs.rig.puppet import PartImage, Puppet, frame_durations_ms

H = 128


@pytest.fixture(scope="module")
def rig():
    return load_rig_spec(rigs_dir() / "humanoid_side_v1.yaml")


@pytest.fixture(scope="module")
def character(rig, tmp_path_factory):
    return load_character(make_programmer_art(rig, tmp_path_factory.mktemp("pa")))


@pytest.fixture(scope="module")
def puppet(rig, character):
    return character.puppet(rig)


@pytest.fixture(scope="module")
def walk(rig):
    take = retarget(parse_bvh(walk_bvh(frames=90, fps=30.0)), rig, "cmu_bvh", drop_tpose_frame=False)
    start, end, _ = find_loop(take, 0.6, 1.6)
    return cut_loop(take, start, end)


def static_clip(rig, lengths, angles: dict[str, float], swaps: dict[str, bool] | None = None) -> Clip:
    """One frame; every bone not listed hangs down (0), the body points up (180)."""
    rest = {b.name: 0.0 for b in rig.bones} | {"pelvis": 180.0, "torso": 180.0, "head": 180.0,
                                                 "foot_near": -90.0, "foot_far": -90.0}
    rest |= angles
    hip = lengths["thigh_near"] + lengths["shin_near"]
    return Clip(rig=rig.id, source="test", profile="test", fps=10, frame_count=1,
                bones=[b.name for b in rig.bones], bone_lengths=dict(lengths),
                angles={k: [v] for k, v in rest.items()}, local_angles={k: [0.0] for k in rest},
                root=RootTrack(x=[0.0], y=[-hip]),
                swaps={k: [v] for k, v in (swaps or {"thigh": False, "upper_arm": False}).items()})


def pixel_at(puppet, clip, bone: str, t: float = 0.5):
    size, origin = (2 * H, 2 * H), (H, 1.5 * H)
    img = puppet.render_frame(clip, 0, H, size, origin)
    pose = pose_bones(puppet.bones, clip, 0, puppet.lengths)
    (x0, y0), (x1, y1) = pose[bone]
    x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    return img.getpixel((round(origin[0] + x * H), round(origin[1] + y * H)))


def close(rgb, ref, tol=6):
    return all(abs(a - b) <= tol for a, b in zip(rgb[:3], ref))


def tinted(color, t=0.75):
    return tuple(round(c * t) for c in color)


# ------------------------------------------------------------------ programmer art


def test_programmer_art_sheet(rig, character):
    sheet = Image.open(character.sheet_path)
    assert sheet.mode == "RGBA"
    assert sheet.size == (rig.part_sheet.cols * 256, rig.part_sheet.rows * 256)
    alpha = np.asarray(sheet.getchannel("A"))
    assert alpha[0, 0] == 0 and alpha.max() == 255  # transparent background, opaque parts
    assert ((alpha > 0) & (alpha < 255)).any()  # soft (antialiased) edges, native alpha
    for p in rig.part_sheet.parts:  # every cell has art
        c, r = p.cell
        assert alpha[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256].max() == 255, p.name
    assert character.spec.bone_lengths["thigh_near"] == rig.bone("thigh_near").rest_length


def test_character_rejects_wrong_rig(rig, character):
    other = character.spec.model_copy(update={"rig": "quadruped_v1"})
    from ggs.rig.character import Character

    with pytest.raises(ValueError, match="rig"):
        Character(other, character.base_dir).puppet(rig)


# ------------------------------------------------------------------ renderer


def test_far_limbs_are_tinted_and_drawn_behind(rig, puppet):
    # both legs exactly overlapping: the near leg (untinted) must be on top
    clip = static_clip(rig, puppet.lengths, {})
    assert close(pixel_at(puppet, clip, "thigh_near"), PANTS)


def test_swap_puts_far_leg_in_front(rig, puppet):
    clip = static_clip(rig, puppet.lengths, {}, {"thigh": True, "upper_arm": False})
    assert close(pixel_at(puppet, clip, "thigh_far"), tinted(PANTS))


def test_arm_layering_over_torso(rig, puppet):
    # forearms hang over the torso: near forearm is in front of the torso, far one behind it
    clip = static_clip(rig, puppet.lengths, {})
    assert close(pixel_at(puppet, clip, "forearm_near", 0.3), SKIN)
    swapped = static_clip(rig, puppet.lengths, {}, {"thigh": False, "upper_arm": True})
    assert close(pixel_at(puppet, swapped, "forearm_far", 0.3), tinted(SKIN))


def test_crossing_legs_keep_near_leg_on_top(rig, puppet):
    # walk-style crossing: near thigh back, far thigh forward; they overlap just under the hip
    arms_up = {f"{b}_{s}": 180.0 for b in ("upper_arm", "forearm", "hand") for s in ("near", "far")}
    clip = static_clip(rig, puppet.lengths, {"thigh_near": 25.0, "thigh_far": -25.0,
                                             "shin_near": 25.0, "shin_far": -25.0, **arms_up})
    assert close(pixel_at(puppet, clip, "thigh_near", 0.08), PANTS)
    assert close(pixel_at(puppet, clip, "thigh_far", 0.9), tinted(PANTS))  # far leg visible when apart


def test_rotation_follows_angle_convention(rig, puppet):
    # thigh pointing screen right (-90): the knee pixel is to the right of the hip
    clip = static_clip(rig, puppet.lengths, {"thigh_near": -90.0, "thigh_far": -90.0})
    pose = pose_bones(puppet.bones, clip, 0, puppet.lengths)
    (hx, hy), (kx, ky) = pose["thigh_near"]
    assert kx > hx and ky == pytest.approx(hy, abs=1e-9)
    assert close(pixel_at(puppet, clip, "thigh_near"), PANTS)


def test_premultiplied_resampling_has_no_dark_fringe():
    white = Image.new("RGBA", (40, 40), (255, 255, 255, 255))
    canvas = Image.new("RGBA", (60, 60), (255, 255, 255, 0))
    canvas.paste(white, (10, 10))
    pup = Puppet([BoneDef("root", None)], {"root": 0.3}, ["root"], [],
                 {"root": PartImage(canvas, (30, 10), 0.0, 40.0)})
    clip = Clip(rig="x", source="t", profile="t", fps=1, frame_count=1, bones=["root"],
                bone_lengths={"root": 0.3}, angles={"root": [33.0]}, local_angles={"root": [33.0]},
                root=RootTrack(x=[0.0], y=[-0.15]))
    img = np.asarray(pup.render_frame(clip, 0, 100, (100, 100), (50, 50)))
    edge = (img[..., 3] > 20) & (img[..., 3] < 235)
    assert edge.sum() > 20
    assert img[edge][:, :3].min() >= 235  # edge colour stays white, never darkened by transparent black


def test_render_clip_shares_one_box(puppet, walk):
    frames, anchor = puppet.render_clip(walk, H)
    assert len(frames) == walk.frame_count
    assert len({f.size for f in frames}) == 1
    w, h = frames[0].size
    assert 0 < anchor[0] < w and anchor[1] == h - 2  # ground contact = lowest pixel (+2 padding below)
    assert h == pytest.approx(H, rel=0.25)  # roughly one standing height tall


# ------------------------------------------------------------------ loops


def test_find_loop_on_periodic_walk(rig):
    take = retarget(parse_bvh(walk_bvh(frames=90, fps=30.0, cycle_s=1.0)), rig, "cmu_bvh",
                    drop_tpose_frame=False)
    start, end, cost = find_loop(take, 0.6, 1.6)
    assert end - start == 30 and cost < 1.0
    loop = cut_loop(take, start, end)
    assert loop.loop and loop.frame_count == 30
    assert loop.loop_travel == pytest.approx(12.0 / 20.0, abs=0.02)  # speed 12 units/s, height 20
    for b in loop.bones:  # wrap-around step is as small as an ordinary step
        a = np.array(loop.angles[b])
        wrap = abs(((a[0] - a[-1]) + 180) % 360 - 180)
        assert wrap <= np.abs(np.diff(a)).max() + 0.5, b


# ------------------------------------------------------------------ exports


def test_frame_durations_keep_exact_timing():
    d = frame_durations_ms(24, 24.0)
    assert sum(d) == 1000 and set(d) <= {41, 42}


def test_animated_round_trip(tmp_path, puppet, walk):
    frames, _ = puppet.render_clip(walk, H)
    n, size = len(frames), frames[0].size
    for save, name in ((save_apng, "a.png"), (save_webp, "a.webp")):
        path = save(frames, tmp_path / name, walk.fps, True)
        im = Image.open(path)
        assert im.n_frames == n and im.size == size and im.info.get("loop") == 0
        total = 0
        for i in range(n):
            im.seek(i)
            im.load()  # WebP fills in per-frame timing on load
            total += im.info["duration"]
            if i == 3:
                got = np.asarray(im.convert("RGBA"), dtype=int)
                ref = np.asarray(frames[3], dtype=int)
                # lossless: alpha everywhere, colour wherever it is visible (RGB under alpha 0 is free)
                assert np.abs(got[..., 3] - ref[..., 3]).max() <= 1, name
                vis = ref[..., 3] > 0
                assert np.abs(got[vis][:, :3] - ref[vis][:, :3]).max() <= 1, name
                assert (got[..., 3] == 0).any() and (got[..., 3] == 255).any()
        assert total == pytest.approx(n * 1000 / walk.fps, abs=n)

    gif = Image.open(save_gif(frames, tmp_path / "a.gif", walk.fps, (136, 170, 102), True))
    assert gif.n_frames == n and gif.size == size
    gif.seek(3)
    rgba = np.asarray(gif.convert("RGBA"))
    assert (rgba[..., 3] == 0).any() and set(np.unique(rgba[..., 3])) <= {0, 255}  # 1-bit alpha


def test_sprite_sheet_slices_back(tmp_path, puppet, walk):
    frames, anchor = puppet.render_clip(walk, H)
    png, js = save_sprite_sheet(frames, anchor, tmp_path / "walk.png", "walk", walk.fps, True)
    sheet = Image.open(png)
    atlas = json.loads(js.read_text(encoding="utf-8"))
    sw, sh = sheet.size
    assert sw == next_pot(sw) and sh == next_pot(sh)
    assert atlas["meta"]["size"] == {"w": sw, "h": sh}
    names = atlas["animations"]["walk"]
    assert len(names) == len(frames)
    rects = []
    for i, key in enumerate(names):
        f = atlas["frames"][key]["frame"]
        rects.append((f["x"], f["y"], f["w"], f["h"]))
        crop = np.asarray(sheet.crop((f["x"], f["y"], f["x"] + f["w"], f["y"] + f["h"])))
        assert np.array_equal(crop, np.asarray(frames[i])), key
        piv = atlas["frames"][key]["pivot"]
        assert piv["x"] * f["w"] == pytest.approx(anchor[0], abs=0.01)
    for a in rects:  # no overlaps, padding kept
        for b in rects:
            if a is not b:
                assert a[0] + a[2] + 2 <= b[0] or b[0] + b[2] + 2 <= a[0] or \
                       a[1] + a[3] + 2 <= b[1] or b[1] + b[3] + 2 <= a[1]
    assert atlas["meta"]["ggs"]["animations"]["walk"] == {"fps": walk.fps, "loop": True,
                                                          "frames": len(frames)}


def test_packers():
    cols, rows, sw, sh = grid_layout(26, 150, 280, 2)
    assert cols * rows >= 26 and sw == next_pot(sw) and sh == next_pot(sh)
    (w, h), pos = shelf_pack([(30, 80), (50, 20), (10, 10), (100, 40)], 2)
    assert w == next_pot(w) and h == next_pot(h)
    boxes = [(x, y, x + s[0], y + s[1]) for (x, y), s in zip(pos, [(30, 80), (50, 20), (10, 10), (100, 40)])]
    for i, a in enumerate(boxes):
        assert a[2] <= w and a[3] <= h
        for b in boxes[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_rig_package_round_trip(tmp_path, rig, character, puppet, walk):
    out = write_rig_package(rig, character, {"walk": walk}, tmp_path / "pkg", H)
    doc = json.loads((out / "rig.json").read_text(encoding="utf-8"))
    assert doc["format"] == "ggs.rig_package" and doc["clips"] == ["walk"]
    assert [b["name"] for b in doc["bones"]] == [b.name for b in rig.bones]
    assert {v["name"] for v in doc["head_variants"]} == {"blink", "mouth_open", "hurt"}
    atlas = Image.open(out / "parts.png")
    assert atlas.size[0] == next_pot(atlas.size[0]) and atlas.size[1] == next_pot(atlas.size[1])
    clip_doc = json.loads((out / "clips" / "walk.json").read_text(encoding="utf-8"))
    assert clip_doc["loop"] and clip_doc["frame_count"] == walk.frame_count

    pkg_puppet, clips, _ = load_rig_package(out)
    a, anchor_a = puppet.render_clip(walk, H)
    b, anchor_b = pkg_puppet.render_clip(clips["walk"], H)
    assert len(a) == len(b)
    for fa, fb in zip(a[::5], b[::5]):
        # same canvas up to a pixel of crop difference
        w, h = min(fa.width, fb.width), min(fa.height, fb.height)
        dx = anchor_a[0] - anchor_b[0]
        A = np.asarray(fa, dtype=float)[:h, max(0, dx):max(0, dx) + w - abs(dx)]
        B = np.asarray(fb, dtype=float)[:h, max(0, -dx):max(0, -dx) + w - abs(dx)]
        A, B = A[: min(len(A), len(B))], B[: min(len(A), len(B))]
        assert np.abs(A[..., 3] - B[..., 3]).mean() < 4.0  # alpha matches within resampling noise
        assert math.isclose(fa.getchannel("A").getbbox()[3] - anchor_a[1],
                            fb.getchannel("A").getbbox()[3] - anchor_b[1], abs_tol=2)


def test_cli_end_to_end(tmp_path, capsys):
    bvh = tmp_path / "walk.bvh"
    bvh.write_text(walk_bvh(frames=90, fps=30.0), encoding="utf-8")
    take, loop = tmp_path / "take.json", tmp_path / "walk.json"
    assert main(["mocap", "retarget", str(bvh), "--profile", "cmu_bvh", "--out", str(take)]) == 0
    assert main(["clip", "loop", str(take), "--out", str(loop)]) == 0
    assert main(["character", "programmer-art", "--out", str(tmp_path / "pa")]) == 0
    char = str(tmp_path / "pa" / "character.yaml")
    base = ["export", str(loop), "--character", char, "--height", "96"]
    assert main([*base, "--format", "apng", "--out", str(tmp_path / "w.png")]) == 0
    assert main([*base, "--format", "webp", "--out", str(tmp_path / "w.webp")]) == 0
    assert main([*base, "--format", "sheet", "--out", str(tmp_path / "s.png")]) == 0
    assert main([*base, "--format", "rig", "--out", str(tmp_path / "pkg")]) == 0
    capsys.readouterr()
    assert main([*base, "--format", "gif", "--out", str(tmp_path / "w.gif")]) == 1
    assert "matte" in capsys.readouterr().err
    assert main([*base, "--format", "gif", "--matte", "#202020", "--out", str(tmp_path / "w.gif")]) == 0
    assert (tmp_path / "s.json").exists() and (tmp_path / "pkg" / "clips" / "walk.json").exists()
    assert Image.open(tmp_path / "w.png").n_frames == 30
