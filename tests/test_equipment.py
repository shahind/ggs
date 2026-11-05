"""M7: held items, two-bone IK, keyframed clips and animation layers (CPU)."""

import math

import numpy as np
import pytest

pytest.importorskip("PIL")

from ggs.cli import main
from ggs.paths import repo_root
from ggs.rig.character import make_programmer_art
from ggs.rig.equipment import (
    apply_two_hand,
    item_angle,
    item_point,
    load_item,
    make_item,
    socket_point,
    two_bone_ik,
)
from ggs.rig.layers import keys_to_clip, layer_clips, load_keys
from ggs.rig.master import MASTER_POSE, pose_clip
from ggs.rig.pose import bone_defs, bone_vector, pose_bones
from ggs.rig.spec import load_rig_spec

CLIPS = repo_root() / "clips"


@pytest.fixture(scope="module")
def rig():
    return load_rig_spec(repo_root() / "rigs" / "humanoid_side_v1.yaml")


@pytest.fixture()
def bar(tmp_path):
    from PIL import Image, ImageDraw

    im = Image.new("RGBA", (40, 300), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([16, 10, 23, 289], fill=(120, 120, 130, 255))
    im.save(tmp_path / "bar.png")
    return tmp_path / "bar.png"


def test_make_item_finds_grips(tmp_path, bar):
    item = load_item(make_item(bar, tmp_path / "spear", name="spear", length=1.0, hold="two_hand"))
    s = item.spec
    assert s.hold == "two_hand" and s.grip2 is not None
    assert s.grip2[1] < s.grip[1]  # the second grip is further up the shaft
    assert s.grip[0] == pytest.approx(3.5, abs=1)  # centre line of the cropped bar
    assert s.px_per_height == pytest.approx(280, abs=2)
    with pytest.raises(ValueError):
        load_item(tmp_path / "missing")


def test_two_bone_ik_reaches():
    for target in ((0.1, 0.2), (0.25, 0.05), (-0.1, 0.25)):
        up, lo = two_bone_ik((0.0, 0.0), target, 0.15, 0.14)
        ex, ey = bone_vector(up, 0.15)
        wx, wy = bone_vector(lo, 0.14)
        assert math.hypot(ex + wx - target[0], ey + wy - target[1]) < 1e-6
    up, lo = two_bone_ik((0.0, 0.0), (1.0, 0.0), 0.15, 0.14)  # out of reach: points straight at it
    assert up == pytest.approx(-90, abs=0.5) and lo == pytest.approx(-90, abs=0.5)


def test_keys_ease_and_validate(rig, tmp_path):
    k = load_keys(CLIPS / "swing_one_hand.yaml")
    clip = keys_to_clip(k, rig)
    a = clip.angles["upper_arm_near"]
    assert clip.frame_count == 16 and a[0] == -20 and a[4] == -165 and a[15] == -20
    assert -165 < a[2] < -20  # eased in between
    assert clip.angles["thigh_near"] == [0.0] * 16  # not keyed: rest pose
    bad = tmp_path / "bad.yaml"
    bad.write_text("format: ggs.keys\nversion: 1\nname: x\nframes: 4\nkeys:\n  - {frame: 1, angles: {tail: 3}}\n",
                   encoding="utf-8")
    with pytest.raises(ValueError):
        keys_to_clip(load_keys(bad), rig)


def test_layer_keeps_arms_on_the_moving_torso(rig):
    lengths = {b.name: b.rest_length for b in rig.bones}
    base = pose_clip(rig, {**MASTER_POSE, "torso": 200.0}, lengths)  # leaning forward
    over = keys_to_clip(load_keys(CLIPS / "hold_one_hand.yaml"), rig)
    out = layer_clips(base, over, rig, "arms")
    assert set(out.angles["thigh_near"]) == set(base.angles["thigh_near"])  # legs from the base
    # arms keep their angle *relative to the torso* from the overlay (torso 180 there, 200 here)
    assert out.angles["upper_arm_near"][0] == pytest.approx(-20 + 20)
    assert out.angles["forearm_near"][0] == pytest.approx(-80 + 20)
    swing = keys_to_clip(load_keys(CLIPS / "swing_one_hand.yaml"), rig)
    assert layer_clips(base, swing, rig, "near_arm").frame_count == 16  # a one-off: the overlay's length


def test_two_hand_ik_puts_the_far_hand_on_grip2(rig, tmp_path, bar):
    item = load_item(make_item(bar, tmp_path / "spear", name="spear", length=1.0, hold="two_hand")).spec
    lengths = {b.name: b.rest_length for b in rig.bones}
    hold = layer_clips(pose_clip(rig, MASTER_POSE, lengths),
                       keys_to_clip(load_keys(CLIPS / "hold_two_hand.yaml"), rig), rig, "arms")
    clip = apply_two_hand(hold, rig, lengths, item)
    pose = pose_bones(bone_defs(rig), clip, 0, lengths)
    grip, _ = socket_point(rig, "grip_near", pose)
    g2 = item_point(item, item.grip2, grip, item_angle(item, clip.angles["hand_near"][0]))
    far_grip, _ = socket_point(rig, "grip_far", pose)
    assert math.hypot(far_grip[0] - g2[0], far_grip[1] - g2[1]) < 1e-3


def test_cli_item_keys_layer_export(tmp_path, bar, rig):
    from PIL import Image

    assert main(["item", "make", str(bar), "--out", str(tmp_path / "sword"), "--name", "sword",
                 "--length", "0.45"]) == 0
    assert main(["clip", "keys", str(CLIPS / "swing_one_hand.yaml"), "--out", str(tmp_path / "swing.json")]) == 0
    lengths = {b.name: b.rest_length for b in rig.bones}
    walk = pose_clip(rig, MASTER_POSE, lengths)
    walk.root.y = [-(lengths["thigh_near"] + lengths["shin_near"])]
    walk = walk.model_copy(update={"loop": True})
    walk.save(tmp_path / "walk.json")
    assert main(["clip", "layer", str(tmp_path / "walk.json"), "--over", str(tmp_path / "swing.json"),
                 "--bones", "near_arm", "--out", str(tmp_path / "ws.json")]) == 0
    char = make_programmer_art(rig, tmp_path / "pa")
    for item_arg, name in (([], "bare"), (["--item", str(tmp_path / "sword")], "armed")):
        assert main(["export", str(tmp_path / "ws.json"), "--character", str(char), "--format", "apng",
                     "--out", str(tmp_path / f"{name}.png"), "--height", "120", *item_arg]) == 0
    bare = np.asarray(Image.open(tmp_path / "bare.png").convert("RGBA"))
    armed = np.asarray(Image.open(tmp_path / "armed.png").convert("RGBA"))
    assert (armed[..., 3] > 0).sum() > (bare[..., 3] > 0).sum()  # the sword adds pixels
    assert main(["export", str(tmp_path / "ws.json"), "--character", str(char), "--format", "godot",
                 "--out", str(tmp_path / "g"), "--item", str(tmp_path / "sword")]) == 0
    assert "item_sword" in next((tmp_path / "g").glob("*.tscn")).read_text(encoding="utf-8")
