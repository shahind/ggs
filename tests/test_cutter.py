"""M5: pose guide -> cut -> part sheet -> reassembly validation, on CPU with the programmer-art puppet."""

import json

import numpy as np
import pytest

pytest.importorskip("PIL")

from ggs.cli import main
from ggs.paths import repo_root
from ggs.rig.cutter import component_sizes, convex_hull_mask, fill_from, nearest_labels
from ggs.rig.make import cut_master, guide_only_master, guide_puppet, make_guide
from ggs.rig.master import (
    MASTER_POSE,
    bone_angle,
    load_master,
    pose_clip,
    render_guide,
    save_master,
)
from ggs.rig.pose import bone_vector
from ggs.rig.spec import load_rig_spec

RIG = repo_root() / "rigs" / "humanoid_side_v1.yaml"
SIZE = 512


@pytest.fixture(scope="module")
def rig():
    return load_rig_spec(RIG)


def test_helpers():
    m = np.zeros((10, 12), bool)
    m[1:3, 1:3] = True
    m[6:9, 5:11] = True
    m[3, 3] = True  # touches the first block diagonally
    assert component_sizes(m) == [18, 5]
    tri = np.zeros((9, 9), bool)
    tri[0, 0] = tri[0, 8] = tri[8, 4] = True
    assert convex_hull_mask(tri)[4, 4] and not convex_hull_mask(tri)[8, 0]
    lab = np.full((5, 5), -1, np.int16)
    lab[0, 0], lab[4, 4] = 1, 2
    out = nearest_labels(lab, np.ones((5, 5), bool))
    assert out[1, 0] == 1 and out[3, 4] == 2 and (out >= 0).all()
    rgb = np.zeros((6, 6, 3), np.uint8)
    rgb[:, :3] = 200
    known = np.zeros((6, 6), bool)
    known[:, :3] = True
    fill = np.zeros((6, 6), bool)
    fill[:, 4:] = True  # not touching the known pixels
    assert np.allclose(fill_from(rgb, known, fill)[:, 4:], 200)


def test_bone_angle_inverts_bone_vector():
    for a in (0.0, 30.0, -90.0, 150.0):
        dx, dy = bone_vector(a, 1.0)
        assert bone_angle((0.0, 0.0), (dx, dy)) == pytest.approx(a)


def test_guide_round_trip(rig, tmp_path):
    g = make_guide(rig, SIZE)
    path = save_master(tmp_path, g.spec, g.image, g)
    spec, master, labels = load_master(path)
    assert master.shape == (SIZE, SIZE, 4) and labels.max() == len(rig.bones) - 1
    assert spec.angles == MASTER_POSE
    s, e = spec.joints["thigh_near"]
    assert bone_angle(s, e) == pytest.approx(MASTER_POSE["thigh_near"], abs=1e-6)


def test_clean_cut_passes(rig, tmp_path):
    out = cut_master(guide_only_master(rig, tmp_path / "m", SIZE), rig, tmp_path / "c", "pa")
    rep = out.report
    assert rep.passed, rep.failures()
    assert rep.score > 0.97
    from PIL import Image

    sheet = Image.open(tmp_path / "c" / "parts.png")
    ps = rig.part_sheet
    assert sheet.size == (ps.cols * ps.cell_px, ps.rows * ps.cell_px)
    info = json.loads((tmp_path / "c" / "validation.json").read_text(encoding="utf-8"))
    assert info["passed"] and {c["name"] for c in info["checks"]} >= {"reassembly_iou", "pose:bend"}
    assert (tmp_path / "c" / "validation.png").exists()


def test_master_ignoring_the_pose_is_rejected(rig, tmp_path):
    """The generator drew the character in another pose: labels from the guide no longer fit."""
    g = make_guide(rig, SIZE)
    puppet, lengths = guide_puppet(rig)
    off = render_guide(rig, puppet, lengths, size=SIZE, angles={
        **MASTER_POSE, "upper_arm_near": -150.0, "forearm_near": -150.0, "hand_near": -150.0,
        "thigh_near": -60.0, "shin_near": -20.0})
    path = save_master(tmp_path / "m", g.spec, off.image, g)  # wrong image, original guide
    out = cut_master(path, rig, tmp_path / "c", "bad")
    assert not out.report.passed
    assert any(f.startswith(("guide_agreement:forearm", "guide_agreement:upper_arm", "cut:"))
               for f in out.report.failures()), out.report.failures()
    assert (tmp_path / "c" / "validation.json").exists()

    # a smaller pose change still leaves pixels to cut; the guide agreement check must catch it
    mild = render_guide(rig, puppet, lengths, size=SIZE, angles={
        **MASTER_POSE, "upper_arm_near": -70.0, "forearm_near": -70.0, "hand_near": -70.0})
    path = save_master(tmp_path / "m2", g.spec, mild.image, g)
    rep = cut_master(path, rig, tmp_path / "c2", "bad").report
    assert not rep.passed
    assert any(f.startswith(("guide_agreement:", "joint:", "connected:")) for f in rep.failures()), rep.failures()


def test_wrong_joints_are_rejected(rig, tmp_path):
    g = make_guide(rig, SIZE)
    joints = dict(g.spec.joints)
    (s, e) = joints["forearm_near"]
    shift = 0.08 * g.spec.height_px
    dx, dy = bone_vector(bone_angle(s, e), shift)
    joints["forearm_near"] = ((s[0] + dx, s[1] + dy), e)  # elbow pushed down the forearm
    us, _ = joints["upper_arm_near"]
    joints["upper_arm_near"] = (us, joints["forearm_near"][0])
    path = save_master(tmp_path / "m", g.spec.model_copy(update={"joints": joints}), g.image, g)
    rep = cut_master(path, rig, tmp_path / "c", "bad").report
    assert not rep.passed
    assert any(f.startswith("joint:forearm") for f in rep.failures())


def test_cut_character_renders_any_pose(rig, tmp_path):
    out = cut_master(guide_only_master(rig, tmp_path / "m", SIZE), rig, tmp_path / "c", "pa")
    from ggs.rig.character import load_character

    ch = load_character(out.character)
    angles = {**MASTER_POSE, "forearm_near": -120.0}
    im = ch.puppet(rig).render_frame(pose_clip(rig, angles, ch.spec.bone_lengths), 0, 300, (600, 600), (300, 250))
    assert np.asarray(im)[..., 3].any()


def test_cli_character_master_and_cut(tmp_path, capsys):
    assert main(["character", "master", "--guide-only", "--out", str(tmp_path / "g")]) == 0
    assert "OK" in capsys.readouterr().out
    # the stub matte is crude (enclosed gaps count as opaque), so only the flow and its files are checked
    assert main(["character", "master", "--model", "stub", "--prompt", "a knight", "--out", str(tmp_path / "s"),
                 "--count", "2", "--no-vram-check"]) in (0, 2)
    assert (tmp_path / "s" / "seed_001" / "character" / "validation.json").exists()
    assert (tmp_path / "s" / "seed_001" / "master_raw.png").exists()
    assert main(["character", "master", "--out", str(tmp_path / "x")]) == 1  # no prompt
    master = tmp_path / "g" / "guide_only" / "master.json"
    assert main(["character", "cut", str(master), "--out", str(tmp_path / "again")]) == 0
