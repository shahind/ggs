"""Joint fitting (detections onto the guide), pixel ownership, and warp animation (CPU, no models)."""

import json
import math

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("cv2")

import itertools

from ggs import items as I
from ggs.cli import main
from ggs.paths import rigs_dir
from ggs.rig import load_rig_spec
from ggs.rig.joints import bone_labels, bone_widths, cloth_share, fit_joints
from ggs.style.package import init_project


@pytest.fixture(scope="module")
def rig():
    return load_rig_spec(rigs_dir() / "humanoid_side_v1.yaml")


@pytest.fixture(scope="module")
def guide(rig):
    from ggs.rig.body import PRESETS
    from ggs.rig.make import guide_puppet
    from ggs.rig.master import render_guide

    puppet, lengths = guide_puppet(rig, PRESETS["standard"])
    return render_guide(rig, puppet, lengths, size=512)


def _kp(joints, moves=None, scores=None):
    """Keypoints where the guide has its joints (r_ = near), optionally moved."""
    where = {"sho": joints["upper_arm_near"][0], "elb": joints["forearm_near"][0], "wri": joints["hand_near"][0],
             "hip": joints["thigh_near"][0], "knee": joints["shin_near"][0], "ank": joints["foot_near"][0]}
    far = {"sho": joints["upper_arm_far"][0], "elb": joints["forearm_far"][0], "wri": joints["hand_far"][0],
           "hip": joints["thigh_far"][0], "knee": joints["shin_far"][0], "ank": joints["foot_far"][0]}
    kp = {}
    for k, pt in where.items():
        kp[f"r_{k}"] = (*pt, 0.9)
        kp[f"l_{k}"] = (*far[k], 0.9)
    for name, (dx, dy) in (moves or {}).items():
        x, y, s = kp[name]
        kp[name] = (x + dx, y + dy, s)
    for name, s in (scores or {}).items():
        kp[name] = (*kp[name][:2], s)
    return kp


def test_fit_moves_joints_onto_confident_close_detections(guide):
    g, h = guide.spec.joints, guide.spec.height_px
    # left/right swapped by the detector: matched back to near/far by position
    kp = _kp(g, moves={"r_elb": (10, 6)})
    kp["l_elb"], kp["r_elb"] = kp["r_elb"], kp["l_elb"]
    joints, report = fit_joints(g, kp, h)
    assert math.dist(joints["forearm_near"][0], (g["forearm_near"][0][0] + 10, g["forearm_near"][0][1] + 6)) < 1e-6
    assert joints["upper_arm_near"][1] == joints["forearm_near"][0]  # the chain stays connected
    assert report["elbow_near"] > 0
    # far off (a mistake) or unsure: the guide's joint is kept
    joints, report = fit_joints(g, _kp(g, moves={"r_knee": (0.4 * h, 0)}, scores={"r_wri": 0.1}), h)
    assert report["knee_near"] == "guide" and report["wrist_near"] == "guide"
    assert joints["shin_near"][0] == tuple(g["shin_near"][0])


def test_labels_follow_the_skeleton(rig, guide):
    rgba = np.asarray(guide.image)
    widths = bone_widths(guide.labels, rig, guide.spec.joints)
    assert widths["upper_arm_far"] == pytest.approx(widths["upper_arm_near"])  # far limbs: as wide as near ones
    lab = bone_labels(rgba[..., 3], rig, guide.spec.joints, widths, rgb=rgba[..., :3])
    on = rgba[..., 3] >= 128
    assert (lab[on] >= 0).all() and (lab[~on] == -1).all()
    agree = (lab[on] == guide.labels[on]).mean()
    assert agree > 0.8, agree  # on the guide itself the labels mostly agree with its own parts
    assert cloth_share(lab, rig, guide.spec.joints, widths) < 0.15  # trousers, no skirt


@pytest.mark.parametrize("mode", ["connected", "layers"])
def test_warp_rest_pose_reproduces_the_drawing_and_clips_play(rig, guide, mode):
    from ggs.rig.master import bone_angle, pose_clip
    from ggs.rig.skin import build_skin

    widths = bone_widths(guide.labels, rig, guide.spec.joints)
    skin = build_skin(rig, guide.image, guide.spec.joints, widths, guide.spec.height_px, mode=mode)
    names = {L.name for L in skin.layers}
    assert not skin.skirt
    assert names == {"body"} if mode == "connected" else names >= {"body", "upper_arm_near", "thigh_far"}
    angles = {b: bone_angle(*guide.spec.joints[b]) for b in guide.spec.joints}
    frames, _ = skin.render_clip(pose_clip(rig, angles, skin.lengths()), round(guide.spec.height_px))
    got = np.asarray(frames[0])[..., 3] >= 128
    want = np.asarray(guide.image)[..., 3] >= 128
    ys, xs = np.nonzero(want)
    want = want[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ys, xs = np.nonzero(got)
    got = got[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = min(got.shape[0], want.shape[0]), min(got.shape[1], want.shape[1])
    iou = (got[:h, :w] & want[:h, :w]).sum() / (got[:h, :w] | want[:h, :w]).sum()
    assert iou > 0.95, iou
    bent = {**angles, "forearm_near": angles["forearm_near"] + 70, "shin_far": angles["shin_far"] - 50}
    frames, _anchor = skin.render_clip(pose_clip(rig, bent, skin.lengths()), 128)
    assert frames[0].getchannel("A").getbbox() is not None and frames[0].height <= 140


def test_legs_go_behind_a_skirt(rig):
    from ggs.rig.skin import legs_behind

    order = legs_behind(rig.draw_order)
    assert sorted(order) == sorted(rig.draw_order)
    assert max(order.index(b) for b in order if b.startswith(("thigh", "shin", "foot"))) < order.index("pelvis")


def test_warp_from_the_design_exports(tmp_path):
    from ggs.rig.library import library_clip
    from ggs.rig.warp import make_design_warp, warp_skin

    p = tmp_path / "g"
    init_project(p, "G")
    I.create(p, "character", "Hero", "a hero", I.ItemSettings(character=I.CharacterSettings(view="side", pose="idle")))
    I.generate(p, "Hero", action="new", model="stub", enforce_vram=False)
    I.accept(p, "Hero")
    make_design_warp(p, "Hero", "coarse", fit=False)  # straight from the design, no sliced version
    d = p / "items" / "character" / "hero"
    skin = warp_skin(d)
    assert skin is not None and json.loads(skin.read_text(encoding="utf-8"))["format"] == "ggs.skin"
    walk = library_clip("walk")
    if not walk.exists():
        pytest.skip("motion library not built (tools/make_motion_library.py)")
    out = tmp_path / "walk.png"
    assert main(["export", str(walk), "--character", str(skin), "--format", "sheet", "--height", "96",
                 "--out", str(out)]) == 0
    assert out.exists() and out.with_suffix(".json").exists()
    assert main(["export", str(walk), "--character", str(skin), "--format", "godot", "--out",
                 str(tmp_path / "g2")]) == 1  # warp animations are pictures


def test_joint_cuts_split_a_limb_straight(rig, guide):
    from ggs.rig.joints import split_along_chain

    j = guide.spec.joints
    chain = ["upper_arm_near", "forearm_near", "hand_near"]
    pts = [j["upper_arm_near"][0], j["forearm_near"][0], j["hand_near"][0], j["hand_near"][1]]
    mids = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in itertools.pairwise(pts)]
    xs = np.array([m[0] for m in mids])
    ys = np.array([m[1] for m in mids])
    assert list(split_along_chain(xs, ys, chain, j)) == [0, 1, 2]  # each bone's middle is on its own side


def test_clean_cut_from_fitted_joints(rig, guide, tmp_path):
    """A master whose joints were fitted is cut along straight joint lines with real overlapping art: no discs,
    no blur, and it rebuilds the drawing."""
    from ggs.rig.joints import fit_master
    from ggs.rig.make import cut_master
    from ggs.rig.master import save_master

    m = save_master(tmp_path / "m", guide.spec, guide.image, guide)
    fit_master(m, _kp(guide.spec.joints), rig)
    spec = json.loads(m.read_text(encoding="utf-8"))
    assert spec["fit"]["skirt"] is False
    res = cut_master(m, rig, tmp_path / "c", "Puppet")
    iou = next(c for c in res.report.checks if c.name == "reassembly_iou").value
    assert res.character is not None and iou > 0.9, res.report.checks


def test_touch_up_turns_a_bone_and_its_children():
    from ggs.rig.clip import load_clip
    from ggs.rig.edits import turn_bone
    from ggs.rig.library import library_clip

    path = library_clip("walk")
    if not path.exists():
        pytest.skip("motion library not built")
    clip = load_clip(path)
    rig = load_rig_spec(rigs_dir() / f"{clip.rig}.yaml")
    parents = {b.name: b.parent for b in rig.bones}
    new = turn_bone(clip, parents, "upper_arm_near", 5, 30.0, spread=2)
    assert new.angles["upper_arm_near"][5] == pytest.approx(clip.angles["upper_arm_near"][5] + 30)
    assert new.angles["hand_near"][5] == pytest.approx(clip.angles["hand_near"][5] + 30)  # carried along
    assert new.angles["torso"][5] == clip.angles["torso"][5]
    assert 0 < new.angles["upper_arm_near"][6] - clip.angles["upper_arm_near"][6] < 30  # fades out
    assert new.angles["upper_arm_near"][9] == clip.angles["upper_arm_near"][9]


def test_moving_a_joint_rebuilds_the_rig(tmp_path):
    from ggs.rig.warp import make_design_warp, move_joints

    p = tmp_path / "g"
    init_project(p, "G")
    I.create(p, "character", "Hero", "a hero", I.ItemSettings(character=I.CharacterSettings(view="side", pose="idle")))
    I.generate(p, "Hero", action="new", model="stub", enforce_vram=False)
    I.accept(p, "Hero")
    skin_json = make_design_warp(p, "Hero", "coarse", fit=False)
    before = json.loads(skin_json.read_text(encoding="utf-8"))["joints"]
    elbow = before["forearm_near"][0]
    rebuilt = move_joints(skin_json, {"forearm_near": (elbow[0] + 5, elbow[1] + 3)})
    after = json.loads(skin_json.read_text(encoding="utf-8"))["joints"]
    assert after["forearm_near"][0] == pytest.approx([elbow[0] + 5, elbow[1] + 3])
    assert after["upper_arm_near"][1] == pytest.approx(after["forearm_near"][0])  # the upper arm ends there too
    assert rebuilt.layers


def test_limbs_bend_only_at_joints_and_stray_bits_join_their_surroundings(rig, guide):
    from ggs.rig.skin import build_skin, tidy_labels

    widths = bone_widths(guide.labels, rig, guide.spec.joints)
    skin = build_skin(rig, guide.image, guide.spec.joints, widths, guide.spec.height_px, mode="layers")
    names = [b.name for b in rig.bones]
    arm = next(L for L in skin.layers if L.name == "upper_arm_near")
    # a vertex in the middle of the forearm follows the forearm alone; at the elbow it is shared half and half
    (sx, sy), (ex, ey) = guide.spec.joints["forearm_near"]
    mid = np.argmin(np.hypot(arm.verts[:, 0] - (sx + ex) / 2, arm.verts[:, 1] - (sy + ey) / 2))
    assert arm.weights[mid, names.index("forearm_near")] > 0.99
    elbow = np.argmin(np.hypot(arm.verts[:, 0] - sx, arm.verts[:, 1] - sy))
    assert 0.2 < arm.weights[elbow, names.index("upper_arm_near")] < 0.8
    assert np.allclose(arm.weights.sum(1), 1.0, atol=1e-4)
    # a few pixels of the face labelled as the near thigh go to the head
    lab = skin.labels.copy()
    hx, hy = (np.array(guide.spec.joints["head"][0]) + 2 * np.array(guide.spec.joints["head"][1])) / 3
    lab[int(hy) - 1:int(hy) + 2, int(hx) - 1:int(hx) + 2] = names.index("thigh_near")
    out = tidy_labels(lab, rig, guide.spec.joints, widths)
    assert out[int(hy), int(hx)] == names.index("head")


def test_connected_warp_keeps_the_drawing_in_one_piece(rig, guide):
    from ggs.rig.master import bone_angle, pose_clip
    from ggs.rig.skin import build_skin

    widths = bone_widths(guide.labels, rig, guide.spec.joints)
    skin = build_skin(rig, guide.image, guide.spec.joints, widths, guide.spec.height_px)
    assert skin.mode == "connected" and len(skin.layers) == 1
    mesh = skin.layers[0]
    names = [b.name for b in rig.bones]
    # the forearm's middle follows the forearm alone (measured through the shape, not across the gap to the body)
    (sx, sy), (ex, ey) = guide.spec.joints["forearm_near"]
    mid = np.argmin(np.hypot(mesh.verts[:, 0] - (sx + ex) / 2, mesh.verts[:, 1] - (sy + ey) / 2))
    assert mesh.weights[mid, names.index("forearm_near")] > 0.99
    assert np.allclose(mesh.weights.sum(1), 1.0, atol=1e-4)
    # bent hard, the posed drawing stays one connected piece
    angles = {b: bone_angle(*guide.spec.joints[b]) for b in guide.spec.joints}
    bent = {**angles, "upper_arm_near": angles["upper_arm_near"] + 80, "thigh_near": angles["thigh_near"] - 40}
    frames, _ = skin.render_clip(pose_clip(rig, bent, skin.lengths()), 256)
    from scipy import ndimage

    solid = np.asarray(frames[0])[..., 3] >= 128
    comp, n = ndimage.label(solid, structure=np.ones((3, 3), bool))
    sizes = np.bincount(comp.ravel())[1:]
    assert sizes.max() > 0.98 * solid.sum(), (n, sorted(sizes)[-3:])


def test_warp_and_sliced_version_know_when_their_picture_changed(tmp_path):
    from ggs.items import ItemRecord, design_dir
    from ggs.rig.slices import rigdraw_dir, sliced_dir, sliced_stale
    from ggs.rig.warp import picture_sig, warp_dir, warp_source, warp_stale

    d = tmp_path / "items" / "character" / "hero"
    design_dir(d).mkdir(parents=True)
    (design_dir(d) / "design.png").write_bytes(b"first design")
    warp_dir(d).mkdir()
    (warp_dir(d) / "skin.json").write_text("{}", encoding="utf-8")
    (warp_dir(d) / "warp.json").write_text(json.dumps({"source_sig": picture_sig(design_dir(d) / "design.png")}),
                                           encoding="utf-8")
    assert warp_source(d) == design_dir(d) / "design.png" and not warp_stale(d)
    (design_dir(d) / "design.png").write_bytes(b"refined design")  # a new design was accepted
    assert warp_stale(d)
    rigdraw_dir(d).mkdir(parents=True)  # a new sliced version drew the animation-ready drawing: bend that
    (rigdraw_dir(d) / "master.json").write_text("{}", encoding="utf-8")
    (rigdraw_dir(d) / "master.png").write_bytes(b"drawing")
    assert warp_source(d) == rigdraw_dir(d) / "master.png" and warp_stale(d)

    rec = ItemRecord(name="hero", asset_class="character", accepted=2)
    (d / "item.json").write_text(rec.model_dump_json(), encoding="utf-8")
    sliced_dir(d).mkdir(parents=True, exist_ok=True)
    (sliced_dir(d) / "slices.json").write_text(json.dumps({"design": 2}), encoding="utf-8")
    assert not sliced_stale(d)
    (d / "item.json").write_text(rec.model_copy(update={"accepted": 3}).model_dump_json(), encoding="utf-8")
    assert sliced_stale(d)
