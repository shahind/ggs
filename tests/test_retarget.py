import itertools
import json
import math

import numpy as np
import pytest
from bvh_synth import HIP_HEIGHT, make_bvh, walk_bvh

from ggs.cli import main
from ggs.paths import rigs_dir
from ggs.rig import draw_order_for, limb_groups, load_clip, load_rig_spec, parse_bvh, retarget
from ggs.rig.pose import pose_frame
from ggs.rig.retarget import RetargetError

RIG = rigs_dir() / "humanoid_side_v1.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_rig_spec(RIG)


def static_bvh(pose: dict, frames: int = 4, fps: float = 30.0) -> str:
    return make_bvh(lambda i, t: pose, frames, fps)


def clip_for(text: str, spec, **kw):
    kw.setdefault("drop_tpose_frame", False)
    return retarget(parse_bvh(text, source="synthetic.bvh"), spec, "cmu_bvh", **kw)


def tpose_then_walk(frames: int = 30) -> str:
    walk = parse_bvh(walk_bvh(frames=frames, fps=30.0))
    lines = walk_bvh(frames=frames, fps=30.0).splitlines()
    # arms raised sideways: LeftArm Zrotation +90 (to +X), RightArm -90 (to -X, the near side)
    vals = [0.0, HIP_HEIGHT, 0.0] + [0.0] * (walk.frames.shape[1] - 3)
    vals[walk.joints[walk.index("LeftArm")].channel_start] = 90.0
    vals[walk.joints[walk.index("RightArm")].channel_start] = -90.0
    rest = " ".join(f"{v:g}" for v in vals)
    i = lines.index(f"Frames: {frames}")
    lines[i] = f"Frames: {frames + 1}"
    lines.insert(i + 2, rest)
    return "\n".join(lines) + "\n"


def test_tpose_frame_dropped_by_default(spec):
    text = tpose_then_walk()
    clip = retarget(parse_bvh(text), spec, "cmu_bvh")
    assert clip.frame_count == 30
    assert any("T-pose" in w for w in clip.warnings)
    walk = clip_for(walk_bvh(frames=30, fps=30.0), spec)
    assert clip.angles["thigh_near"] == walk.angles["thigh_near"]

    kept = retarget(parse_bvh(text), spec, "cmu_bvh", drop_tpose_frame=False)
    assert kept.frame_count == 31 and not any("T-pose" in w for w in kept.warnings)
    walk_only = retarget(parse_bvh(walk_bvh(frames=30, fps=30.0)), spec, "cmu_bvh")
    assert walk_only.frame_count == 30  # arms down: first frame kept


def test_limb_groups(spec):
    groups = {g.name: g for g in limb_groups(spec)}
    assert set(groups) == {"upper_arm", "thigh"}
    assert groups["thigh"].near == ("thigh_near", "shin_near", "foot_near")
    assert groups["upper_arm"].far == ("upper_arm_far", "forearm_far", "hand_far")


def test_rest_pose_angles(spec):
    clip = clip_for(static_bvh({}), spec)
    a = {k: v[0] for k, v in clip.angles.items()}
    for bone in ("thigh_near", "shin_near", "thigh_far", "upper_arm_near", "forearm_far"):
        assert a[bone] == pytest.approx(0.0, abs=1e-6), bone  # hanging straight down
    for bone in ("pelvis", "torso", "head"):
        assert abs(a[bone]) == pytest.approx(180.0, abs=1e-6), bone  # pointing up
    # feet point forward = screen right = -90 deg (clockwise from down)
    assert a["foot_near"] == pytest.approx(-90 + math.degrees(math.atan2(0.5, 1.5)), abs=1e-3)
    assert clip.warnings == []


def test_known_thigh_swing(spec):
    # Xrotation -30 swings the right thigh forward: screen vector (sin30, cos30) -> angle -30
    clip = clip_for(static_bvh({"RightUpLeg": (0, 0, -30.0), "LeftUpLeg": (0, 0, 20.0)}), spec)
    assert clip.angles["thigh_near"][0] == pytest.approx(-30.0, abs=1e-6)
    assert clip.angles["thigh_far"][0] == pytest.approx(20.0, abs=1e-6)
    assert clip.local_angles["thigh_near"][0] == pytest.approx(-30.0 - 180.0 + 360.0, abs=1e-6)


def test_knee_bend_is_local(spec):
    clip = clip_for(static_bvh({"RightUpLeg": (0, 0, -40.0), "RightLeg": (0, 0, 60.0)}), spec)
    assert clip.angles["shin_near"][0] == pytest.approx(20.0, abs=1e-6)
    assert clip.local_angles["shin_near"][0] == pytest.approx(60.0, abs=1e-6)


def test_scale_and_root(spec):
    clip = clip_for(static_bvh({"_root": (0.0, HIP_HEIGHT, 0.0)}), spec)
    assert clip.root.y[0] == pytest.approx(-HIP_HEIGHT / 20.0, abs=1e-4)  # hips above ground, +y down
    assert clip.root.x[0] == pytest.approx(0.0)
    assert clip.bone_lengths["shin_near"] == pytest.approx(4.5 / 20.0, abs=1e-4)


def test_walk_cycles_without_flips(spec):
    clip = clip_for(walk_bvh(frames=240), spec)
    thigh = np.array(clip.angles["thigh_near"])
    assert np.abs(np.diff(thigh)).max() < 5.0  # 120 fps, no flips
    assert 40 < thigh.max() - thigh.min() < 60  # +/- 25 deg swing
    centered = thigh - thigh.mean()
    crossings = np.count_nonzero(np.diff(np.sign(centered)) != 0)
    assert crossings >= 3  # two full cycles
    far = np.array(clip.angles["thigh_far"])
    assert np.corrcoef(thigh, far)[0, 1] < -0.9  # legs in anti-phase
    assert clip.root.x[-1] > clip.root.x[0]  # walks toward screen right
    assert not any(clip.swaps["thigh"]) and not any(clip.swaps["upper_arm"])
    assert clip.warnings == []


def test_facing_direction_does_not_matter(spec):
    a = clip_for(walk_bvh(frames=60), spec)
    for yaw in (90.0, -135.0):
        b = clip_for(walk_bvh(frames=60, yaw=yaw), spec)
        for bone in a.bones:
            np.testing.assert_allclose(a.angles[bone], b.angles[bone], atol=1e-2, err_msg=bone)
        np.testing.assert_allclose(a.root.x, b.root.x, atol=1e-3)


def test_near_far_swap_when_far_leg_crosses(spec):
    def pose(i, t):
        # second half: the far (left) leg swings across the body toward the camera (-X)
        return {"LeftUpLeg": (-40.0 if i >= 5 else 0.0, 0.0, 0.0)}

    clip = clip_for(make_bvh(pose, 10, 30.0), spec)
    assert clip.swaps["thigh"] == [False] * 5 + [True] * 5
    assert not any(clip.swaps["upper_arm"])
    order = draw_order_for(spec, {"thigh": True})
    assert order.index("thigh_far") > order.index("thigh_near")
    assert order.index("foot_far") > order.index("torso")
    assert order.index("upper_arm_far") < order.index("torso")  # arms untouched
    assert sorted(order) == sorted(spec.draw_order)


def test_swap_hysteresis_prevents_flicker(spec):
    def pose(i, t):  # far leg hovering right at the near leg's depth, jittering
        return {"LeftUpLeg": (-20.4 + (0.5 if i % 2 else -0.5), 0.0, 0.0)}

    clip = clip_for(make_bvh(pose, 12, 30.0), spec)
    track = clip.swaps["thigh"]
    assert sum(a != b for a, b in itertools.pairwise(track)) == 0


def test_angles_unwrap_through_full_turns(spec):
    clip = clip_for(make_bvh(lambda i, t: {"RightUpLeg": (0, 0, -6.0 * i)}, 150, 30.0), spec)
    thigh = np.array(clip.angles["thigh_near"])
    assert np.abs(np.diff(thigh)).max() < 7.0
    assert thigh[-1] - thigh[0] == pytest.approx(-6.0 * 149, abs=1e-3)
    local = np.array(clip.local_angles["thigh_near"])
    assert local.max() <= 180 and local.min() > -180


def test_turning_clip_warns(spec):
    clip = clip_for(make_bvh(lambda i, t: {"Hips": (0, 120.0 * i / 9, 0)}, 10, 30.0), spec)
    assert any("turns" in w for w in clip.warnings)


def test_resample_and_range(spec):
    text = walk_bvh(frames=121, fps=120.0)
    full = clip_for(text, spec)
    clip = clip_for(text, spec, fps=30)
    assert clip.frame_count == 31 and clip.fps == 30
    np.testing.assert_allclose(clip.angles["thigh_near"], full.angles["thigh_near"][::4], atol=1e-3)
    part = clip_for(text, spec, start=10, end=50)
    assert part.frame_count == 40
    assert part.angles["thigh_near"][0] == pytest.approx(full.angles["thigh_near"][10], abs=1e-3)


def test_in_place(spec):
    clip = clip_for(walk_bvh(frames=60), spec, in_place=True)
    assert clip.in_place and set(clip.root.x) == {0.0}
    assert max(clip.root.y) != min(clip.root.y)  # vertical bob kept


def test_missing_joints_reported(spec):
    text = walk_bvh(frames=4).replace("RightToeBase", "RToe")
    with pytest.raises(RetargetError, match="RightToeBase"):
        clip_for(text, spec)


def test_unknown_profile(spec):
    with pytest.raises(RetargetError, match="available"):
        retarget(parse_bvh(walk_bvh(frames=4)), spec, "no_such_profile")


def test_zero_length_bone_rejected(spec):
    raw = spec.model_dump()
    raw["mocap_profiles"]["cmu_bvh"]["hand_near"] = ["RightHand", "RightFingerBase"]
    bad = type(spec).model_validate(raw)
    with pytest.raises(RetargetError, match="coincident"):
        clip_for(walk_bvh(frames=4), bad)


def test_fk_reconstructs_projected_joints(spec):
    # The 2D pose from angles + lengths must land the foot where the mocap foot projects.
    clip = clip_for(static_bvh({"RightUpLeg": (0, 0, -30.0), "RightLeg": (0, 0, 50.0)}), spec)
    pose = pose_frame(spec, clip, 0)
    (_, _), (fx, fy) = pose["shin_near"]
    hip_y = clip.root.y[0]
    t, s = math.radians(-30.0), math.radians(20.0)
    exp_x = (4.5 * -math.sin(t) + 4.5 * -math.sin(s)) / 20.0
    exp_y = hip_y + (1 + 4.5 * math.cos(t) + 4.5 * math.cos(s)) / 20.0
    assert fx == pytest.approx(exp_x, abs=1e-3)
    # the hip joint sits 1 unit below Hips; the rig attaches thighs at the pelvis start, so FK is
    # off by exactly that offset in y (documented approximation of the cut-out model)
    assert fy == pytest.approx(exp_y - 1 / 20.0, abs=1e-3)


def test_cli_end_to_end(tmp_path, capsys):
    bvh = tmp_path / "walk.bvh"
    bvh.write_text(walk_bvh(frames=120), encoding="utf-8")
    out = tmp_path / "walk.json"

    assert main(["mocap", "inspect", str(bvh)]) == 0
    assert "profile cmu_bvh: OK" in capsys.readouterr().out

    assert main(["mocap", "retarget", str(bvh), "--profile", "cmu_bvh", "--out", str(out),
                 "--fps", "24"]) == 0
    clip = load_clip(out)
    assert clip.rig == "humanoid_side_v1" and clip.fps == 24 and clip.frame_count == 24
    assert json.loads(out.read_text(encoding="utf-8"))["format"] == "ggs.clip"

    assert main(["mocap", "retarget", str(bvh), "--profile", "mixamo", "--out", str(out)]) == 1
    assert "missing" in capsys.readouterr().err

    pytest.importorskip("PIL")
    png, gif = tmp_path / "strip.png", tmp_path / "walk.gif"
    assert main(["mocap", "preview", str(out), "--out", str(png), "--frames", "6"]) == 0
    assert main(["mocap", "preview", str(out), "--out", str(gif)]) == 0
    from PIL import Image

    assert Image.open(png).size == (6 * 160, 160)
    assert Image.open(gif).n_frames == 24
