"""Characters v2: poses, front-view rig, character items (stub) and the body & pose editor."""

import json
import os

import pytest

pytest.importorskip("PIL")

from ggs import items as I
from ggs.rig.poses import POSES, pose_angles, rig_for
from ggs.style.package import init_project


def test_pose_presets():
    for view, poses in POSES.items():
        rig = rig_for(view)
        names = {b.name for b in rig.bones}
        for pid in poses:
            assert set(pose_angles(view, pid)) == names, (view, pid)
    t = pose_angles("front", "t_pose")
    assert t["upper_arm_l"] == 90 and t["upper_arm_r"] == -90
    custom = pose_angles("front", {"upper_arm_l": 30})
    from ggs.rig.poses import FRONT_REST

    assert custom["upper_arm_l"] == 30 and custom["thigh_l"] == FRONT_REST["thigh_l"]
    with pytest.raises(ValueError):
        pose_angles("front", "walk")


def test_front_guide_is_symmetric():
    cs = I.CharacterSettings(view="front", pose="t_pose")
    _rig, guide, _ = I.character_guide(cs, size=512)
    j = guide.spec.joints
    mid = j["head"][0][0]
    assert abs((mid - j["hand_l"][1][0]) - (j["hand_r"][1][0] - mid)) < 1.0
    assert j["hand_l"][1][1] < j["torso"][1][1] + 0.1 * guide.spec.height_px  # arms out, not down


def test_character_item_with_stub(tmp_path):
    init_project(tmp_path / "g", "G")
    p = tmp_path / "g"
    rec, d = I.create(p, "character", "Knight", "a knight")
    assert rec.settings.character.view == "front" and rec.settings.character.pose == "a_pose"
    rec, new = I.generate(p, "Knight", action="new", model="stub", enforce_vram=False)
    v = new[0]
    assert v.passed, v.failures
    sk = json.loads((d / "versions" / "v001_skeleton.json").read_text(encoding="utf-8"))
    assert sk["view"] == "front" and {b["name"] for b in sk["bones"]} >= {"hand_l", "hand_r", "head"}
    w, h = sk["size"]
    assert all(0 <= x <= w and 0 <= y <= h for b in sk["bones"] for x, y in (b["start"], b["end"]))
    rec, new = I.generate(p, "Knight", action="refine", text="add a red cape", model="stub", enforce_vram=False)
    assert new[0].parent == 1 and new[0].seed == v.seed
    out = I.export_item(p, "Knight")
    assert out.with_name(out.stem + "_skeleton.json").exists()


def test_limbs_missed_spots_wrong_arms():
    rig, guide, _ = I.character_guide(I.CharacterSettings(view="front", pose="t_pose"), size=512)
    good = guide.labels >= 0
    assert I.limbs_missed(good, guide, rig) == []
    _, idle, _ = I.character_guide(I.CharacterSettings(view="front", pose="idle"), size=512)
    missed = I.limbs_missed(idle.labels >= 0, guide, rig)  # arms down where the pose wants them out
    assert any("forearm" in m for m in missed)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_editor_changes_pose_and_body(qapp, tmp_path):
    from ggs.ui.character_editor import BodyPoseEditor

    ed = BodyPoseEditor(I.CharacterSettings(view="front", pose="a_pose"))
    got = []
    ed.changed.connect(got.append)
    ed._dragged("forearm_l", 120.0)
    assert ed.cs.pose["forearm_l"] == 120.0 and ed.cs.pose["hand_l"] != pose_angles("front", "a_pose")["hand_l"]
    ed.sliders["head"].setValue(180)
    assert ed.cs.body["head"] == 1.8
    ed._render()
    assert got and got[-1].body["head"] == 1.8
    side = next(b for b in ed.views.buttons() if b.property("view") == "side")
    ed._view(side)
    assert ed.cs.view == "side" and ed.cs.pose == "idle" and len(ed.poses.buttons()) == 3
