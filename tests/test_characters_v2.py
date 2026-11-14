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
    assert v.character.view == "front" and v.model == "stub"  # the recipe can redraw it
    rec, new = I.generate(p, "Knight", action="refine", text="add a red cape", model="stub", enforce_vram=False)
    assert new[0].parent == 1 and new[0].seed == v.seed
    rec, design = I.accept(p, "Knight")
    assert rec.accepted == 2 and (design / "design.png").exists() and (design / "design_skeleton.json").exists()
    assert json.loads((design / "design.json").read_text(encoding="utf-8"))["text"] == ["a knight", "add a red cape"]
    I.generate(p, "Knight", action="regenerate", model="stub", enforce_vram=False)  # the design stays
    out = I.export_item(p, "Knight")
    assert json.loads(out.with_suffix(".export.json").read_text(encoding="utf-8"))["version"] == 2
    sk = json.loads(out.with_name(out.stem + "_skeleton.json").read_text(encoding="utf-8"))
    assert sk["image"] == out.name and sk["size"] == list(__import__("PIL.Image").Image.open(out).size)


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
    assert ed.cs.body["head"] == 1.8 and ed.body_head.save.isVisibleTo(ed)
    ed.sliders["forearm"].setValue(150)
    assert ed.body_shape().factor("forearm") == 1.5 and ed.body_shape().factor("upper_arm") == 1.0
    ed._emit()
    assert got and got[-1].body["head"] == 1.8
    side = next(b for b in ed.views.buttons() if b.property("view") == "side")
    ed._view(side)
    assert ed.cs.view == "side" and ed.cs.pose == "idle" and len(ed.poses.buttons()) == 3


@pytest.fixture()
def library_home(tmp_path, monkeypatch):
    import ggs.library as L

    monkeypatch.setattr(L, "shared_dir", lambda: tmp_path / "home")
    return tmp_path / "home"


def test_library_save_list_delete(tmp_path, library_home):
    from ggs import library as L

    init_project(tmp_path / "g", "G")
    p = tmp_path / "g"
    names = [s.name for s in L.list_poses("front", p)]
    assert names[:3] == ["T-pose", "A-pose", "Standing"]
    L.save_pose("Wave", "front", {**pose_angles("front", "idle"), "upper_arm_r": -150}, p)
    L.save_pose("Bow", "front", pose_angles("front", "idle"), p, scope="global")
    L.save_pose("Crouch", "side", pose_angles("side", "idle"), p)
    front = {s.name: s.scope for s in L.list_poses("front", p)}
    assert front["Wave"] == "project" and front["Bow"] == "global" and "Crouch" not in front
    assert "Bow" in {s.name for s in L.list_poses("front", tmp_path / "other")}  # global: every project
    L.save_body("Giant", {"head": 0.8, "legs": 1.3, "width": 1.4}, p)
    bodies = {s.name: s for s in L.list_bodies(p)}
    assert bodies["Giant"].data["legs"] == 1.3 and "Chibi" in bodies
    with pytest.raises(ValueError):
        L.save_body("Bad", {"head": 9}, p)
    with pytest.raises(ValueError):
        L.save_pose("  ", "front", {}, p)
    L.delete_pose("Wave", p)
    assert "Wave" not in {s.name for s in L.list_poses("front", p)}


def test_editor_saves_and_offers_presets(qapp, tmp_path, library_home):
    from ggs.ui.character_editor import BodyPoseEditor

    init_project(tmp_path / "g", "G")
    ed = BodyPoseEditor(I.CharacterSettings(view="front", pose="a_pose"), tmp_path / "g")
    n_poses, n_bodies = len(ed.poses.buttons()), len(ed.bodies.buttons())
    ed._dragged("forearm_r", -120.0)
    ed._save_pose("Salute")
    assert len(ed.poses.buttons()) == n_poses + 1
    assert ed.poses.checkedButton().property("saved").name == "Salute"
    ed.sliders["thigh"].setValue(150)
    ed._save_body("Long legs")
    assert len(ed.bodies.buttons()) == n_bodies + 1 and ed.cs.body["name"] == "Long legs"
    other = BodyPoseEditor(I.CharacterSettings(view="front"), tmp_path / "g")  # a second character sees them
    assert "Salute" in [b.property("saved").name for b in other.poses.buttons()]


def test_body_sliders_at_their_ends_stay_valid(qapp):
    """Every slider pushed to its far end gives a body the model accepts (the 2.2 vs <=2.0 crash)."""
    from ggs.rig.body import BodyShape, limits
    from ggs.ui.character_editor import BodyPoseEditor

    ed = BodyPoseEditor(I.CharacterSettings(view="side", pose="idle"))
    for key, sl in ed.sliders.items():
        lo, hi = limits(key)
        assert lo <= sl.minimum() / 100 and sl.maximum() / 100 <= hi, key
    for end in ("maximum", "minimum"):
        for sl in ed.sliders.values():
            sl.setValue(getattr(sl, end)())
        BodyShape(**ed.cs.body)  # strict: no clamping needed
    ed._emit()


def test_a_body_saved_out_of_range_still_opens(qapp):
    from ggs.rig.body import saved_body
    from ggs.ui.character_editor import BodyPoseEditor

    bad = {"name": "custom", "head": 1.0, "torso": 2.2, "thigh": 2.13, "shin": 0.1, "width": 9.0}
    b = saved_body(bad)
    assert b.thigh == 2.0 and b.torso == 2.0 and b.width == 2.5 and 0.3 < b.shin < 0.32
    ed = BodyPoseEditor(I.CharacterSettings(view="side", pose="idle", body=bad))
    assert ed.sliders["thigh"].value() == 200
