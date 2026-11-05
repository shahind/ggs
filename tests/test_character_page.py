"""Characters v3 (owner review 2026-11-05): stick-figure pose references, accepted designs, sliced versions from
templates, drawn frames, exports with several animations or none, separate limb lengths (stub model, CPU)."""

import json

import pytest

pytest.importorskip("PIL")

from ggs import items as I
from ggs.cli import main
from ggs.style.package import init_project


@pytest.fixture()
def hero(tmp_path):
    p = tmp_path / "g"
    init_project(p, "G")
    I.create(p, "character", "Hero", "a hero",
             I.ItemSettings(character=I.CharacterSettings(view="side", pose="idle")))
    I.generate(p, "Hero", action="new", model="stub", enforce_vram=False)
    I.accept(p, "Hero")
    return p


def test_stick_figure_reference_has_no_puppet_shapes():
    import numpy as np

    from ggs.rig.make import pose_reference
    from ggs.rig.stick import openpose_image, openpose_points

    rig, guide, _ = I.character_guide(I.CharacterSettings(view="side", pose="walk"), size=512)
    stick = np.asarray(pose_reference(rig, guide))
    assert stick.shape == (512, 512, 3)
    ink = (stick < 200).any(axis=2)
    assert 0 < ink.mean() < 0.06  # thin lines, not filled body shapes
    assert len(set(map(tuple, stick[ink].reshape(-1, 3)))) <= 40  # greys only, no puppet colours
    pts = openpose_points(guide.spec.joints, "side")
    assert pts[1] == guide.spec.joints["head"][0] and pts[15] is None and pts[17] is None  # far eye/ear hidden
    assert openpose_image(guide.spec.joints, (512, 512), "side").getpixel((0, 0)) == (0, 0, 0)


def test_separate_limb_lengths():
    from ggs.rig.body import BodyShape, body_lengths, parse_body
    from ggs.rig.poses import rig_for

    rig = rig_for("side")
    base = body_lengths(rig, BodyShape())
    long_fore = body_lengths(rig, BodyShape(forearm=1.5))
    assert long_fore["forearm_near"] / long_fore["upper_arm_near"] > 1.4 * base["forearm_near"] / base["upper_arm_near"]
    assert parse_body("thigh=1.3,shin=0.8").factor("shin") == 0.8
    assert BodyShape(arms=1.2).factor("upper_arm") == 1.2  # no own value: the arms factor


def test_templates_and_previews():
    from ggs.rig.slices import load_templates, template_preview

    ts = load_templates()
    assert {t.id for t in ts} >= {"side_simple", "side_detailed", "front"}
    assert all(template_preview(t, 120).exists() for t in ts)


def test_slice_then_export_parts_alone(hero, tmp_path):
    from ggs.rig.slices import make_slices, sliced_character, sliced_info

    d = hero / "items" / "character" / "hero"
    res = make_slices(hero, "Hero", "side_simple", enforce_vram=False)
    assert res.character == sliced_character(d) and not res.rough, res.tries
    assert sliced_info(d)["template"] == "side_simple" and (d / "sliced" / "parts.png").exists()
    out = tmp_path / "pkg"
    assert main(["export", "--character", str(res.character), "--format", "godot", "--out", str(out)]) == 0
    rig = json.loads((out / "rig.json").read_text(encoding="utf-8"))
    assert rig["clips"] == ["pose"]  # the parts on their own: one still frame
    assert main(["export", "--character", str(res.character), "--format", "gif", "--out", str(tmp_path / "x.gif"),
                 "--matte", "#FFFFFF"]) == 1  # a picture format needs an animation


def test_export_several_animations(hero, tmp_path):
    from ggs.rig.library import library_clip
    from ggs.rig.slices import make_slices

    ch = make_slices(hero, "Hero", "side_simple", enforce_vram=False).character
    clips = [str(library_clip(n)) for n in ("walk", "punch")]
    if not all(__import__("pathlib").Path(c).exists() for c in clips):
        pytest.skip("motion library not built (tools/make_motion_library.py)")
    assert main(["export", *clips, "--character", str(ch), "--format", "unity", "--out", str(tmp_path / "u")]) == 0
    assert set(json.loads((tmp_path / "u" / "rig.json").read_text(encoding="utf-8"))["clips"]) == {"walk", "punch"}
    assert main(["export", *clips, "--character", str(ch), "--format", "webp", "--out", str(tmp_path / "w")]) == 0
    assert (tmp_path / "w" / "walk.webp").exists() and (tmp_path / "w" / "punch.webp").exists()


def test_drawn_frames_from_design(hero):
    from ggs.rig.library import library_clip
    from ggs.rig.sprites import make_frames, pose_preview

    clip = library_clip("walk")
    if not clip.exists():
        pytest.skip("motion library not built")
    sheet = pose_preview(clip, 6)
    assert sheet.size == (4 * 256, 2 * 384)
    out, _results = make_frames(hero, "Hero", clip, frames=6, enforce_vram=False)
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep["frames"] == 6 and len(list((out / "frames").glob("*.png"))) == 6
    assert (out / "walk_sheet.png").exists() and (out / "poses.png").exists()


def test_slice_needs_a_design(tmp_path):
    from ggs.rig.slices import make_slices

    p = tmp_path / "g"
    init_project(p, "G")
    I.create(p, "character", "Nobody", "x")
    with pytest.raises(I.ItemError, match="no design"):
        make_slices(p, "Nobody", "side_simple", enforce_vram=False)
