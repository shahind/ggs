"""Body shapes (proportions of the pose guide), stature, and the part refiner's plumbing (stub model, CPU)."""

import json

import numpy as np
import pytest
import yaml

pytest.importorskip("PIL")

from ggs.cli import main
from ggs.paths import repo_root
from ggs.rig.body import PRESETS, body_lengths, parse_body, standing_height
from ggs.rig.make import cut_master, guide_only_master, make_masters
from ggs.rig.refine import refine_characters
from ggs.rig.spec import load_rig_spec

SIZE = 512


@pytest.fixture(scope="module")
def rig():
    return load_rig_spec(repo_root() / "rigs" / "humanoid_side_v1.yaml")


def test_parse_body():
    assert parse_body(None).name == "standard"
    assert parse_body("Chibi").head > 1.5
    custom = parse_body("head=1.6, legs=0.7")
    assert (custom.head, custom.legs, custom.arms) == (1.6, 0.7, 1.0)
    assert "large head" in custom.describe() and "short legs" in custom.describe()
    for bad in ("giant", "head=big", "tail=2", "head=9"):
        with pytest.raises(ValueError):
            parse_body(bad)


def test_body_lengths_keep_one_standing_height(rig):
    rest = {b.name: b.rest_length for b in rig.bones}
    for name, body in PRESETS.items():
        lengths = body_lengths(rig, body)
        assert standing_height(lengths, body.head) == pytest.approx(standing_height(rest), rel=0.03), name
        assert lengths["thigh_far"] == lengths["thigh_near"]
    chibi, lanky = body_lengths(rig, PRESETS["chibi"]), body_lengths(rig, PRESETS["lanky"])
    assert chibi["head"] / chibi["thigh_near"] > 3 * rest["head"] / rest["thigh_near"]
    assert lanky["shin_near"] > rest["shin_near"] > chibi["shin_near"]


@pytest.mark.parametrize("body", ["chibi", "lanky", "stocky"])
def test_every_body_cuts_cleanly(rig, tmp_path, body):
    out = cut_master(guide_only_master(rig, tmp_path / "m", SIZE, body=PRESETS[body]), rig, tmp_path / "c", body)
    assert out.report.passed, out.report.failures()
    lengths = yaml.safe_load((tmp_path / "c" / "character.yaml").read_text(encoding="utf-8"))["bone_lengths"]
    assert lengths["head"] == pytest.approx(body_lengths(rig, PRESETS[body])["head"], rel=0.02)
    assert json.loads((tmp_path / "m" / "master.json").read_text(encoding="utf-8"))["body"]["name"] == body


def test_body_words_reach_the_master_prompt(rig, tmp_path):
    (m,) = make_masters(rig, "a knight", tmp_path, seeds=[0], model="stub", enforce_vram=False, size=SIZE,
                        body=PRESETS["chibi"])
    assert "chibi proportions" in json.loads(m.read_text(encoding="utf-8"))["prompt"]


def test_stature_scales_exports(tmp_path):
    from PIL import Image

    walk = tmp_path / "walk.json"
    assert main(["character", "programmer-art", "--out", str(tmp_path / "pa"), "--body", "lanky"]) == 0
    char = tmp_path / "pa" / "character.yaml"
    from ggs.rig.master import MASTER_POSE, pose_clip

    rig = load_rig_spec(repo_root() / "rigs" / "humanoid_side_v1.yaml")
    lengths = yaml.safe_load(char.read_text(encoding="utf-8"))["bone_lengths"]
    clip = pose_clip(rig, MASTER_POSE, lengths)
    clip.root.y = [-(lengths["thigh_near"] + lengths["shin_near"])]  # hips above the ground, like a real clip
    clip.save(walk)
    heights = []
    for stature in (1.0, 1.5):
        data = yaml.safe_load(char.read_text(encoding="utf-8"))
        data["stature"] = stature
        char.write_text(yaml.safe_dump(data), encoding="utf-8")
        out = tmp_path / f"s{stature}.png"
        assert main(["export", str(walk), "--character", str(char), "--format", "apng", "--out", str(out),
                     "--height", "120"]) == 0
        heights.append(Image.open(out).size[1])
    assert heights[1] > 1.35 * heights[0]


def test_refiner_plumbing_with_stub(rig, tmp_path):
    out = cut_master(guide_only_master(rig, tmp_path / "m", SIZE), rig, tmp_path / "c", "pa")
    assert (tmp_path / "c" / "parts_filled.png").exists()
    (rep,) = refine_characters([tmp_path / "c"], rig, model="stub", enforce_vram=False)
    kinds = {e["kind"] for e in rep.edits}
    assert kinds == {"fill", "expression"}
    assert {e.get("variant") for e in rep.edits if e["kind"] == "expression"} == {"blink", "mouth_open", "hurt"}
    assert (tmp_path / "c" / "refine.json").exists()
    if rep.applied:
        assert (tmp_path / "c" / "parts_cut.png").exists()
    # the stub "edit" returns its input, so visible pixels and every part's shape stay the same
    from PIL import Image

    before = np.asarray(Image.open(tmp_path / "c" / "parts_cut.png" if rep.applied else tmp_path / "c" / "parts.png"))
    after = np.asarray(Image.open(tmp_path / "c" / "parts.png"))
    cell = rig.part_sheet.cell_px
    for p in rig.part_sheet.parts:
        c, r = p.cell
        a0 = before[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell, 3]
        a1 = after[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell, 3]
        assert np.array_equal(a0 > 0, a1 > 0), p.name
    assert out.report.passed
