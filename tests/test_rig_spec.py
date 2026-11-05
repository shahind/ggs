from copy import deepcopy

import pytest
import yaml

from ggs.paths import rigs_dir
from ggs.rig import RigSpec, load_rig_spec

RIG = rigs_dir() / "humanoid_side_v1.yaml"


def _raw():
    return yaml.safe_load(RIG.read_text(encoding="utf-8"))


def test_shipped_rig_is_valid():
    spec = load_rig_spec(RIG)
    assert len(spec.bones) == 15
    assert len(spec.part_sheet.parts) == 9
    assert spec.near_side == "right"


def test_every_bone_has_art():
    spec = load_rig_spec(RIG)
    covered = {p.bone for p in spec.part_sheet.parts} | {
        p.bone_far for p in spec.part_sheet.parts if p.bone_far
    }
    assert covered == {b.name for b in spec.bones}


def test_cell_collision_rejected():
    raw = _raw()
    raw["part_sheet"]["head_variants"][0]["cell"] = raw["part_sheet"]["parts"][0]["cell"]
    with pytest.raises(ValueError, match="cell"):
        RigSpec.model_validate(raw)


def test_unknown_parent_rejected():
    raw = _raw()
    raw["bones"][1]["parent"] = "nope"
    with pytest.raises(ValueError, match="unknown parent"):
        RigSpec.model_validate(raw)


def test_two_roots_rejected():
    raw = deepcopy(_raw())
    raw["bones"][1]["parent"] = None
    with pytest.raises(ValueError, match="root"):
        RigSpec.model_validate(raw)


def test_incomplete_mocap_profile_rejected():
    raw = _raw()
    del raw["mocap_profiles"]["mixamo"]["head"]
    with pytest.raises(ValueError, match="mocap profile"):
        RigSpec.model_validate(raw)


def test_draw_order_must_cover_all_bones():
    raw = _raw()
    raw["draw_order"].pop()
    with pytest.raises(ValueError, match="draw_order"):
        RigSpec.model_validate(raw)
