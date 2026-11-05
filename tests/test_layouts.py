"""Body layouts (docs/BODY_LAYOUTS.md), clip playback rules, trusted body parsing, piece editing (CPU)."""

import numpy as np
import pytest

pytest.importorskip("PIL")

from ggs.paths import rigs_dir
from ggs.rig import load_rig_spec
from ggs.rig.layout import (
    PRESETS,
    Layout,
    capture_pose,
    expand_clip,
    far_stays_back,
    layout_rig,
    with_lead_in,
)
from ggs.rig.library import library_clip


@pytest.fixture(scope="module")
def base():
    return load_rig_spec(rigs_dir() / "humanoid_side_v1.yaml")


@pytest.fixture(scope="module")
def walk():
    from ggs.rig.clip import load_clip

    p = library_clip("walk")
    if not p.exists():
        pytest.skip("motion library not built")
    return load_clip(p)


def test_layout_rig_adds_bones_parts_and_draw_order(base):
    tall = layout_rig(base, Layout(neck="separate", trunk="two", extras=["skirt", "hair_back", "cape"]))
    names = [b.name for b in tall.bones]
    assert {"chest", "neck", "skirt", "hair_back", "cape"} <= set(names) and tall.id == base.id
    assert tall.bone("head").parent == "neck" and tall.bone("upper_arm_near").parent == "chest"
    assert tall.draw_order[:2] == ["cape", "hair_back"]  # loose parts behind everything
    legs = [tall.draw_order.index(b) for b in tall.draw_order if b.startswith(("thigh", "shin", "foot"))]
    assert max(legs) < tall.draw_order.index("skirt")  # the skirt over the legs
    assert {p.bone for p in tall.part_sheet.parts} >= {"chest", "neck", "skirt"}
    assert layout_rig(base, PRESETS["standard"]).bones == base.bones  # standard = the plain rig


def test_clips_expand_for_layouts(base, walk):
    rig = layout_rig(base, Layout(neck="separate", trunk="two", extras=["cape"]))
    c = expand_clip(walk, rig)
    assert set(c.angles) == {b.name for b in rig.bones} and c.frame_count == walk.frame_count
    assert all(abs(x) < 60 for x in c.angles["cape"])  # follow-through stays plausible
    mitten = expand_clip(walk, layout_rig(base, PRESETS["chibi"]), PRESETS["chibi"])
    assert mitten.angles["hand_near"] == mitten.angles["forearm_near"]  # no wrist bend
    assert capture_pose(Layout(extras=["skirt"]))["thigh_near"] > capture_pose(Layout())["thigh_near"]


def test_far_limbs_stay_back_and_lead_in(walk):
    swapped = walk.model_copy(update={"swaps": {g: [True] * walk.frame_count for g in walk.swaps or {"thigh": 0}}})
    assert not any(any(v) for v in far_stays_back(swapped).swaps.values())
    rest = {b: 180.0 if b in ("pelvis", "torso", "head") else 0.0 for b in walk.angles}
    led = with_lead_in(walk, rest, 0.5)
    n = led.frame_count - walk.frame_count
    assert n >= 2 and led.angles["torso"][n] == walk.angles["torso"][0]
    assert abs(((led.angles["upper_arm_near"][0] - 0.0 + 180) % 360) - 180) < 1e-6  # starts at the drawing's pose


def test_trusted_parts_put_face_with_head_and_shoes_with_feet(base):
    from ggs.rig import parsing as P
    from ggs.rig.body import PRESETS as BODIES
    from ggs.rig.joints import bone_labels, bone_widths, region_labels
    from ggs.rig.make import guide_puppet
    from ggs.rig.master import render_guide

    puppet, lengths = guide_puppet(base, BODIES["standard"])
    g = render_guide(base, puppet, lengths, size=512)
    rgba = np.asarray(g.image)
    on = rgba[..., 3] >= 128
    j = g.spec.joints
    idx = {b.name: i for i, b in enumerate(base.bones)}
    parse = np.where(on, P.UPPER, P.BACKGROUND).astype(np.int8)
    neck_y = int(j["head"][0][1])
    parse[(g.labels == idx["head"]) & on] = P.FACE
    parse[max(0, neck_y - 2):neck_y + 6, :][on[max(0, neck_y - 2):neck_y + 6, :]] = P.FACE  # the neck is face
    feet = (g.labels == idx["foot_near"]) | (g.labels == idx["foot_far"])
    parse[feet & on] = P.R_SHOE
    widths = bone_widths(g.labels, base, j)
    lab = region_labels(bone_labels(rgba[..., 3], base, j, widths, rgb=rgba[..., :3]), rgba, base, j, parse)
    face = (parse == P.FACE) & on
    assert (lab[face] == idx["head"]).mean() > 0.95
    shoe_owner = lab[feet & on]
    assert np.isin(shoe_owner, [idx["foot_near"], idx["foot_far"]]).mean() > 0.95


def test_edited_pieces_go_back_into_the_sheet(tmp_path):
    from PIL import Image

    from ggs import items as I
    from ggs.rig.slices import make_slices, pieces_preview, save_pieces, sliced_dir
    from ggs.style.package import init_project

    p = tmp_path / "g"
    init_project(p, "G")
    I.create(p, "character", "Hero", "a hero", I.ItemSettings(character=I.CharacterSettings(view="side", pose="idle")))
    I.generate(p, "Hero", action="new", model="stub", enforce_vram=False)
    I.accept(p, "Hero")
    make_slices(p, "Hero", "side_simple", enforce_vram=False)
    d = p / "items" / "character" / "hero"
    grid = pieces_preview(d)
    im = Image.open(grid).convert("RGBA")
    im.paste((255, 0, 0, 255), (0, 0, 20, 20))  # paint on the first piece (the head)
    im.save(grid)
    save_pieces(d, grid)
    sheet = Image.open(sliced_dir(d) / "parts.png").convert("RGBA")
    assert sheet.getpixel((5, 5)) == (255, 0, 0, 255)
    c = sheet.width // 4
    variant = sheet.crop((c * 1, c * 2, c * 2, c * 3))  # the first expression slot copies the edited head
    assert variant.getpixel((5, 5)) == (255, 0, 0, 255)
