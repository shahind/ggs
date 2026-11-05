"""M8 asset kinds (background, tile, ui_panel, ui_icon, vfx): helpers and the stub pipeline, on CPU."""

import json

import numpy as np
import pytest

pytest.importorskip("PIL")

from ggs.assets.seamless import band_mask, roll_half, seam_score
from ggs.assets.ui_panel import find_slices, state
from ggs.assets.vfx import additive_rgba
from ggs.generate import run_recipe
from ggs.recipes import PanelSettings, available_recipes, load_recipe


def test_recipes_load():
    names = set(available_recipes())
    assert {"background", "background_wrap", "tile", "ui_panel", "ui_icon", "vfx", "sprite", "held_item"} <= names
    assert load_recipe("background").layers[0].opaque and load_recipe("background_wrap").seamless_x
    assert load_recipe("tile").tile.seamless == "xy"
    assert load_recipe("vfx").background_rgb == (0, 0, 0)


def test_seam_score_and_roll():
    x = np.linspace(0, 4 * np.pi, 64, endpoint=False)
    periodic = (127 + 100 * np.sin(x))[None, :, None].repeat(64, 0).repeat(3, 2).astype(np.uint8)
    ramp = np.linspace(0, 255, 64)[None, :, None].repeat(64, 0).repeat(3, 2).astype(np.uint8)
    assert seam_score(periodic, "x") < 0.98 < seam_score(ramp, "x")
    rolled = roll_half(ramp, "x")
    assert rolled[0, 0, 0] == ramp[0, 32, 0]  # the wrap seam moved to the middle
    m = band_mask(64, 64, "xy", 0.2)
    assert m[32, 32] == 1 and m[0, 0] == 0 and m[5, 31] == pytest.approx(1)


def test_find_slices_and_states():
    panel = np.zeros((100, 160, 4), np.uint8)
    panel[...] = (120, 80, 40, 255)  # frame colour
    panel[12:88, 20:140, :3] = (220, 210, 190)  # plain centre
    m, scores = find_slices(panel, PanelSettings())
    assert m == {"top": 12, "bottom": 12, "left": 20, "right": 20}
    assert scores["centre_spread"] < 1
    assert state(panel, 1.2)[50, 80, 0] == 255 and state(panel, 0.5)[50, 80, 0] == 110


def test_additive_rgba():
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[1, 1] = (255, 128, 0)
    rgb[2, 2] = (64, 64, 128)
    out = additive_rgba(rgb)
    assert out[0, 0, 3] == 0 and out[1, 1, 3] == 255 and out[2, 2, 3] == 128
    assert tuple(out[1, 1, :3]) == (255, 128, 0) and tuple(out[2, 2, :3]) == (127, 127, 255)


@pytest.mark.parametrize("cls,files", [
    ("background", ["bg_far.png", "bg.png"]),
    ("background_wrap", ["bg_far.png", "bg.png"]),
    ("tile", ["bg.png"]),
    ("ui_panel", ["bg.png", "bg_hover.png", "bg_pressed.png", "bg_slices.json"]),
    ("ui_icon", ["bg.png"]),
    ("vfx", ["bg.png"]),
])
def test_stub_pipeline_writes_files(tmp_path, cls, files):
    results, meta = run_recipe(load_recipe(cls), "a forest", tmp_path / "bg.png", model="stub", seeds=[0],
                               enforce_vram=False, keep_failed=True, size=(512, 512))
    assert len(results) == 1
    for f in files:
        assert (tmp_path / f).exists(), f
    side = json.loads((tmp_path / "bg.json").read_text(encoding="utf-8"))
    assert side["class"] == cls and side["assets"][0]["seed"] == 0
    if cls.startswith("background"):
        layers = meta["background"]["sets"][0]["layers"]
        assert [x["name"] for x in layers] == ["far"] and layers[0]["opaque"]
