"""v2 project items: create, actions (new, refine, regenerate, variations, goto), history, export (stub, CPU)."""

import json

import pytest

pytest.importorskip("PIL")

from ggs import items as I
from ggs.cli import main
from ggs.style.package import init_project


@pytest.fixture()
def project(tmp_path):
    init_project(tmp_path / "game", "Game")
    return tmp_path / "game"


def gen(project, ref, **kw):
    return I.generate(project, ref, model="stub", enforce_vram=False, **kw)


def test_create_and_lookup(project):
    _, d = I.create(project, "sprite", "Health potion", "a red health potion")
    assert d == project / "items" / "sprite" / "health_potion" and (d / "item.json").exists()
    assert I.load(project, "Health potion")[0].name == "Health potion"
    assert I.load(project, "sprite/Health potion")[1] == d
    with pytest.raises(I.ItemError, match="already"):
        I.create(project, "sprite", "Health potion")
    with pytest.raises(I.ItemError, match="unknown asset type"):
        I.create(project, "spaceship", "x")
    I.create(project, "ui_icon", "Health potion")
    with pytest.raises(I.ItemError, match="ambiguous"):
        I.load(project, "Health potion")


def test_actions_keep_history_not_images(project):
    I.create(project, "sprite", "Potion", "a red health potion")
    with pytest.raises(I.ItemError, match="nothing to refine"):
        gen(project, "Potion", action="refine", text="make it blue")
    rec, new = gen(project, "Potion", action="new")
    assert [v.n for v in new] == [1] and rec.current == 1 and new[0].file == "versions/v001.png"
    rec, new = gen(project, "Potion", action="variations", count=3)
    assert [v.n for v in new] == [2, 3, 4] and len({v.seed for v in new}) == 3
    rec, new = gen(project, "Potion", action="regenerate")
    assert new[0].n == 5 and new[0].prompt == "a red health potion"
    before = rec.get_version()
    rec, new = gen(project, "Potion", action="refine", text="make it blue")
    v = new[0]
    assert v.seed == before.seed and v.parent == before.n and v.prompt == "make it blue"
    item = project / "items" / "sprite" / "potion"
    names = {f.name for f in (item / "versions").iterdir()}
    assert "v006.png" in names and "v006_raw.png" in names  # only the drawing on screen stays in the project
    assert not any(n.startswith(("v001", "v002", "v005")) for n in names)
    assert not any("QA_FAILED" in n or "_01" in n for n in names)
    assert rec.design_text() == ["a red health potion", "make it blue"] and v.description == "a red health potion"
    # going back: from the cache when it has the drawing ...
    rec, _ = gen(project, "Potion", action="goto", version=2)
    assert rec.current == 2 and (item / "versions" / "v002.png").exists()
    assert not (item / "versions" / "v006.png").exists()
    # ... and drawn again from its recipe when it does not (a change is redrawn on top of its parent)
    import shutil

    shutil.rmtree(I.cache_dir(item))
    rec, _ = gen(project, "Potion", action="goto", version=6)
    assert rec.current == 6 and (item / "versions" / "v006.png").exists()
    with pytest.raises(I.ItemError, match="no drawing"):
        gen(project, "Potion", action="goto", version=99)


def test_export_sizes(project):
    from PIL import Image

    I.create(project, "sprite", "Gem", "a blue gem")
    gen(project, "Gem", action="new")
    out = I.export_item(project, "Gem", size="small")
    assert max(Image.open(out).size) <= 128
    meta = json.loads(out.with_suffix(".export.json").read_text(encoding="utf-8"))
    assert meta["version"] == 1 and meta["class"] == "sprite"


def test_friendly_errors():
    text = I.friendly(["halo: 30% of the rim", "cut_off: touches the top edge"])
    assert "light rim" in text and "edge of the picture" in text and "Regenerate" in text


def test_cli_asset_flow(project):
    p = str(project)
    assert main(["asset", "new", "Panel", "--class", "ui_panel", "--project", p, "--describe", "a stone panel"]) == 0
    assert main(["asset", "gen", "Panel", "--project", p, "--model", "stub", "--no-vram-check"]) in (0, 2)
    assert main(["asset", "list", "--project", p]) == 0
    assert main(["asset", "export", "Panel", "--project", p]) == 0
    assert (project / "exports" / "ui_panel" / "panel.png").exists()
    assert main(["asset", "gen", "Nope", "--project", p, "--model", "stub"]) == 1
