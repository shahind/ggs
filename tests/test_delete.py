"""Deleting: drawings from the history, whole items, sliced versions, drawn frames, animations, projects (CPU)."""

import pytest

pytest.importorskip("PIL")

from ggs import items as I
from ggs.cli import main
from ggs.style.package import StyleError, delete_project, init_project


@pytest.fixture()
def project(tmp_path):
    init_project(tmp_path / "game", "Game")
    return tmp_path / "game"


def gen(project, ref, **kw):
    return I.generate(project, ref, model="stub", enforce_vram=False, **kw)


def test_delete_drawing_steps_back_and_keeps_recipes_a_change_needs(project):
    I.create(project, "sprite", "Potion", "a red health potion")
    gen(project, "Potion", action="new")          # 1
    gen(project, "Potion", action="regenerate")   # 2
    rec, _ = gen(project, "Potion", action="refine", text="make it blue")  # 3, drawn on top of 2
    item = project / "items" / "sprite" / "potion"
    assert rec.current == 3
    # deleting the one on screen shows the one before it (2, from the cache)
    rec, redraw = I.delete_drawing(project, "Potion")
    assert not redraw and rec.current == 2 and [v.n for v in rec.shown()] == [1, 2]
    assert (item / "versions" / "v002.png").exists() and not list(I.cache_dir(item).glob("v003*"))
    # a deleted drawing that a change was drawn on top of stays as a hidden recipe
    gen(project, "Potion", action="refine", text="make it green")  # 3 again? no: numbers go on from the highest
    rec = I.load(project, "Potion")[0]
    child = rec.get_version()
    assert child.parent == 2
    rec, _ = I.delete_drawing(project, "Potion", 2)
    assert 2 not in [v.n for v in rec.shown()] and rec.get_version(2).hidden
    assert rec.design_text(child.n) == ["a red health potion", "make it green"]
    # ... and is forgotten once nothing shown needs it
    rec, _ = I.delete_drawing(project, "Potion", child.n)
    assert rec.get_version(2) is None and [v.n for v in rec.shown()] == [1] and rec.current == 1
    rec, _ = I.delete_drawing(project, "Potion")
    assert rec.current is None and rec.shown() == []
    with pytest.raises(I.ItemError, match="no drawing"):
        I.delete_drawing(project, "Potion", 7)


def test_delete_drawing_redraws_when_not_cached(project):
    import shutil

    I.create(project, "sprite", "Potion", "a red health potion")
    gen(project, "Potion", action="new")
    gen(project, "Potion", action="regenerate")
    item = project / "items" / "sprite" / "potion"
    shutil.rmtree(I.cache_dir(item))
    rec, redraw = I.delete_drawing(project, "Potion")
    assert redraw and rec.current == 1
    assert main(["asset", "gen", "Potion", "--project", str(project), "--action", "goto", "--version", "1",
                 "--model", "stub", "--no-vram-check"]) == 0
    assert (item / "versions" / "v001.png").exists()


def test_design_drawing_cannot_be_deleted_and_item_delete(project):
    I.create(project, "character", "Hero", "a hero",
             I.ItemSettings(character=I.CharacterSettings(view="side", pose="idle")))
    gen(project, "Hero", action="new")
    I.accept(project, "Hero")
    with pytest.raises(I.ItemError, match="design"):
        I.delete_drawing(project, "Hero")
    d = project / "items" / "character" / "hero"
    (d / "sliced").mkdir()
    (d / "sliced" / "parts.png").write_bytes(b"x")
    (d / "frames" / "walk").mkdir(parents=True)
    p = str(project)
    assert main(["asset", "delete", "Hero", "--project", p, "--sliced"]) == 0 and not (d / "sliced").exists()
    assert main(["asset", "delete", "Hero", "--project", p, "--sliced"]) == 1
    assert main(["asset", "delete", "Hero", "--project", p, "--frames", "../x"]) == 1
    assert main(["asset", "delete", "Hero", "--project", p, "--frames", "walk"]) == 0
    assert not (d / "frames" / "walk").exists()
    assert main(["asset", "delete", "Hero", "--project", p]) == 0
    assert not d.exists() and I.list_items(project) == []
    assert main(["asset", "delete", "Hero", "--project", p]) == 1


def test_cli_delete_drawing_and_list(project, capsys):
    I.create(project, "sprite", "Potion", "a red health potion")
    gen(project, "Potion", action="new")
    gen(project, "Potion", action="regenerate")
    assert main(["asset", "delete", "Potion", "--project", str(project), "--current"]) == 0
    assert "on screen now: v001" in capsys.readouterr().out
    main(["asset", "list", "--project", str(project)])
    assert "  1 drawings" in capsys.readouterr().out


def test_delete_animation_and_project(project, tmp_path):
    clips = project / "clips"
    clips.mkdir()
    for ext in (".json", ".gif", ".motion.json"):
        (clips / f"wave{ext}").write_text("{}", encoding="utf-8")
    (clips / "walk.json").write_text("{}", encoding="utf-8")
    assert main(["motion", "remove", "wave", "--project", str(project)]) == 0
    assert sorted(f.name for f in clips.iterdir()) == ["walk.json"]
    assert main(["motion", "remove", "nope", "--project", str(project)]) == 1
    assert main(["motion", "remove", "../walk", "--project", str(project)]) == 1
    with pytest.raises(StyleError, match="not a GGS project"):
        delete_project(tmp_path)
    assert main(["project", "delete", str(project)]) == 0 and not project.exists()
