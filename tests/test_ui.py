"""Desktop app v2: command builders and friendly errors (pure Python), and offscreen smoke tests of the start
screen, project window, wizard, item workspace and job queue."""

import json
import os
import sys

import pytest

from ggs.ui import commands as C
from ggs.ui.errors import friendly_error, stage_text


def test_slug_and_paths(tmp_path):
    assert C.slug("A Red Health-Potion!!") == "a_red_health_potion"
    assert C.slug("???") == "asset"
    p = C.new_asset_path(tmp_path, "sprite", "a lantern")
    assert p.parent == tmp_path / "assets" / "sprite" and p.name.startswith("a_lantern_") and p.suffix == ".png"


def test_command_builders(tmp_path):
    g = C.generate("tile", "cobblestone", tmp_path / "t.png", project=tmp_path, style="ink", style_weight=0.8,
                   count=3, seed=7, size="512x512")
    assert g[:3] == [sys.executable, "-m", "ggs"] and g[g.index("--class") + 1] == "tile"
    a = C.asset_gen(tmp_path, "sprite/Potion", "refine", "make it blue", seed=5)
    assert a[3:5] == ["asset", "gen"] and a[a.index("--action") + 1] == "refine"
    assert a[a.index("--text") + 1] == "make it blue" and a[a.index("--seed") + 1] == "5" and "--model" not in a
    n = C.asset_new(tmp_path, "tile", "Floor", "grey stone", "small")
    assert n[n.index("--class") + 1] == "tile" and n[n.index("--size") + 1] == "small"
    e = C.asset_export(tmp_path, "Floor", "large", 3)
    assert e[e.index("--version") + 1] == "3"
    assert "--i-have-the-rights" in C.dataset_import(tmp_path, tmp_path / "ds", rights=True)


def test_friendly_errors_and_stages():
    assert "graphics card ran out of memory" in friendly_error("VramBudgetError: stage 'denoise' peaked at 9 GB")
    assert "model files are missing" in friendly_error("ERROR: flux2 vae: none of [x] found in [y]")
    assert friendly_error("ERROR: the moon is blue").startswith("Something went wrong: the moon is blue")
    assert stage_text("@stage denoise") == "Drawing..." and stage_text("plain line") is None


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture()
def settings(tmp_path, qapp):
    from PySide6.QtCore import QSettings

    return QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)


def test_start_screen_lists_projects(qapp, tmp_path, settings):
    from ggs.style.package import init_project
    from ggs.ui.app import MainWindow
    from ggs.ui.start import known_projects, remember

    init_project(tmp_path / "a", "Alpha")
    remember(settings, tmp_path / "a")
    remember(settings, tmp_path / "missing")  # gone folders are dropped from the list
    assert known_projects(settings) == [tmp_path / "a"]
    w = MainWindow(settings=settings)
    assert w.stack.currentWidget() is w.start and w.start.grid.count() == 2  # "+ New" and Alpha
    w.close()


def test_project_window_tree_and_workspace(qapp, tmp_path, settings):
    from PySide6.QtCore import Qt

    from ggs import items as I
    from ggs.style.package import init_project
    from ggs.ui.app import MainWindow
    from ggs.ui.workspace import ItemWorkspace

    init_project(tmp_path / "g", "Game", base_model="sdxl")
    I.create(tmp_path / "g", "sprite", "Potion", "a red potion")
    I.generate(tmp_path / "g", "Potion", action="new", model="stub", enforce_vram=False)
    w = MainWindow(tmp_path / "g", settings=settings)
    v = w.view
    assert v.model.currentData() == "sdxl"
    (it,) = v.tree.findItems("Potion", Qt.MatchFlag.MatchRecursive)
    v.tree.setCurrentItem(it)
    ws = v.stack.currentWidget()
    assert isinstance(ws, ItemWorkspace)
    d = ws.draw
    assert d.change.isVisibleTo(ws) and d.new.text() == "Draw new" and d.regen.isEnabled()
    assert d.where.text() == "1 of 1" and not d.prev.isEnabled() and d.stage.pix is not None
    assert d.input.toPlainText() == "a red potion"  # the prompt stays in the box (typos stay visible)
    assert not hasattr(ws, "chat_lay") and not hasattr(d, "vars")  # no chat, no variations
    assert not ws.adv.isVisible()  # Advanced is closed by default
    v.model.setCurrentIndex(v.model.findData("flux2-klein-4b"))
    assert "flux2-klein-4b" in (tmp_path / "g" / ".ggs" / "project.yaml").read_text(encoding="utf-8")
    w.close()


def test_wizard_tiles(qapp):
    from ggs.ui.wizard import NewItemDialog

    d = NewItemDialog(preselect="background")
    assert d.kind_id() == "background" and not d.ok.isEnabled()
    assert len(d.sizes.buttons()) == 1  # backgrounds have no size choice
    d.kinds.buttons()[0].setChecked(True)
    assert d.kind_id() == "sprite" and d.size() == "medium" and len(d.sizes.buttons()) == 4
    got = []
    d.chosen.connect(lambda *a: got.append(a))
    d.name.setText("Potion")
    d.describe.setPlainText("a red potion")
    assert d.ok.isEnabled()
    d._accept()
    assert got == [("sprite", "Potion", "medium", "a red potion")]


def test_job_queue_runs_cli_and_reports_stages(qapp, tmp_path):
    from PySide6.QtCore import QEventLoop, QTimer

    from ggs.style.package import init_project
    from ggs.ui.jobs import Job, JobQueue

    init_project(tmp_path / "p", "P")
    q = JobQueue()
    stages, results = [], []
    q.stage.connect(stages.append)
    loop = QEventLoop()
    job = Job("gen", [C.asset_new(tmp_path / "p", "sprite", "Gem", "a gem"),
                      C.asset_gen(tmp_path / "p", "Gem", "new", model="stub")], exit_ok=(0, 2))
    job.done = lambda ok: (results.append(ok), loop.quit())
    q.submit(job)
    QTimer.singleShot(120000, loop.quit)
    loop.exec()
    assert results == [True] and "Drawing..." in stages
    rec = json.loads((tmp_path / "p" / "items" / "sprite" / "gem" / "item.json").read_text(encoding="utf-8"))
    assert rec["current"] == 1


def test_character_workspace_tabs_open_after_accept(qapp, tmp_path, settings):
    from ggs import items as I
    from ggs.style.package import init_project
    from ggs.ui.app import MainWindow

    init_project(tmp_path / "g", "Game")
    p = tmp_path / "g"
    I.create(p, "character", "Hero", "a hero")
    I.generate(p, "Hero", action="new", model="stub", enforce_vram=False)
    w = MainWindow(p, settings=settings)
    w.view._open_item(p / "items" / "character" / "hero")
    ws = w.view.cache[str(p / "items" / "character" / "hero")]
    assert ws.accept_btn.isVisibleTo(ws) and not ws.tab_group.button(1).isEnabled()
    I.accept(p, "Hero")
    ws.reload()
    assert ws.kept.isVisibleTo(ws) and not ws.accept_btn.isVisibleTo(ws) and ws.tab_group.button(2).isEnabled()
    ws.show_tab(1)
    assert len(ws.sliced.group.buttons()) >= 3 and not ws.sliced.export.isEnabled()  # nothing sliced yet
    ws.show_tab(2)
    assert ws.animate.cutout.need.isVisibleTo(ws)  # cut-out needs the sliced version first
    ws.animate.methods.button(1).click()
    ws.animate.drawn.list.list.setCurrentRow(1)  # first library animation: its poses are shown
    assert ws.animate.drawn.poses.pixmap() is not None
    w.close()


def test_spelling_marks_typos():
    pytest.importorskip("spellchecker")
    from ggs.ui.spell import suggestions, unknown_words

    assert unknown_words("a little clawn boy with exagarated eyes") == {"clawn", "exagarated"}
    assert unknown_words("a chibi knight sprite, GGSINK style") == set()
    assert suggestions("clawn")[0] == "clown"


def test_progress_estimates_from_stage_times(tmp_path, monkeypatch):
    import time

    from ggs.pipeline import progress as P

    monkeypatch.setattr(P, "_times_file", lambda: tmp_path / "times.json")
    (tmp_path / "times.json").write_text(json.dumps({"k:encode": 1.0, "k:denoise": 0.5}), encoding="utf-8")
    t = P._Ticker("k", ["encode", "denoise"])
    t.stop.set()
    t.enter("encode")
    t.leave("encode")
    t.enter("denoise")
    time.sleep(0.2)
    first = t.percent()
    time.sleep(0.1)
    assert 10 < first <= t.percent() < 99
    t.finish()
    saved = json.loads((tmp_path / "times.json").read_text(encoding="utf-8"))
    assert saved["k:denoise"] != 0.5 and "*:encode" in saved  # learnt from this run
