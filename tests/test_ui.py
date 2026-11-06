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
    assert d.change.isVisibleTo(ws) and d.new.text() == "Generate" and d.regen.isEnabled()
    assert d.where.text() == "1 of 1" and not d.prev.isEnabled() and d.stage.pix is not None
    assert d.input.toPlainText() == "a red potion"  # the prompt stays in the box
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
    assert ws.sliced.layout_box.count() == 5 and not ws.sliced.export.isEnabled()  # nothing sliced yet
    ws.show_tab(2)
    assert ws.animate.warp.need.isVisibleTo(ws)  # bending and cut-out need the sliced version first
    ws.animate.methods.button(1).click()
    assert ws.animate.cutout.need.isVisibleTo(ws)
    ws.animate.methods.button(2).click()
    ws.animate.drawn.list.list.setCurrentRow(1)  # first library animation: its poses are shown
    assert ws.animate.drawn.poses.pixmap() is not None
    w.close()


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


def test_markers_glued_to_a_loading_bar_still_count(qapp):
    """A model-loading bar writes to stderr without a newline; a step or progress marker printed meanwhile ends up
    on the same line and must still move the loading placeholder."""
    import sys

    from PySide6.QtCore import QEventLoop, QTimer

    from ggs.ui.jobs import Job, JobQueue

    code = ("import sys; sys.stderr.write('Loading checkpoint shards:  50%|##   | 1/2 [00:01<00:01, 1.2it/s]'); "
            "sys.stderr.flush(); print('@stage matte', flush=True); "
            "sys.stderr.write(chr(13) + ' 60%|###  | 3/5 [00:02<00:01, 1.5it/s]'); sys.stderr.flush(); "
            "print('@progress 42', flush=True)")
    q = JobQueue()
    stages, pcts = [], []
    q.stage.connect(stages.append)
    q.progress.connect(pcts.append)
    loop = QEventLoop()
    job = Job("glued", [[sys.executable, "-c", code]])
    results = []
    job.done = lambda ok: (results.append(ok), loop.quit())
    q.submit(job)
    QTimer.singleShot(30000, loop.quit)
    loop.exec()
    assert results == [True] and "Cutting it out..." in stages and 42 in pcts


def test_long_texts_do_not_widen_the_window(qapp, tmp_path, settings):
    """Every character tab fits a 1280 px wide window (a laptop at 150 % scaling)."""
    from ggs import items as I
    from ggs.style.package import init_project
    from ggs.ui.app import MainWindow

    p = tmp_path / "g"
    init_project(p, "Game")
    I.create(p, "character", "Hero", "a hero")
    I.generate(p, "Hero", action="new", model="stub", enforce_vram=False)
    I.accept(p, "Hero")
    w = MainWindow(p, settings=settings)
    w.view._open_item(p / "items" / "character" / "hero")
    hint = w.minimumSizeHint()
    assert hint.width() <= 1250 and hint.height() <= 720, hint
    w.close()


def test_live_editor_plays_and_touches_up(qapp, tmp_path):
    from ggs import items as I
    from ggs.rig.clip import load_clip
    from ggs.rig.library import library_clip
    from ggs.rig.warp import make_design_warp
    from ggs.style.package import init_project
    from ggs.ui.anim_editor import AnimEditor, WarpRenderer

    walk = library_clip("walk")
    if not walk.exists():
        pytest.skip("motion library not built")
    p = tmp_path / "g"
    init_project(p, "G")
    I.create(p, "character", "Hero", "a hero", I.ItemSettings(character=I.CharacterSettings(view="side", pose="idle")))
    I.generate(p, "Hero", action="new", model="stub", enforce_vram=False)
    I.accept(p, "Hero")
    ed = AnimEditor()
    ed.resize(600, 600)
    ed.set_renderer(WarpRenderer(make_design_warp(p, "Hero", "coarse", fit=False)))
    assert ed.modes.checkedId() == 1 and ed.confirm.isVisibleTo(ed)  # joints first, until confirmed
    ed._confirm()
    assert ed.modes.checkedId() == 0
    ed.set_clip(load_clip(walk), "walk", tmp_path / "edits")
    assert ed.lead_n == 0 and ed.clip.frame_count == load_clip(walk).frame_count  # plays from the clip's own start
    for f in (0, 3, ed.lead_n + 4):
        ed.slider.setValue(f)
    assert ed.canvas.image is not None and not ed.canvas.image.isNull() and ed.canvas.handles
    key = next(k for k in ed.canvas.handles if k.startswith("hand"))
    pos = ed.canvas.handles[key]
    from PySide6.QtCore import QPointF

    ed._dragged(key, QPointF(pos.x() + 20, pos.y() - 30))
    assert (tmp_path / "edits" / "walk.json").exists()  # the touched-up clip is kept for export
    ed.modes.button(1).click()
    assert ed.canvas.handles  # joints mode: the design with its joints


def test_paint_dialog_fills_and_saves(qapp, tmp_path):
    from PIL import Image
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QColor

    from ggs.ui.paint import PaintDialog

    f = tmp_path / "p.png"
    Image.new("RGBA", (40, 40), (200, 30, 30, 255)).save(f)
    dlg = PaintDialog(f)
    dlg.canvas.colour = QColor(0, 0, 255)
    dlg.canvas._snapshot()
    dlg.canvas._fill(QPointF(5, 5))
    dlg._save()
    assert Image.open(f).convert("RGBA").getpixel((20, 20)) == (0, 0, 255, 255)


def test_every_icon_the_app_uses_exists(qapp):
    import re
    from pathlib import Path

    from ggs.ui import icons
    from ggs.ui.kinds import KINDS

    have = set(icons.names())
    src = Path(icons.__file__).resolve().parents[1]
    used = set()
    for f in src.glob("*.py"):
        used |= set(re.findall(r'(?:set_icon\([\w.]+, |icon\(|pixmap\(|padded\()"(\w+)"', f.read_text(encoding="utf-8")))
    used |= {k.icon for k in KINDS} | {"app"}
    assert used <= have, used - have
    assert not icons.icon("delete").isNull() and not icons.icon("export", "dark").isNull()


def test_rig_key_changes_when_pieces_are_edited(tmp_path, qapp):
    import os

    from ggs.ui.character_page import rig_key

    (tmp_path / "character.yaml").write_text("rig: humanoid_side_v1\n", encoding="utf-8")
    (tmp_path / "parts.png").write_bytes(b"one")
    before = rig_key(tmp_path / "character.yaml")
    (tmp_path / "parts.png").write_bytes(b"two!")
    st = (tmp_path / "parts.png").stat()
    os.utime(tmp_path / "parts.png", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert rig_key(tmp_path / "character.yaml") != before


def test_a_playing_gif_can_be_deleted(qapp, tmp_path):
    from PIL import Image
    from PySide6.QtWidgets import QLabel

    from ggs.ui.widgets import movie_from_file

    gif = tmp_path / "preview.gif"
    frames = [Image.new("RGB", (16, 16), c) for c in ("red", "blue")]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=50, loop=0)
    label = QLabel()
    movie = movie_from_file(gif)
    label.setMovie(movie)
    movie.start()
    qapp.processEvents()
    assert movie.frameCount() == 2
    gif.unlink()  # Windows refuses this while a file is open
    assert not gif.exists()
