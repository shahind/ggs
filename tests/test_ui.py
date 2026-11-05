"""Desktop app (M9): command builders (pure Python), and an offscreen smoke test of the window and job queue."""

import json
import os
import sys

import pytest

from ggs.ui import commands as C


def test_slug_and_paths(tmp_path):
    assert C.slug("A Red Health-Potion!!") == "a_red_health_potion"
    assert C.slug("???") == "asset"
    p = C.new_asset_path(tmp_path, "sprite", "a lantern")
    assert p.parent == tmp_path / "assets" / "sprite" and p.name.startswith("a_lantern_") and p.suffix == ".png"


def test_command_builders(tmp_path):
    g = C.generate("tile", "cobblestone", tmp_path / "t.png", project=tmp_path, style="ink", style_weight=0.8,
                   count=3, seed=7, size="512x512")
    assert g[:3] == [sys.executable, "-m", "ggs"] and g[3:5] == ["generate", "--class"]
    i = g.index
    assert g[i("--class") + 1] == "tile" and g[i("--count") + 1] == "3" and g[i("--style") + 1] == "ink"
    assert "--keep-failed" in g and "--model" not in g
    m = C.character_master("a knight", tmp_path / "k", "knight", body="chibi", stature=0.8, refine=False)
    assert m[m.index("--body") + 1] == "chibi" and "--no-refine" in m and m[m.index("--stature") + 1] == "0.8"
    e = C.export(tmp_path / "c.json", tmp_path / "ch.yaml", "godot", tmp_path / "o", items=[tmp_path / "sword"])
    assert e[e.index("--item") + 1].endswith("sword") and e[e.index("--format") + 1] == "godot"
    d = C.dataset_import(tmp_path, tmp_path / "ds", rights=False)
    assert "--i-have-the-rights" not in d
    assert "--i-have-the-rights" in C.dataset_import(tmp_path, tmp_path / "ds", rights=True)


def test_read_results(tmp_path):
    out = tmp_path / "a.png"
    (tmp_path / "a.json").write_text(json.dumps({"assets": [
        {"file": "a_01.png", "passed": True, "failures": []},
        {"file": "a_02_QA_FAILED.png", "passed": False, "failures": ["halo: 30%"]},
        {"file": None, "passed": False, "failures": ["empty"]}]}), encoding="utf-8")
    rows = C.read_generate_results(out)
    assert [r.passed for r in rows] == [True, False] and rows[1].detail == "halo: 30%"
    seed = tmp_path / "run" / "seed_000" / "character"
    seed.mkdir(parents=True)
    (seed / "validation.json").write_text(json.dumps({"passed": False, "score": 0.8, "checks": [
        {"name": "reassembly_iou", "passed": False, "detail": "IoU 0.8"}]}), encoding="utf-8")
    (r,) = C.read_character_results(tmp_path / "run")
    assert not r["passed"] and r["failures"] == ["reassembly_iou: IoU 0.8"]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_window_pages_and_project(qapp, tmp_path):
    from ggs.style.package import init_project
    from ggs.ui.app import PAGES, MainWindow

    init_project(tmp_path / "game", "Game", base_model="sdxl")
    w = MainWindow(tmp_path / "game")
    assert [w.nav.item(i).text() for i in range(w.nav.count())] == [n for n, _ in PAGES]
    assert w.state.project == tmp_path / "game" and w.state.base_model() == "sdxl"
    gen = w.pages["Generate"]
    kinds = [gen.kind.itemData(i) for i in range(gen.kind.count())]
    assert {"sprite", "tile", "background", "ui_panel", "vfx", "held_item"} <= set(kinds)
    assert "sdxl" in gen.model.itemText(0)  # the project's base model is the default
    anim = w.pages["Animate"]
    assert anim.upper.count() >= 4  # (none) + the keyframe clips in clips/
    styles = w.pages["Styles"]
    assert not styles.import_btn.isEnabled()  # the rights checkbox gates the import
    styles.rights.setChecked(True)
    assert styles.import_btn.isEnabled()
    w.close()


def test_job_queue_runs_cli_commands(qapp, tmp_path):
    from PySide6.QtCore import QEventLoop, QTimer

    from ggs.ui.jobs import Job, JobQueue

    q = JobQueue()
    lines, results = [], []
    q.line.connect(lines.append)
    loop = QEventLoop()
    done = lambda ok: (results.append(ok), loop.quit())
    q.submit(Job("init", [C.project_init(tmp_path / "p", "P")], done=done))
    QTimer.singleShot(60000, loop.quit)
    loop.exec()
    assert results == [True]
    assert (tmp_path / "p" / ".ggs" / "project.yaml").exists()
    assert any(ln.startswith("$ ") for ln in lines) and any("created" in ln for ln in lines)
