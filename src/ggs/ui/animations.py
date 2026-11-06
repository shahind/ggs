"""Animations section: the project's own motions, the motion library (starter set made with Kimodo + the CMU and
100STYLE catalogs, searchable), "Describe a motion" (text to animation), and export (the classic Animate page).

Everything that changes files runs through the ``ggs`` command line (jobs); previews are drawn in-process (fast,
CPU only) and cached.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QMovie
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ggs.ui import commands as C
from ggs.ui.icons import set_icon
from ggs.ui.jobs import Job, JobQueue

ROLE = 256
STARTER = "starter"
WROTE = re.compile(r"^wrote (.+\.json): ")


def _project_clips(project: Path) -> list[Path]:
    d = C.clips_dir(project)
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.json") if not p.name.endswith(".motion.json"))


def _clip_preview(path: Path) -> Path:
    """GIF next to the clip (made by ``ggs motion make``), else one drawn now into the cache."""
    gif = path.with_suffix(".gif")
    if gif.is_file():
        return gif
    from ggs.paths import repo_root, shared_dir
    from ggs.rig import load_rig_spec
    from ggs.rig.clip import load_clip
    from ggs.rig.preview import render_preview

    clip = load_clip(path)
    out = shared_dir() / "cache" / "previews" / "project" / f"{abs(hash(str(path)))}_{int(path.stat().st_mtime)}.gif"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        render_preview(load_rig_spec(repo_root() / "rigs" / f"{clip.rig}.yaml"), clip, out, cell=200, max_frames=240)
    return out


def _packs_ready() -> dict[str, bool]:
    try:
        from ggs.pipeline.download import load_packs, pack_ready

        packs = load_packs()
        return {n: pack_ready(packs[n]) for n in ("motion", "motion_text") if n in packs}
    except Exception:  # noqa: BLE001 - a broken manifest must not break the page
        return {"motion": False, "motion_text": False}


class Preview(QFrame):
    """Animated stick-figure preview with a title and details."""

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        self.title = QLabel("")
        self.title.setObjectName("h2")
        self.title.setWordWrap(True)
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(240, 240)
        self.view.setStyleSheet("background: #fafafa; border-radius: 10px;")
        self.info = QLabel()
        self.info.setObjectName("muted")
        self.info.setWordWrap(True)
        self.buttons = QHBoxLayout()
        lay.addWidget(self.title)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.info)
        lay.addLayout(self.buttons)
        self.movie: QMovie | None = None

    def show_gif(self, title: str, gif: Path | None, info: str = "") -> None:
        self.title.setText(title)
        self.info.setText(info)
        if self.movie is not None:
            self.movie.stop()
        self.movie = None
        self.view.clear()
        if gif and Path(gif).is_file():
            self.movie = QMovie(str(gif))
            self.view.setMovie(self.movie)
            self.movie.start()
        else:
            self.view.setText("no preview")


class LibraryTab(QWidget):
    """Search the starter set and the catalogs; add picks to the project."""

    def __init__(self, page: AnimationsPage):
        super().__init__()
        self.page = page
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search")
        self.search.textChanged.connect(self._later)
        self.category = QComboBox()
        self.category.currentIndexChanged.connect(self._fill)
        row.addWidget(self.search, 1)
        row.addWidget(self.category)
        left.addLayout(row)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.currentItemChanged.connect(self._pick)
        self.list.itemDoubleClicked.connect(lambda _it: self._add())
        left.addWidget(self.list, 1)
        self.count = QLabel()
        self.count.setObjectName("muted")
        left.addWidget(self.count)
        lay.addLayout(left, 3)
        self.preview = Preview()
        add = QPushButton("Add to Project")
        add.setObjectName("primary")
        set_icon(add, "new")
        add.clicked.connect(self._add)
        self.take = QPushButton("New Take...")
        set_icon(self.take, "generate")
        self.take.setToolTip("Make another version of this motion with the motion model")
        self.take.clicked.connect(self._new_take)
        self.preview.buttons.addWidget(add)
        self.preview.buttons.addWidget(self.take)
        lay.addWidget(self.preview, 2)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._fill)
        self.rows: list[dict] = []
        self._load()

    def _load(self) -> None:
        from ggs.rig import catalog
        from ggs.rig.library import library_clip, load_library

        rows = []
        try:
            for m in load_library().motions:
                f = library_clip(m.name)
                if f.is_file():
                    clip = json.loads(f.read_text(encoding="utf-8"))
                    rows.append({"source": STARTER, "ref": m.name, "name": m.label, "category": m.category,
                                 "seconds": clip["frame_count"] / clip["fps"], "text": f"{m.label} {m.prompt} {m.category}".lower(),
                                 "prompt": m.prompt, "credit": "Made with NVIDIA Kimodo", "loop": m.loop})
        except ValueError:
            pass
        try:
            for idx, m in catalog.all_motions():
                rows.append({"source": idx.source, "ref": f"{idx.source}/{m.id}", "name": m.name,
                             "category": m.category, "seconds": m.seconds, "loop": m.loop,
                             "text": " ".join([m.name, m.category, *m.tags, m.id]).lower(),
                             "credit": f"{idx.credit} ({idx.license})", "note": m.note})
        except (ValueError, OSError):
            pass
        self.rows = rows
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItem("All kinds", None)
        for c in sorted({r["category"] for r in rows}):
            self.category.addItem(c, c)
        self.category.blockSignals(False)
        self._fill()

    def _later(self) -> None:
        self.timer.start()

    def _fill(self) -> None:
        terms = self.search.text().lower().split()
        cat = self.category.currentData()
        hits = [r for r in self.rows if (not cat or r["category"] == cat) and all(t in r["text"] for t in terms)]
        self.list.clear()
        for r in hits[:2000]:
            it = QListWidgetItem(f"{r['name']}    {r['seconds']:.1f} s" + ("  (loop)" if r.get("loop") else ""))
            it.setData(ROLE, r)
            self.list.addItem(it)
        self.count.setText(f"{len(hits)} motions" + (" (first 2000 shown)" if len(hits) > 2000 else "")
                           if self.rows else "The motion library is not installed.")

    def _pick(self, it: QListWidgetItem | None, _prev=None) -> None:
        if it is None:
            return
        r = it.data(ROLE)
        self.take.setEnabled(r["source"] == STARTER)
        try:
            if r["source"] == STARTER:
                from ggs.rig.library import library_clip

                gif = library_clip(r["ref"]).with_suffix(".gif")
            else:
                from ggs.rig import catalog

                gif = catalog.preview(r["ref"], cell=200)
        except Exception:  # noqa: BLE001 - a broken clip shows "no preview"
            gif = None
        info = f"{r['category']} - {r['seconds']:.1f} s. {r.get('note', '')}".strip()
        self.preview.show_gif(r["name"], gif, info)

    def _add(self) -> None:
        if not self.page.project:
            return
        refs = [it.data(ROLE)["ref"] for it in self.list.selectedItems()]
        if not refs:
            return
        self.page.queue.submit(Job(f"add {len(refs)} motion(s)", [C.motion_add(refs, self.page.project)],
                                   done=lambda ok: self.page.refresh_project(ok)))

    def _new_take(self) -> None:
        it = self.list.currentItem()
        if it is not None and it.data(ROLE)["source"] == STARTER:
            r = it.data(ROLE)
            self.page.describe.prefill(r["prompt"], r["seconds"], r.get("loop", False))
            self.page.tabs.setCurrentWidget(self.page.describe)


class DescribeTab(QWidget):
    """Text to animation: prompt -> ``ggs motion make`` -> the project's clips."""

    def __init__(self, page: AnimationsPage):
        super().__init__()
        self.page = page
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        h = QLabel("Describe a motion")
        h.setObjectName("h2")
        left.addWidget(h)
        tip = QLabel("Say what one person does, in plain words. Several sentences play one after another, "
                     "e.g. \"A person walks forward. The person jumps over a gap.\"")
        tip.setObjectName("muted")
        tip.setWordWrap(True)
        left.addWidget(tip)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("A person swings a sword and steps forward.")
        self.prompt.setMaximumHeight(110)
        left.addWidget(self.prompt)
        row = QHBoxLayout()
        row.addWidget(QLabel("Length"))
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(1.0, 10.0)
        self.seconds.setSingleStep(0.5)
        self.seconds.setValue(3.0)
        self.seconds.setSuffix(" s")
        row.addWidget(self.seconds)
        row.addWidget(QLabel("Versions"))
        self.count = QSpinBox()
        self.count.setRange(1, 6)
        self.count.setValue(2)
        row.addWidget(self.count)
        row.addStretch(1)
        left.addLayout(row)
        self.loop = QCheckBox("Make it loop (walks, runs, idles)")
        self.in_place = QCheckBox("Stay in place")
        left.addWidget(self.loop)
        left.addWidget(self.in_place)
        btns = QHBoxLayout()
        self.go = QPushButton("Generate")
        self.go.setObjectName("primary")
        set_icon(self.go, "generate")
        self.go.clicked.connect(self._run)
        self.models = QPushButton("Download Models...")
        set_icon(self.models, "download")
        self.models.clicked.connect(self._setup)
        btns.addWidget(self.go)
        btns.addWidget(self.models)
        btns.addStretch(1)
        left.addLayout(btns)
        self.note = QLabel()
        self.note.setObjectName("muted")
        self.note.setWordWrap(True)
        left.addWidget(self.note)
        self.results = QListWidget()
        self.results.currentItemChanged.connect(self._pick)
        left.addWidget(self.results, 1)
        lay.addLayout(left, 3)
        self.preview = Preview()
        self.preview.show_gif("Your new animations appear here", None, "")
        lay.addWidget(self.preview, 2)
        self.seed = 0
        self.check()

    def check(self) -> None:
        ready = _packs_ready()
        self.models.setVisible(not all(ready.values()))
        if not ready.get("motion"):
            self.note.setText("Needs the motion model (1.1 GB download).")
        elif not ready.get("motion_text"):
            self.note.setText("Only the starter set's own descriptions work until the text reader (15 GB) is "
                              "downloaded; use \"Make a new take\" in the Library.")
        else:
            self.note.setText("")

    def prefill(self, prompt: str, seconds: float, loop: bool) -> None:
        from ggs.rig.library import load_library

        self.prompt.setPlainText(prompt)
        lib = {m.prompt: m for m in load_library().motions}
        self.seconds.setValue(lib[prompt].seconds if prompt in lib else seconds)
        self.loop.setChecked(loop)
        self.seed += 1

    def _setup(self) -> None:
        from ggs.ui.setup import ModelSetupDialog

        ModelSetupDialog(self.page.queue, self).exec()
        self.check()

    def _run(self) -> None:
        text = " ".join(self.prompt.toPlainText().split())
        if not text or not self.page.project:
            return
        out = C.clips_dir(self.page.project) / f"{C.slug(text, 32)}_{C.stamp()}.json"
        count = self.count.value()
        job = Job(f"motion: {text[:40]}", [C.motion_make(text, out, seconds=self.seconds.value(), seed=self.seed,
                                                         count=count, loop=self.loop.isChecked(),
                                                         in_place=self.in_place.isChecked())])
        self.seed += count
        self.go.setEnabled(False)

        def done(ok: bool) -> None:
            self.go.setEnabled(True)
            if not ok:
                from ggs.ui.errors import show_error

                self.note.setText("")
                show_error(self, "\n".join(job.output), "The animation was not made")
                return
            self.note.setText("")
            self.page.refresh_project(True)
            made = [Path(m.group(1)) for m in (WROTE.match(line) for line in job.output) if m]
            self.results.clear()
            for p in made:
                it = QListWidgetItem(p.stem)
                it.setData(ROLE, str(p))
                self.results.addItem(it)
            if self.results.count():
                self.results.setCurrentRow(0)

        job.done = done
        self.note.setText("Making the animation... (the first time, reading the text takes a minute)")
        self.page.queue.submit(job)

    def _pick(self, it: QListWidgetItem | None, _prev=None) -> None:
        if it is None:
            return
        p = Path(it.data(ROLE))
        self.preview.show_gif(p.stem, p.with_suffix(".gif"), "Saved in this project's animations.")


class ProjectTab(QWidget):
    """The animations this project has (clips folder)."""

    def __init__(self, page: AnimationsPage):
        super().__init__()
        self.page = page
        lay = QHBoxLayout(self)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._pick)
        lay.addWidget(self.list, 3)
        self.preview = Preview()

        rm = QPushButton("Delete")
        set_icon(rm, "delete")
        rm.clicked.connect(self._remove)
        self.preview.buttons.addWidget(rm)
        hint = QLabel("")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        self.preview.layout().addWidget(hint)
        lay.addWidget(self.preview, 2)

    def refresh(self, select: Path | None = None) -> None:
        self.list.clear()
        clips = _project_clips(self.page.project) if self.page.project else []
        for p in clips:
            label = p.stem
            side = p.with_suffix(".motion.json")
            if side.is_file():
                try:
                    meta = json.loads(side.read_text(encoding="utf-8"))
                    label = meta.get("name") or meta.get("prompt") or label
                except (OSError, ValueError):
                    pass
            it = QListWidgetItem(label)
            it.setData(ROLE, str(p))
            self.list.addItem(it)
            if select and Path(select) == p:
                self.list.setCurrentItem(it)
        if not clips:
            self.preview.show_gif("No animations yet", None,
                                  "Add some from the Library, or describe one in words.")

    def _pick(self, it: QListWidgetItem | None, _prev=None) -> None:
        if it is None:
            return
        p = Path(it.data(ROLE))
        info = ""
        side = p.with_suffix(".motion.json")
        if side.is_file():
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
                info = meta.get("prompt") or ""
            except (OSError, ValueError):
                pass
        try:
            gif = _clip_preview(p)
        except Exception:  # noqa: BLE001
            gif = None
        self.preview.show_gif(it.text(), gif, info)

    def _remove(self) -> None:
        it = self.list.currentItem()
        if it is None:
            return
        from ggs.rig.library import delete_clip
        from ggs.ui.errors import confirm_delete, show_error

        if not confirm_delete(self, f"the animation '{it.text()}'", "Characters already exported keep it."):
            return
        self.preview.show_gif(it.text(), None)  # the preview GIF is open while it plays
        try:
            delete_clip(it.data(ROLE))
        except OSError as exc:
            show_error(self, str(exc), "Not deleted")
        self.page.refresh_project(True)


class AnimationsPage(QWidget):
    changed = Signal()  # the project's animations changed (the project tree lists them)

    def __init__(self, queue: JobQueue, project: Path, state):
        super().__init__()
        self.queue, self.project = queue, Path(project)
        lay = QVBoxLayout(self)
        h = QLabel("Animations")
        h.setObjectName("h1")
        lay.addWidget(h)
        self.tabs = QTabWidget()
        self.mine = ProjectTab(self)
        self.library = LibraryTab(self)
        self.describe = DescribeTab(self)
        self.tabs.addTab(self.mine, "In this project")
        self.tabs.addTab(self.library, "Library")
        self.tabs.addTab(self.describe, "Describe a motion")
        self.tabs.currentChanged.connect(self._tab)
        lay.addWidget(self.tabs, 1)
        self.mine.refresh()

    def _tab(self, i: int) -> None:
        w = self.tabs.widget(i)
        if w is self.describe:
            self.describe.check()

    def refresh_project(self, ok: bool = True) -> None:
        if ok:
            self.mine.refresh()
            self.changed.emit()

    def show_clip(self, path: Path) -> None:
        self.tabs.setCurrentWidget(self.mine)
        self.mine.refresh(select=path)
