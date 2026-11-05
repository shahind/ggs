"""The character page's Sliced version and Animate tabs (after the design is accepted).

Sliced version: pick a template (pictures of the part layouts), make it, see the parts, export them on their own.
Animate: two ways.
  Cut-out    the sliced version plays the animations: a list on one side, the character moving on the other,
             tick several and export them in one go (engine package or one file per animation).
  Drawn      the drawing model draws the design in every pose of one animation: pick it, see the stick figures
             it will be given, choose how many frames, then wait for the drawing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QMovie, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ggs.ui import commands as C
from ggs.ui.jobs import Job, run_on_stage
from ggs.ui.widgets import ImageStage

ROW = 256
ENGINE_FORMATS = [("Godot", "godot"), ("Unity", "unity"), ("Defold", "defold"), ("Parts + JSON", "rig")]
ANIM_FORMATS = [*ENGINE_FORMATS, ("GIF (one per animation)", "gif"), ("Animated PNG", "apng"),
                ("WebP", "webp"), ("Sprite sheets", "sheet")]
WARP_FORMATS = [("Sprite sheets + JSON (Godot, Unity, Defold)", "sheet"), ("Animated PNG", "apng"), ("WebP", "webp"),
                ("GIF (one per animation)", "gif")]
ADDED = re.compile(r"^added (.+\.json)\s*$")
METHODS = [("Bend the drawing", ("The whole drawing bends with the skeleton: smooth elbows and knees, nothing cut "
                                 "apart. Best for simple characters; exported as pictures or sprite sheets.")),
           ("Cut-out pieces", ("The sliced pieces move like a puppet: any animation from the library, instantly, "
                               "and they stay editable in your engine.")),
           ("Drawn frames", ("The drawing model draws the character in every frame of one animation: clothes and hair "
                            "move naturally; takes a few minutes per animation."))]


def _h(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("h2")
    return lab


def _muted(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("muted")
    lab.setWordWrap(True)
    return lab


def _out_name(text: str) -> str:
    return C.slug(text, 32)


# ====================================================================== animation list


class MotionList(QWidget):
    """The project's animations first, then the library (search it). Only names, lengths and kinds are shown."""

    current_changed = Signal(object)  # the row dict (or None)
    removed = Signal()  # one of the project's animations was deleted

    def __init__(self, project: Path, multi: bool = False):
        super().__init__()
        self.project = Path(project)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search animations: walk, punch, angry, dance...")
        lay.addWidget(self.search)
        self.list = QListWidget()
        if multi:
            self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.currentItemChanged.connect(lambda it, _p: self.current_changed.emit(it.data(ROW) if it else None))
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.list, 1)
        self.multi = multi
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._fill)
        self.search.textChanged.connect(lambda: self.timer.start())
        self.library: list[dict] = []
        self.reload()

    def reload(self) -> None:
        from ggs.rig.library import clip_label, library_clip, load_library
        from ggs.ui.animations import _project_clips

        self.own = []
        for p in _project_clips(self.project):
            try:
                clip = json.loads(p.read_text(encoding="utf-8"))
                secs = clip["frame_count"] / clip["fps"]
            except (OSError, ValueError, KeyError):
                continue
            self.own.append({"kind": "project", "path": str(p), "name": clip_label(p), "seconds": secs,
                             "text": f"{clip_label(p)} {p.stem}".lower(), "rig": clip.get("rig", "")})
        if not self.library:
            rows = []
            try:
                for m in load_library().motions:
                    f = library_clip(m.name)
                    if f.is_file():
                        rows.append({"kind": "starter", "ref": m.name, "name": m.label, "seconds": m.seconds,
                                     "text": f"{m.label} {m.prompt} {m.category}".lower(), "category": m.category})
            except ValueError:
                pass
            try:
                from ggs.rig import catalog

                for idx, m in catalog.all_motions():
                    rows.append({"kind": "catalog", "ref": f"{idx.source}/{m.id}", "name": m.name,
                                 "seconds": m.seconds, "category": m.category,
                                 "text": " ".join([m.name, m.category, *m.tags]).lower()})
            except (ValueError, OSError):
                pass
            self.library = rows
        self._fill()

    def _header(self, text: str) -> None:
        it = QListWidgetItem(text)
        it.setFlags(Qt.ItemFlag.NoItemFlags)
        f = it.font()
        f.setBold(True)
        it.setFont(f)
        self.list.addItem(it)

    def _row(self, r: dict) -> None:
        it = QListWidgetItem(f"{r['name']}    {r['seconds']:.1f} s")
        it.setData(ROW, r)
        if self.multi:
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked)
        self.list.addItem(it)

    def _fill(self) -> None:
        terms = self.search.text().lower().split()
        keep = [it.data(ROW) for it in self.checked_items()] if self.multi else []
        self.list.clear()
        own = [r for r in self.own if all(t in r["text"] for t in terms)]
        lib = [r for r in self.library if all(t in r["text"] for t in terms)]
        if own:
            self._header("In this project")
            for r in own:
                self._row(r)
        self._header("Library" + (f" ({len(lib)})" if terms else ""))
        for r in lib[:600]:
            self._row(r)
        if len(lib) > 600:
            self._header("... type more words to see the rest")
        if keep:
            ids = {(k["kind"], k.get("path") or k.get("ref")) for k in keep}
            for i in range(self.list.count()):
                it = self.list.item(i)
                r = it.data(ROW)
                if r and (r["kind"], r.get("path") or r.get("ref")) in ids:
                    it.setCheckState(Qt.CheckState.Checked)

    def _menu(self, pos) -> None:
        it = self.list.itemAt(pos)
        r = it.data(ROW) if it else None
        if not r or r["kind"] != "project":
            return
        from PySide6.QtWidgets import QMenu

        from ggs.ui.errors import trash_icon

        menu = QMenu(self)
        act = menu.addAction(trash_icon(self), f"Delete '{r['name']}' from this project...")
        if menu.exec(self.list.mapToGlobal(pos)) is act:
            self.delete_row(r)

    def delete_row(self, r: dict) -> None:
        from ggs.rig.library import delete_clip
        from ggs.ui.errors import confirm_delete, show_error

        if not confirm_delete(self, f"the animation '{r['name']}'", "Characters already exported keep it."):
            return
        try:
            delete_clip(r["path"])
        except OSError as exc:
            show_error(self, str(exc), "Not deleted")
        self.reload()
        self.removed.emit()

    def checked_items(self) -> list[QListWidgetItem]:
        return [self.list.item(i) for i in range(self.list.count())
                if self.list.item(i).data(ROW) and self.list.item(i).checkState() == Qt.CheckState.Checked]

    def current(self) -> dict | None:
        it = self.list.currentItem()
        return it.data(ROW) if it else None


def clip_for_row(r: dict) -> Path:
    """A clip file for a list row (library motions are read into the cache, for previews)."""
    from ggs.paths import shared_dir

    if r["kind"] == "project":
        return Path(r["path"])
    if r["kind"] == "starter":
        from ggs.rig.library import library_clip

        return library_clip(r["ref"])
    from ggs.rig import catalog

    out = shared_dir() / "cache" / "clips" / (r["ref"].replace("/", "_") + ".json")
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        catalog.read_clip(r["ref"]).save(out)
    return out


def with_project_clips(page, rows: list[dict], then) -> None:
    """Make sure every row's motion is in the project (library ones are added first), then call
    ``then(list of project clip paths)``."""
    have = [Path(r["path"]) for r in rows if r["kind"] == "project"]
    refs = [r["ref"] for r in rows if r["kind"] != "project"]
    if not refs:
        then(have)
        return
    job = Job(f"add {len(refs)} animation(s)", [C.motion_add(refs, page.project)])

    def done(ok: bool) -> None:
        if ok:
            added = [Path(m.group(1)) for m in (ADDED.match(ln) for ln in job.output) if m]
            then(have + added)

    job.done = done
    run_on_stage(page.queue, job, page.stage)
    page.stage.set_step("Adding the animations to the project...")


def preview_gif(character_yaml: Path, clip_path: Path, height: int = 230, seconds: float = 6.0) -> Path:
    """The cut-out character playing a clip (first ``seconds``), as a GIF in the cache."""
    import hashlib

    from ggs.export.animated import save_gif
    from ggs.paths import rigs_dir, shared_dir
    from ggs.rig import load_rig_spec
    from ggs.rig.character import load_character
    from ggs.rig.clip import load_clip

    key = hashlib.sha1(f"{character_yaml}|{Path(character_yaml).stat().st_mtime}|{clip_path}|{height}".encode()
                       ).hexdigest()[:16]
    out = shared_dir() / "cache" / "previews" / "characters" / f"{key}.gif"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    warp = Path(character_yaml).name == "skin.json"
    if warp:
        from ggs.rig.skin import load_skin

        skin = load_skin(character_yaml)
    else:
        ch = load_character(character_yaml)
        spec = load_rig_spec(rigs_dir() / f"{ch.spec.rig}.yaml")
    clip = load_clip(clip_path)
    n = min(clip.frame_count, max(1, int(seconds * clip.fps)))
    if n < clip.frame_count:
        cut = {k: v[:n] for k, v in clip.angles.items()}
        clip = clip.model_copy(update={
            "frame_count": n, "angles": cut, "local_angles": {k: v[:n] for k, v in clip.local_angles.items()},
            "root": clip.root.model_copy(update={"x": clip.root.x[:n], "y": clip.root.y[:n]}),
            "swaps": {k: v[:n] for k, v in clip.swaps.items()}})
    frames, _ = (skin if warp else ch.puppet(spec)).render_clip(clip, height)
    save_gif(frames, out, clip.fps, (255, 255, 255), True)
    return out


class Movie(QLabel):
    """Plays a GIF centred in a white card; shows text when there is none."""

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(260, 260)
        self.setStyleSheet("background: white; border-radius: 12px; color: #6a6d80;")
        self.movie: QMovie | None = None

    def play(self, gif: Path | None, text: str = "") -> None:
        if self.movie is not None:
            self.movie.stop()
            self.movie = None
        if gif is None or not Path(gif).exists():
            self.setMovie(None)
            self.setText(text)
            return
        self.movie = QMovie(str(gif))
        self.setMovie(self.movie)
        self.movie.start()

    def picture(self, path: Path | None, text: str = "") -> None:
        self.play(None, text)
        if path is not None and Path(path).exists():
            pm = QPixmap(str(path)).scaled(self.width() - 16, self.height() - 16, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(pm)


# ====================================================================== Sliced version


class SlicedTab(QWidget):
    def __init__(self, ws):
        super().__init__()
        from ggs.rig.slices import load_templates, template_preview

        self.ws = ws
        self.queue, self.project = ws.queue, ws.project
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(_h("1. Pick a layout"))
        left.addWidget(_muted("The character is drawn again with arms and legs apart, its joints are found, and it "
                              "is cut into these pieces. The same drawing is used to bend it in one piece."))
        self.group = QButtonGroup(self)
        self.templates = load_templates()
        for i, t in enumerate(self.templates):
            b = QPushButton()
            b.setObjectName("tile")
            b.setCheckable(True)
            b.setIcon(QIcon(QPixmap(str(template_preview(t, 200)))))
            b.setIconSize(QSize(120, 90))
            b.setText(f"  {t.label}")
            b.setToolTip(t.about)
            b.setStyleSheet("text-align: left; padding: 6px;")
            self.group.addButton(b, i)
            left.addWidget(b)
        self.group.button(0).setChecked(True)
        self.group.idClicked.connect(self._picked)
        self.about = _muted(self.templates[0].about)
        left.addWidget(self.about)
        self.make = QPushButton("Make sliced version")
        self.make.setObjectName("primary")
        self.make.clicked.connect(self._make)
        left.addWidget(self.make)
        from ggs.ui.errors import trash_icon

        self.delete = QPushButton("Delete sliced version")
        self.delete.setIcon(trash_icon(self))
        self.delete.clicked.connect(self._delete)
        left.addWidget(self.delete)
        left.addStretch(1)
        left.addWidget(_h("Export the sliced version"))
        left.addWidget(_muted("The pieces and the skeleton on their own, for your engine (no animations needed)."))
        row = QHBoxLayout()
        self.fmt = QComboBox()
        for label, f in ENGINE_FORMATS:
            self.fmt.addItem(label, f)
        self.height = QSpinBox()
        self.height.setRange(32, 2048)
        self.height.setValue(256)
        self.height.setSuffix(" px tall")
        row.addWidget(self.fmt, 1)
        row.addWidget(self.height)
        left.addLayout(row)
        self.export = QPushButton("Export")
        self.export.setIcon(QIcon())
        self.export.clicked.connect(self._export)
        left.addWidget(self.export)
        wrap = QWidget()
        wrap.setLayout(left)
        from PySide6.QtWidgets import QScrollArea

        scroll = QScrollArea()  # scrolls on short screens instead of making the window taller than them
        scroll.setWidget(wrap)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(348)
        lay.addWidget(scroll)
        right = QVBoxLayout()
        self.stage = ImageStage(empty="No sliced version yet.\nPick a layout and press Make sliced version.")
        right.addWidget(self.stage, 1)
        self.status = _muted("")
        right.addWidget(self.status)
        lay.addLayout(right, 1)

    def _picked(self, i: int) -> None:
        self.about.setText(self.templates[i].about)

    def refresh(self) -> None:
        from ggs.rig.slices import sliced_character, sliced_dir, sliced_info

        info = sliced_info(self.ws.dir)
        ok = sliced_character(self.ws.dir) is not None
        tid = info.get("template")
        for i, t in enumerate(self.templates):
            if t.id == tid:
                self.group.button(i).setChecked(True)
                self.about.setText(t.about)
        parts = sliced_dir(self.ws.dir) / "parts.png"
        self.stage.set_image(parts if ok and parts.exists() else None)
        stale = ok and info.get("design") != self.ws.rec.accepted
        self.status.setText("" if not ok else ("Made from an earlier design: make it again to match the new one."
                                               if stale else "The pieces of your character. Animate them in the "
                                               "Animate tab, or export them below."))
        if ok and info.get("rough"):
            self.status.setText("A rough cut: the pieces do not rebuild the drawing exactly, so some may show gaps "
                                "when they move. For a character like this, Animate > Bend the drawing usually looks "
                                "better (it keeps the drawing in one piece). Or make it again for other tries.")
            self.status.setStyleSheet("color: #f0a040;")
        else:
            self.status.setStyleSheet("")
        if info and not ok:
            self.status.setText("The last try could not be cut into clean pieces. Press Make sliced version to "
                                "try again, or change the design (simpler clothes cut better).")
        self.make.setText("Make it again" if ok else "Make sliced version")
        self.export.setEnabled(ok)
        self.delete.setVisible(sliced_dir(self.ws.dir).exists())

    def _delete(self) -> None:
        from ggs.items import ItemError, delete_sliced
        from ggs.ui.errors import busy_refusal, confirm_delete, show_error

        if busy_refusal(self, self.queue) or not confirm_delete(
                self, "the sliced version", "Cut-out animation needs it; you can make it again any time."):
            return
        try:
            delete_sliced(self.project, f"character/{self.ws.rec.name}")
        except (ItemError, OSError) as exc:
            show_error(self, str(exc), "Not deleted")
        self.refresh()
        self.ws.changed.emit()

    def _make(self) -> None:
        t = self.templates[self.group.checkedId()]
        job = Job(f"slice {self.ws.rec.name}",
                  [C.character_slice(self.project, f"character/{self.ws.rec.name}", t.id)], exit_ok=(0, 2))
        job.done = lambda ok: self._made(ok, job)
        self.make.setEnabled(False)
        run_on_stage(self.queue, job, self.stage)

    def _made(self, ok: bool, job: Job) -> None:
        self.make.setEnabled(True)
        self.refresh()
        from ggs.rig.slices import sliced_character

        if ok and job.exit_code == 2 and sliced_character(self.ws.dir) is None:
            from ggs.ui.errors import show_error

            show_error(self, "\n".join(job.output[-40:]), "Not sliced yet",
                       "The drawing could not be cut into clean pieces this time. Press Make it again (it tries "
                       "other seeds), or simplify the design: long coats, capes and big props are hard to cut.")
        self.ws.changed.emit()

    def _export(self) -> None:
        from ggs.rig.slices import sliced_character

        ch = sliced_character(self.ws.dir)
        if ch is None:
            return
        fmt = self.fmt.currentData()
        out = C.exports_dir(self.project) / _out_name(self.ws.rec.name) / f"sliced_{fmt}"
        job = Job(f"export sliced {self.ws.rec.name}", [C.export_many([], ch, fmt, out, height=self.height.value())])

        def done(ok: bool) -> None:
            if ok:
                from ggs.ui.errors import show_saved

                show_saved(self, "Exported", out)

        job.done = done
        run_on_stage(self.queue, job, self.stage)


# ====================================================================== Animate


class AnimateTab(QWidget):
    def __init__(self, ws):
        super().__init__()
        self.ws = ws
        self.queue, self.project = ws.queue, ws.project
        lay = QVBoxLayout(self)
        lay.addWidget(_h("How should it move?"))
        row = QHBoxLayout()
        self.methods = QButtonGroup(self)
        for i, (title, text) in enumerate(METHODS):
            b = QPushButton(title)
            b.setObjectName("tile")
            b.setCheckable(True)
            b.setMinimumHeight(40)
            b.setToolTip(text)
            self.methods.addButton(b, i)
            row.addWidget(b, 1)
        self.methods.button(0).setChecked(True)
        self.methods.idClicked.connect(self._method)
        lay.addLayout(row)
        self.about = _muted(METHODS[0][1])
        lay.addWidget(self.about)
        self.stack = QStackedWidget()
        self.warp = CutoutPanel(self, warp=True)
        self.cutout = CutoutPanel(self)
        self.drawn = DrawnPanel(self)
        self.stack.addWidget(self.warp)
        self.stack.addWidget(self.cutout)
        self.stack.addWidget(self.drawn)
        lay.addWidget(self.stack, 1)

    def _method(self, i: int) -> None:
        self.about.setText(METHODS[i][1])
        self.stack.setCurrentIndex(i)
        self.refresh()

    def refresh(self) -> None:
        self.stack.currentWidget().refresh()


class CutoutPanel(QWidget):
    """Animations played by the sliced version: the cut-out puppet, or (``warp``) the drawing bent in one piece."""

    def __init__(self, tab: AnimateTab, warp: bool = False):
        super().__init__()
        self.warp = warp
        self.tab, self.ws = tab, tab.ws
        self.queue, self.project = tab.queue, tab.project
        outer = QVBoxLayout(self)
        self.need = QFrame()
        self.need.setObjectName("panel")
        nl = QVBoxLayout(self.need)
        self.need_text = _muted("")
        nl.addWidget(self.need_text)
        go = QPushButton("Make the sliced version")
        go.setObjectName("primary")
        go.clicked.connect(lambda: self.ws.show_tab(1))
        nl.addWidget(go, 0, Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(self.need)
        self.body = QWidget()
        lay = QHBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        left = QVBoxLayout()
        left.addWidget(_muted("Click an animation to see it; tick the ones to export."))
        self.list = MotionList(self.project, multi=True)
        self.list.current_changed.connect(self._show)
        self.list.removed.connect(self.ws.changed.emit)
        left.addWidget(self.list, 1)
        lay.addLayout(left, 2)
        right = QVBoxLayout()
        self.movie = Movie("Pick an animation on the left")
        right.addWidget(self.movie, 1)
        self.stage = ImageStage()
        self.stage.setVisible(False)
        right.addWidget(self.stage, 1)
        row = QHBoxLayout()
        self.fmt = QComboBox()
        for label, f in (WARP_FORMATS if warp else ANIM_FORMATS):
            self.fmt.addItem(label, f)
        self.height = QSpinBox()
        self.height.setRange(32, 2048)
        self.height.setValue(256)
        self.height.setSuffix(" px tall")
        row.addWidget(self.fmt, 1)
        row.addWidget(self.height)
        right.addLayout(row)
        self.export = QPushButton("Export ticked animations")
        self.export.setObjectName("primary")
        self.export.clicked.connect(self._export)
        right.addWidget(self.export)
        lay.addLayout(right, 3)
        outer.addWidget(self.body, 1)

    def character(self) -> Path | None:
        from ggs.rig.slices import sliced_character, warp_skin

        return warp_skin(self.ws.dir) if self.warp else sliced_character(self.ws.dir)

    def refresh(self) -> None:
        ch = self.character()
        side = False
        if ch is not None:
            if self.warp:
                side = json.loads(ch.read_text(encoding="utf-8")).get("rig") == "humanoid_side_v1"
            else:
                import yaml

                side = (yaml.safe_load(ch.read_text(encoding="utf-8")) or {}).get("rig") == "humanoid_side_v1"
        self.need.setVisible(not side)
        self.body.setVisible(side)
        if ch is None:
            self.need_text.setText(("Bending the drawing needs the sliced version (it finds the joints in a "
                                    "drawing of your character with arms and legs apart). " if self.warp else
                                    "Cut-out animation moves the sliced pieces of your character. ") +
                                   "Make the sliced version first (it takes a few minutes).")
        elif not side:
            self.need_text.setText("This sliced version is front view; the animations are side view. Make a side "
                                   "view sliced version to animate it.")
        else:
            self.list.reload()

    def _show(self, row: dict | None) -> None:
        ch = self.character()
        if not row or ch is None:
            return
        from PySide6.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            gif = preview_gif(ch, clip_for_row(row))
            self.movie.play(gif)
        except Exception as exc:  # noqa: BLE001 - a broken clip shows a message, not a crash
            self.movie.play(None, f"Cannot show this one: {exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _export(self) -> None:
        rows = [it.data(ROW) for it in self.list.checked_items()]
        if not rows and self.list.current():
            rows = [self.list.current()]
        if not rows:
            from ggs.ui.errors import show_message

            show_message(self, "Tick at least one animation in the list.")
            return
        fmt = self.fmt.currentData()
        ch = self.character()
        base = C.exports_dir(self.project) / _out_name(self.ws.rec.name)
        if self.warp:
            base = base / "bent"
        engine = fmt in ("godot", "unity", "defold", "rig")
        self.movie.setVisible(False)
        self.stage.setVisible(True)

        def go(clips: list[Path]) -> None:
            if not clips:
                return
            if engine or len(clips) > 1:
                out = base / (fmt if engine else f"animations_{fmt}")
            else:
                ext = {"gif": ".gif", "webp": ".webp", "apng": ".png", "sheet": "_sheet.png"}[fmt]
                out = base / f"{clips[0].stem}{ext}"
            job = Job(f"export {self.ws.rec.name}", [C.export_many(
                clips, ch, fmt, out, height=self.height.value(), matte="#FFFFFF" if fmt == "gif" else None)])

            def done(ok: bool) -> None:
                self.stage.setVisible(False)
                self.movie.setVisible(True)
                self.list.reload()
                if ok:
                    from ggs.ui.errors import show_saved

                    show_saved(self, "Exported", out, f"Exported {len(clips)} animation(s) to\n{out}")
                    self.ws.changed.emit()

            job.done = done
            run_on_stage(self.queue, job, self.stage)
            self.stage.set_step("Exporting...")

        class _Page:  # with_project_clips shows its progress on this panel's stage
            queue, project, stage = self.queue, self.project, self.stage

        with_project_clips(_Page, rows, go)


class DrawnPanel(QWidget):
    def __init__(self, tab: AnimateTab):
        super().__init__()
        self.tab, self.ws = tab, tab.ws
        self.queue, self.project = tab.queue, tab.project
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(_muted("1. Pick an animation"))
        self.list = MotionList(self.project)
        self.list.current_changed.connect(lambda _r: self._poses())
        self.list.removed.connect(self.ws.changed.emit)
        left.addWidget(self.list, 1)
        done_lab = _muted("Drawn before")
        left.addWidget(done_lab)
        self.done_list = QListWidget()
        self.done_list.setMaximumHeight(110)
        self.done_list.currentItemChanged.connect(self._show_done)
        left.addWidget(self.done_list)
        lay.addLayout(left, 2)
        right = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(_muted("2. These poses are given to the model. Frames:"))
        self.frames = QSpinBox()
        self.frames.setRange(2, 16)
        self.frames.setValue(8)
        self.frames.valueChanged.connect(lambda _v: self._later.start())
        top.addWidget(self.frames)
        top.addStretch(1)
        right.addLayout(top)
        self.poses = Movie("Pick an animation to see its poses")
        right.addWidget(self.poses, 1)
        self.stage = ImageStage()
        self.stage.setVisible(False)
        right.addWidget(self.stage, 1)
        self.result = Movie("")
        self.result.setVisible(False)
        right.addWidget(self.result, 1)
        row = QHBoxLayout()
        self.draw = QPushButton("3. Draw the frames")
        self.draw.setObjectName("primary")
        self.draw.clicked.connect(self._draw)
        self.open = QPushButton("Open folder")
        self.open.clicked.connect(self._open)
        self.open.setEnabled(False)
        from ggs.ui.errors import trash_icon

        self.delete = QPushButton("Delete")
        self.delete.setIcon(trash_icon(self))
        self.delete.setToolTip("Delete these drawn frames")
        self.delete.clicked.connect(self._delete)
        self.delete.setEnabled(False)
        row.addWidget(self.draw)
        row.addWidget(self.open)
        row.addWidget(self.delete)
        row.addStretch(1)
        right.addLayout(row)
        self.note = _muted("")
        right.addWidget(self.note)
        lay.addLayout(right, 3)
        self._later = QTimer(self)
        self._later.setSingleShot(True)
        self._later.setInterval(250)
        self._later.timeout.connect(self._poses)
        self.shown: Path | None = None

    def refresh(self) -> None:
        self.list.reload()
        self._fill_done()

    def _fill_done(self) -> None:
        from ggs.rig.sprites import frames_dir

        self.done_list.clear()
        d = frames_dir(self.ws.dir)
        for f in sorted(d.glob("*/report.json")) if d.is_dir() else []:
            it = QListWidgetItem(f.parent.name.replace("_", " "))
            it.setData(ROW, str(f.parent))
            self.done_list.addItem(it)

    def _poses(self) -> None:
        row = self.list.current()
        self.result.setVisible(False)
        self.poses.setVisible(True)
        if not row:
            return
        from ggs.paths import shared_dir
        from ggs.rig.sprites import pose_preview

        cs = self.ws.rec.settings.character
        try:
            img = pose_preview(clip_for_row(row), self.frames.value(), body=cs.body if cs else None,
                               body_preset=cs.body_preset if cs else "standard")
            out = shared_dir() / "cache" / "previews" / "poses.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out)
            self.poses.picture(out)
        except Exception as exc:  # noqa: BLE001
            self.poses.play(None, f"Cannot show this one: {exc}")

    def _draw(self) -> None:
        row = self.list.current()
        if not row:
            from ggs.ui.errors import show_message

            show_message(self, "Pick an animation in the list first.")
            return
        self.poses.setVisible(False)
        self.result.setVisible(False)
        self.stage.setVisible(True)
        self.draw.setEnabled(False)

        def go(clips: list[Path]) -> None:
            if not clips:
                self._finish()
                return
            name = clips[0].stem
            job = Job(f"frames {self.ws.rec.name}", [C.character_frames(
                self.project, f"character/{self.ws.rec.name}", clips[0], self.frames.value(), name)], exit_ok=(0, 2))
            job.done = lambda ok: self._drawn(ok, job, name)
            run_on_stage(self.queue, job, self.stage)

        class _Page:
            queue, project, stage = self.queue, self.project, self.stage

        with_project_clips(_Page, [row], go)

    def _finish(self) -> None:
        self.stage.setVisible(False)
        self.draw.setEnabled(True)

    def _drawn(self, ok: bool, job: Job, name: str) -> None:
        from ggs.rig.sprites import frames_dir

        self._finish()
        self._fill_done()
        if not ok:
            self.poses.setVisible(True)
            return
        folder = frames_dir(self.ws.dir) / name
        self._show_folder(folder)
        if job.exit_code == 2:
            self.note.setText("Some frames did not follow their pose well. Draw again for another try, or use "
                              "fewer frames.")

    def _show_done(self, it: QListWidgetItem | None, _prev=None) -> None:
        if it is not None:
            self._show_folder(Path(it.data(ROW)))

    def _show_folder(self, folder: Path) -> None:
        self.shown = folder
        gif = folder / "preview.gif"
        if not gif.exists():
            from PIL import Image

            from ggs.export.animated import save_gif

            frames = [Image.open(f).convert("RGBA") for f in sorted((folder / "frames").glob("*.png"))]
            info = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            if frames:
                save_gif(frames, gif, info.get("fps", 8), (255, 255, 255), True)
        self.poses.setVisible(False)
        self.result.setVisible(True)
        self.result.play(gif)
        self.open.setEnabled(True)
        self.delete.setEnabled(True)
        self.note.setText(f"Saved with the character: {folder.name} (animated PNG, WebP, sprite sheet and frames).")

    def _delete(self) -> None:
        from ggs.items import ItemError, delete_frames
        from ggs.ui.errors import busy_refusal, confirm_delete, show_error

        if self.shown is None or busy_refusal(self, self.queue):
            return
        label = self.shown.name.replace("_", " ")
        if not confirm_delete(self, f"the drawn frames '{label}'"):
            return
        self.result.play(None)  # the GIF file is open while it plays
        try:
            delete_frames(self.project, f"character/{self.ws.rec.name}", self.shown.name)
        except (ItemError, OSError) as exc:
            show_error(self, str(exc), "Not deleted")
        self.shown = None
        self.open.setEnabled(False)
        self.delete.setEnabled(False)
        self.note.setText("")
        self.result.setVisible(False)
        self.poses.setVisible(True)
        self._fill_done()

    def _open(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        if self.shown:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.shown)))
