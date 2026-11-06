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

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QMovie, QPixmap
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
    QVBoxLayout,
    QWidget,
)

from ggs.ui import commands as C
from ggs.ui.icons import set_icon
from ggs.ui.jobs import Job, run_on_stage
from ggs.ui.widgets import CurrentStack, ImageStage

ROW = 256
ENGINE_FORMATS = [("Godot", "godot"), ("Unity", "unity"), ("Defold", "defold"), ("Parts + JSON", "rig")]
ANIM_FORMATS = [*ENGINE_FORMATS, ("GIF (one per animation)", "gif"), ("Animated PNG", "apng"),
                ("WebP", "webp"), ("Sprite sheets", "sheet")]
WARP_FORMATS = [("Sprite sheets + JSON (Godot, Unity, Defold)", "sheet"), ("Animated PNG", "apng"), ("WebP", "webp"),
                ("GIF (one per animation)", "gif")]
ADDED = re.compile(r"^added (.+\.json)\s*$")
METHODS = [("Warp", "Deforms the whole drawing with the skeleton. Exports pictures and sprite sheets."),
           ("Cut-Out", "Moves the sliced pieces like a puppet. Exports engine rigs and pictures."),
           ("Generated Frames", "Draws every frame with the pose-driven frame model.")]


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
        self.search.setPlaceholderText("Search")
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
    from ggs.paths import shared_dir
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
        spec = ch.rig()
    clip = load_clip(clip_path)
    if not warp:
        clip = ch.clip(clip)
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


class FramePlayer(QLabel):
    """Plays a list of PIL frames in a loop at ``fps`` (the drawn-frames skeleton preview)."""

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(260, 260)
        self.setStyleSheet("background: #111216; border-radius: 8px; color: #8a8d9a;")
        self.pix: list[QPixmap] = []
        self.k = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next)

    def set_frames(self, frames: list, fps: float, text: str = "") -> None:
        from PIL.ImageQt import ImageQt

        self.timer.stop()
        self.pix = [QPixmap.fromImage(ImageQt(f.convert("RGBA"))) for f in frames]
        self.k = 0
        if not self.pix:
            self.setPixmap(QPixmap())
            self.setText(text)
            return
        self.timer.start(max(30, round(1000 / max(fps, 0.5))))
        self._next()

    def _next(self) -> None:
        if not self.pix:
            return
        pm = self.pix[self.k % len(self.pix)]
        self.setPixmap(pm.scaled(self.width() - 12, self.height() - 12, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation))
        self.k += 1


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
    """Layout, generation and export on the left; the pieces open in an image editor on the right."""

    LAYOUTS: tuple = (("Suggested", "auto"), ("Chibi", "chibi"), ("Standard", "standard"), ("Tall", "tall"),
               ("Front View", "front"))
    EXTRAS: tuple = (("Long Hair", "hair_back"), ("Skirt / Dress", "skirt"), ("Cape", "cape"))

    def __init__(self, ws):
        super().__init__()
        from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox

        from ggs.ui.paint import ImageEditor

        self.ws = ws
        self.queue, self.project = ws.queue, ws.project
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        box = QGroupBox("Layout")
        form = QFormLayout(box)
        self.layout_box = QComboBox()
        for label, key in self.LAYOUTS:
            self.layout_box.addItem(label, key)
        self.layout_box.setToolTip("Body layout: neck, trunk and hands. Suggested is measured from the drawing.")
        self.layout_box.currentIndexChanged.connect(self._layout_changed)
        form.addRow("Body", self.layout_box)
        self.extras: dict[str, QCheckBox] = {}
        for label, key in self.EXTRAS:
            cb = QCheckBox(label)
            self.extras[key] = cb
            form.addRow("", cb)
        left.addWidget(box)
        self.make = QPushButton("Generate")
        self.make.setObjectName("primary")
        set_icon(self.make, "generate")
        self.make.setToolTip("Draw the character in the layout's pose and cut it into pieces")
        self.make.clicked.connect(self._make)
        left.addWidget(self.make)
        self.apply = QPushButton("Apply Layout")
        set_icon(self.apply, "layout")
        self.apply.setToolTip("Cut the existing drawings again with this layout (no new drawing)")
        self.apply.clicked.connect(self._apply)
        left.addWidget(self.apply)
        exp = QGroupBox("Export")
        ef = QFormLayout(exp)
        self.fmt = QComboBox()
        for label, f in ENGINE_FORMATS:
            self.fmt.addItem(label, f)
        self.height = QSpinBox()
        self.height.setRange(32, 2048)
        self.height.setValue(256)
        self.height.setSuffix(" px")
        ef.addRow("Format", self.fmt)
        ef.addRow("Height", self.height)
        self.export = QPushButton("Export...")
        set_icon(self.export, "export")
        self.export.clicked.connect(self._export)
        ef.addRow(self.export)
        left.addWidget(exp)
        left.addStretch(1)
        self.status = _muted("")
        left.addWidget(self.status)
        self.delete = QPushButton("Delete")
        set_icon(self.delete, "delete")
        self.delete.setToolTip("Delete the sliced version")
        self.delete.clicked.connect(self._delete)
        left.addWidget(self.delete)
        wrap = QWidget()
        wrap.setLayout(left)
        wrap.setFixedWidth(300)
        lay.addWidget(wrap)
        self.right = CurrentStack()
        self.stage = ImageStage(empty="No sliced version")
        self.editor = ImageEditor()
        self.editor.saved.connect(self._pieces_saved)
        self.right.addWidget(self.stage)
        self.right.addWidget(self.editor)
        lay.addWidget(self.right, 1)

    def _layout_changed(self) -> None:
        key = self.layout_box.currentData()
        for cb in self.extras.values():
            cb.setEnabled(key not in ("auto", "front"))

    def _layout_args(self) -> tuple[str, str, list[str] | None]:
        key = self.layout_box.currentData()
        if key == "front":
            return "front", "standard", None
        if key == "auto":
            return "side_simple", "auto", None
        return "side_simple", key, [k for k, cb in self.extras.items() if cb.isChecked()]

    def refresh(self) -> None:
        from ggs.rig.layout import Layout
        from ggs.rig.slices import pieces_preview, sliced_character, sliced_dir, sliced_info

        info = sliced_info(self.ws.dir)
        ok = sliced_character(self.ws.dir) is not None
        lay = info.get("layout")
        if lay:
            layout = Layout(**lay)
            for k, cb in self.extras.items():
                cb.setChecked(k in layout.extras)
        has_tries = bool(list((sliced_dir(self.ws.dir) / "work").glob("seed_*/master.json")))
        self.apply.setEnabled(has_tries and self.layout_box.currentData() != "front")
        self.export.setEnabled(ok)
        self.delete.setVisible(sliced_dir(self.ws.dir).exists())
        self._layout_changed()
        if ok:
            self.editor.load(pieces_preview(self.ws.dir))
            self.right.setCurrentWidget(self.editor)
        else:
            self.stage.set_image(None)
            self.right.setCurrentWidget(self.stage)
        parts = []
        if lay:
            parts.append(Layout(**lay).label())
        if ok and info.get("rough"):
            parts.append("Rough cut")
        elif info and not ok:
            parts.append("No clean cut")
        if ok and info.get("design") != self.ws.rec.accepted:
            parts.append("Older design")
        self.status.setText("  |  ".join(parts))

    def _pieces_saved(self) -> None:
        from ggs.rig.slices import save_pieces

        save_pieces(self.ws.dir, self.editor.path)
        self.ws.changed.emit()

    def _delete(self) -> None:
        from ggs.items import ItemError, delete_sliced
        from ggs.ui.errors import busy_refusal, confirm_delete, show_error

        if busy_refusal(self, self.queue) or not confirm_delete(self, "the sliced version"):
            return
        try:
            delete_sliced(self.project, f"character/{self.ws.rec.name}")
        except (ItemError, OSError) as exc:
            show_error(self, str(exc), "Not deleted")
        self.refresh()
        self.ws.changed.emit()

    def _run(self, recut: bool) -> None:
        template, layout, extras = self._layout_args()
        cmd = C.character_slice(self.project, f"character/{self.ws.rec.name}", template, layout, extras, recut)
        job = Job(f"slice {self.ws.rec.name}", [cmd], exit_ok=(0, 2))
        job.done = lambda ok: self._made(ok, job)
        self.make.setEnabled(False)
        self.apply.setEnabled(False)
        self.right.setCurrentWidget(self.stage)
        run_on_stage(self.queue, job, self.stage)

    def _make(self) -> None:
        self._run(False)

    def _apply(self) -> None:
        self._run(True)

    def _made(self, ok: bool, job: Job) -> None:
        self.make.setEnabled(True)
        self.refresh()
        from ggs.rig.slices import sliced_character

        if ok and job.exit_code == 2 and sliced_character(self.ws.dir) is None:
            from ggs.ui.errors import show_error

            show_error(self, "\n".join(job.output[-40:]), "No Clean Cut",
                       "None of the drawings could be cut into clean pieces. Generate again, or try another layout.")
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
            self.refresh()
            if ok:
                from ggs.ui.errors import show_saved

                show_saved(self, "Exported", out)

        job.done = done
        self.right.setCurrentWidget(self.stage)
        run_on_stage(self.queue, job, self.stage)


# ====================================================================== Animate


class AnimateTab(QWidget):
    def __init__(self, ws):
        super().__init__()
        self.ws = ws
        self.queue, self.project = ws.queue, ws.project
        lay = QVBoxLayout(self)
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
        self.stack = CurrentStack()
        self.warp = CutoutPanel(self, warp=True)
        self.cutout = CutoutPanel(self)
        self.drawn = DrawnPanel(self)
        self.stack.addWidget(self.warp)
        self.stack.addWidget(self.cutout)
        self.stack.addWidget(self.drawn)
        lay.addWidget(self.stack, 1)

    def _method(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        self.refresh()

    def refresh(self) -> None:
        self.stack.currentWidget().refresh()


class CutoutPanel(QWidget):
    """Animations on the character, in a live editor: the drawing bent in one piece (``warp``) or the cut-out
    puppet (the sliced version). Frames are only rendered to files on export."""

    def __init__(self, tab: AnimateTab, warp: bool = False):
        super().__init__()
        from ggs.ui.anim_editor import AnimEditor

        self.warp = warp
        self.tab, self.ws = tab, tab.ws
        self.queue, self.project = tab.queue, tab.project
        self._rig_key = None
        outer = QVBoxLayout(self)
        self.need = QFrame()
        self.need.setObjectName("panel")
        nl = QVBoxLayout(self.need)
        self.need_text = _muted("")
        nl.addWidget(self.need_text)
        nrow = QHBoxLayout()
        self.detail = QComboBox()
        for label, d in (("Mesh: Normal", "normal"), ("Mesh: Fine", "fine"), ("Mesh: Coarse", "coarse")):
            self.detail.addItem(label, d)
        self.detail.setVisible(warp)
        self.go = QPushButton("Create Rig" if warp else "Sliced Version")
        self.go.setObjectName("primary")
        set_icon(self.go, "rig")
        self.go.clicked.connect(self._make_warp if warp else (lambda: self.ws.show_tab(1)))
        nrow.addWidget(self.detail)
        nrow.addWidget(self.go)
        nrow.addStretch(1)
        nl.addLayout(nrow)
        self.need_stage = ImageStage()
        self.need_stage.setVisible(False)
        nl.addWidget(self.need_stage, 1)
        outer.addWidget(self.need, 1)
        self.body = QWidget()
        lay = QHBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        left = QVBoxLayout()
        self.list = MotionList(self.project, multi=True)
        self.list.current_changed.connect(self._show)
        self.list.removed.connect(self.ws.changed.emit)
        left.addWidget(self.list, 1)
        if warp:
            remake = QPushButton("Rebuild Rig")
            set_icon(remake, "regenerate")
            remake.setToolTip("Find the joints again and lay a new mesh (after a new design, or another mesh size)")
            remake.clicked.connect(self._make_warp)
            left.addWidget(self.detail_again(remake))
            touch = QPushButton("Edit Picture...")
            set_icon(touch, "brush")
            touch.setToolTip("Paint over the drawing that is bent (brush, eraser, colour picker, fill)")
            touch.clicked.connect(self._touch_up)
            left.addWidget(touch)
        lay.addLayout(left, 2)
        right = QVBoxLayout()
        self.editor = AnimEditor()
        self.editor.edited.connect(self._edited)
        right.addWidget(self.editor, 1)
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
        self.export = QPushButton("Export Selected...")
        self.export.setObjectName("primary")
        set_icon(self.export, "export")
        self.export.clicked.connect(self._export)
        right.addWidget(self.export)
        lay.addLayout(right, 3)
        outer.addWidget(self.body, 1)

    def detail_again(self, button: QPushButton) -> QWidget:
        """The rebuild button with its own mesh-size choice (the first one sits on the 'get it ready' card)."""
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        self.detail2 = QComboBox()
        for i in range(self.detail.count()):
            self.detail2.addItem(self.detail.itemText(i), self.detail.itemData(i))
        row.addWidget(self.detail2, 1)
        row.addWidget(button)
        return w

    def character(self) -> Path | None:
        from ggs.rig.slices import sliced_character
        from ggs.rig.warp import warp_skin

        return warp_skin(self.ws.dir) if self.warp else sliced_character(self.ws.dir)

    def _side(self, ch: Path) -> bool:
        if self.warp:
            return json.loads(ch.read_text(encoding="utf-8")).get("rig") == "humanoid_side_v1"
        import yaml

        return (yaml.safe_load(ch.read_text(encoding="utf-8")) or {}).get("rig") == "humanoid_side_v1"

    def refresh(self) -> None:
        ch = self.character()
        side = ch is not None and self._side(ch)
        self.need.setVisible(not side)
        self.body.setVisible(side)
        cs = self.ws.rec.settings.character
        view = cs.view if cs else "side"
        if ch is None and self.warp:
            self.need_text.setText("No warp rig." if view == "side" else "Warp needs a side-view design.")
            self.go.setEnabled(view == "side")
        elif ch is None:
            self.need_text.setText("No sliced version.")
        elif not side:
            self.need_text.setText("The sliced version is front view; animations need a side view.")
        else:
            key = (str(ch), ch.stat().st_mtime)
            if key != self._rig_key:
                from ggs.ui.anim_editor import PuppetRenderer, WarpRenderer

                try:
                    self.editor.set_renderer(WarpRenderer(ch) if self.warp else PuppetRenderer(ch))
                    self._rig_key = key
                except Exception as exc:  # noqa: BLE001 - a broken rig shows a message, not a crash
                    from ggs.ui.errors import show_error

                    show_error(self, str(exc), "Cannot show this character")
            self.list.reload()

    def _touch_up(self) -> None:
        from PySide6.QtWidgets import QApplication

        from ggs.rig.warp import move_joints, warp_skin
        from ggs.ui.paint import PaintDialog

        skin = warp_skin(self.ws.dir)
        if skin is None:
            return
        dlg = PaintDialog(skin.parent / "skin.png", self, "Edit Picture")
        if dlg.exec() and dlg.saved:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                move_joints(skin, {})  # same joints, new picture: rebuild the layers and mesh
            finally:
                QApplication.restoreOverrideCursor()
            self._rig_key = None
            self.refresh()

    def _make_warp(self) -> None:
        on_card = self.need.isVisible()
        stage = self.need_stage if on_card else self.stage
        combo = self.detail if on_card or not hasattr(self, "detail2") else self.detail2
        job = Job(f"warp {self.ws.rec.name}", [C.character_warp(self.project, f"character/{self.ws.rec.name}",
                                                                 combo.currentData() or "normal")])

        def done(ok: bool) -> None:
            stage.setVisible(False)
            self.editor.setVisible(True)
            self._rig_key = None
            self.refresh()

        job.done = done
        stage.setVisible(True)
        if stage is self.stage:
            self.editor.setVisible(False)
        run_on_stage(self.queue, job, stage)
        stage.set_step("Finding the joints and laying the mesh...")

    @staticmethod
    def _key(row: dict) -> str:
        return C.slug(row["name"], 40)

    def _show(self, row: dict | None) -> None:
        if not row or self.character() is None:
            return
        from ggs.rig.clip import load_clip
        from ggs.rig.edits import edited_clip, edits_dir

        key = self._key(row)
        try:
            path = edited_clip(self.ws.dir, key) or clip_for_row(row)
            self.editor.set_clip(load_clip(path), key, edits_dir(self.ws.dir))
        except Exception as exc:  # noqa: BLE001 - a broken clip shows a message, not a crash
            self.editor.canvas.set_content(None, text=f"Cannot show this one: {exc}")

    def _edited(self, path) -> None:
        if path is None:  # touch-ups undone: show the library version again
            self._show(self.list.current())

    def _export(self) -> None:
        from ggs.rig.edits import edited_clip

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
        touched = [edited_clip(self.ws.dir, self._key(r)) for r in rows]  # touch-ups win over the library
        plain = [r for r, t in zip(rows, touched) if t is None]
        mine = [t for t in touched if t is not None]
        self.editor.setVisible(False)
        self.stage.setVisible(True)

        def go(clips: list[Path]) -> None:
            clips = mine + clips
            if not clips:
                self.stage.setVisible(False)
                self.editor.setVisible(True)
                return
            if engine or len(clips) > 1:
                out = base / (fmt if engine else f"animations_{fmt}")
            else:
                ext = {"gif": ".gif", "webp": ".webp", "apng": ".png", "sheet": "_sheet.png"}[fmt]
                out = base / f"{clips[0].stem}{ext}"
            cmd = C.export_many(clips, ch, fmt, out, height=self.height.value(),
                                matte="#FFFFFF" if fmt == "gif" else None)
            job = Job(f"export {self.ws.rec.name}", [cmd])

            def done(ok: bool) -> None:
                self.stage.setVisible(False)
                self.editor.setVisible(True)
                self.list.reload()
                if ok:
                    from ggs.ui.errors import show_saved

                    show_saved(self, "Exported", out, f"Exported {len(clips)} animation(s) to\n{out}")
                    self.ws.changed.emit()

            job.done = done
            run_on_stage(self.queue, job, self.stage)
            self.stage.set_step("Rendering the frames...")

        class _Page:  # with_project_clips shows its progress on this panel's stage
            queue, project, stage = self.queue, self.project, self.stage

        with_project_clips(_Page, plain, go)


class DrawnPanel(QWidget):
    def __init__(self, tab: AnimateTab):
        super().__init__()
        self.tab, self.ws = tab, tab.ws
        self.queue, self.project = tab.queue, tab.project
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.list = MotionList(self.project)
        self.list.current_changed.connect(lambda _r: self._poses())
        self.list.removed.connect(self.ws.changed.emit)
        left.addWidget(self.list, 1)
        left.addWidget(_muted("Generated"))
        self.done_list = QListWidget()
        self.done_list.setMaximumHeight(110)
        self.done_list.currentItemChanged.connect(self._show_done)
        left.addWidget(self.done_list)
        lay.addLayout(left, 2)
        right = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(QLabel("Frames"))
        self.frames = QSpinBox()
        self.frames.setRange(2, 16)
        self.frames.setValue(12)
        self.frames.setToolTip("Frames per animation; the preview plays the skeleton at this count")
        self.frames.valueChanged.connect(lambda _v: self._later.start())
        top.addWidget(self.frames)
        top.addStretch(1)
        right.addLayout(top)
        self.poses = FramePlayer("Select an animation")
        right.addWidget(self.poses, 1)
        self.stage = ImageStage()
        self.stage.setVisible(False)
        right.addWidget(self.stage, 1)
        self.result = Movie("")
        self.result.setVisible(False)
        right.addWidget(self.result, 1)
        row = QHBoxLayout()
        self.draw = QPushButton("Generate Frames")
        self.draw.setObjectName("primary")
        set_icon(self.draw, "generate")
        self.draw.clicked.connect(self._draw)
        self.open = QPushButton("Open Folder")
        set_icon(self.open, "open")
        self.open.clicked.connect(self._open)
        self.open.setEnabled(False)

        self.delete = QPushButton("Delete")
        set_icon(self.delete, "delete")
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

        try:
            from ggs.rig.clip import load_clip
            from ggs.rig.poses import rig_for
            from ggs.rig.sprites import _design_lengths, frame_indices, openpose_frames

            clip = load_clip(clip_for_row(row))
            n = self.frames.value()
            frames = openpose_frames(rig_for("side"), clip, frame_indices(clip, n), _design_lengths(self.ws.dir),
                                     size=(288, 384))
            self.poses.set_frames(frames, n / clip.duration)  # the skeleton the model follows, at this count
        except Exception as exc:  # noqa: BLE001
            self.poses.set_frames([], 1.0, f"Cannot show this one: {exc}")

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
