"""Item workspace: big preview + versions on top, the chat underneath, Advanced as a side drawer."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ggs.ui import commands as C
from ggs.ui.errors import friendly_error
from ggs.ui.jobs import Job, JobQueue
from ggs.ui.kinds import SIZE_LABELS, kind
from ggs.ui.theme import BAD, MUTED
from ggs.ui.widgets import checker_pixmap


class Bubble(QFrame):
    """One chat message. App messages can carry version thumbnails and a hidden 'details' text."""

    picked = Signal(int)

    def __init__(self, text: str, user: bool, thumbs: list[tuple[int, Path]] = (), error: bool = False,
                 details: str = ""):
        super().__init__()
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if error:
            lab.setStyleSheet(f"color: {BAD};")
        lay.addWidget(lab)
        if thumbs:
            row = QHBoxLayout()
            for n, p in thumbs:
                b = QPushButton()
                b.setObjectName("flat")
                b.setIcon(QIcon(checker_pixmap(p, 96)))
                b.setIconSize(QSize(96, 96))
                b.setToolTip(f"version {n}")
                b.clicked.connect(lambda _=False, n=n: self.picked.emit(n))
                row.addWidget(b)
            row.addStretch(1)
            lay.addLayout(row)
        if details:
            more = QPushButton("details")
            more.setObjectName("flat")
            box = QPlainTextEdit(details)
            box.setReadOnly(True)
            box.setVisible(False)
            box.setFixedHeight(120)
            more.clicked.connect(lambda: box.setVisible(not box.isVisible()))
            lay.addWidget(more, 0, Qt.AlignmentFlag.AlignLeft)
            lay.addWidget(box)
        self.setMaximumWidth(620)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)
        self.user = user


class Advanced(QFrame):
    """Settings that need some idea of how the model works. Closed by default."""

    def __init__(self, styles: list[str], models: list[str]):
        super().__init__()
        self.setObjectName("panel")
        self.setFixedWidth(250)
        f = QFormLayout(self)
        title = QLabel("Advanced")
        title.setObjectName("h2")
        f.addRow(title)
        self.fix_seed = QCheckBox("Use this seed")
        self.seed = QSpinBox()
        self.seed.setRange(0, 999_999)
        f.addRow(self.fix_seed)
        f.addRow("Seed", self.seed)
        self.count = QSpinBox()
        self.count.setRange(2, 8)
        self.count.setValue(3)
        f.addRow("Variations", self.count)
        self.style = QComboBox()
        self.style.addItem("(no style)", None)
        for s in styles:
            self.style.addItem(s, s)
        self.weight = QDoubleSpinBox()
        self.weight.setRange(0.0, 2.0)
        self.weight.setSingleStep(0.1)
        self.weight.setValue(1.0)
        f.addRow("Style", self.style)
        f.addRow("Style strength", self.weight)
        self.model = QComboBox()
        self.model.addItem("(project's)", None)
        for m in models:
            self.model.addItem(m, m)
        f.addRow("Model", self.model)
        note = QLabel("Seeds make a drawing repeatable: the same seed and words give the same picture.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        f.addRow(note)

    def seed_value(self) -> int | None:
        return self.seed.value() if self.fix_seed.isChecked() else None


class ItemWorkspace(QWidget):
    changed = Signal()  # the item's versions changed (the tree refreshes its thumbnail)

    def __init__(self, queue: JobQueue, project: Path, item_dir: Path, styles: list[str], models: list[str]):
        super().__init__()
        self.queue, self.project, self.dir = queue, Path(project), Path(item_dir)
        self.rec = None
        self.shown: int | None = None
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        main = QVBoxLayout()
        root.addLayout(main, 1)
        self.adv = Advanced(styles, models)
        self.adv.setVisible(False)
        root.addWidget(self.adv)

        # header
        head = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("h2")
        self.kind = QLabel()
        self.kind.setObjectName("chip")
        head.addWidget(self.title)
        head.addWidget(self.kind)
        head.addStretch(1)
        self.size_group = QButtonGroup(self)
        self.size_box = QHBoxLayout()
        head.addLayout(self.size_box)
        self.export_btn = QPushButton("Export")
        self.export_btn.setObjectName("primary")
        self.export_btn.clicked.connect(self._export)
        head.addWidget(self.export_btn)
        adv = QPushButton("Advanced")
        adv.setObjectName("flat")
        adv.setCheckable(True)
        adv.toggled.connect(self.adv.setVisible)
        head.addWidget(adv)
        main.addLayout(head)

        # preview + versions
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(260)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main.addWidget(self.preview, 3)
        self.versions = QListWidget()
        self.versions.setViewMode(QListWidget.ViewMode.IconMode)
        self.versions.setFlow(QListWidget.Flow.LeftToRight)
        self.versions.setWrapping(False)
        self.versions.setIconSize(QSize(64, 64))
        self.versions.setFixedHeight(104)
        self.versions.setGridSize(QSize(80, 92))
        self.versions.setMovement(QListWidget.Movement.Static)
        self.versions.itemClicked.connect(lambda it: self._show_version(it.data(256)))
        main.addWidget(self.versions)

        # chat
        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_inner = QWidget()
        self.chat_lay = QVBoxLayout(self.chat_inner)
        self.chat_lay.addStretch(1)
        self.chat_area.setWidget(self.chat_inner)
        bar = self.chat_area.verticalScrollBar()
        bar.rangeChanged.connect(lambda _a, hi: bar.setValue(hi))  # keep the newest message in view
        main.addWidget(self.chat_area, 2)

        # progress + input
        self.progress_text = QLabel("")
        self.progress_text.setObjectName("muted")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        for w in (self.progress_text, self.progress):
            w.setVisible(False)
            main.addWidget(w)
        box = QHBoxLayout()
        self.input = QPlainTextEdit()
        self.input.setFixedHeight(64)
        box.addWidget(self.input, 1)
        col = QVBoxLayout()
        self.send = QPushButton("Send")
        self.send.setObjectName("primary")
        self.send.clicked.connect(self._send)
        row2 = QHBoxLayout()
        self.regen = QPushButton("↻ Regenerate")
        self.regen.setToolTip("Draw it again with a new seed")
        self.regen.clicked.connect(lambda: self._run("regenerate"))
        self.vars = QPushButton("Variations")
        self.vars.setToolTip("Several new versions to choose from")
        self.vars.clicked.connect(lambda: self._run("variations"))
        row2.addWidget(self.regen)
        row2.addWidget(self.vars)
        col.addWidget(self.send)
        col.addLayout(row2)
        box.addLayout(col)
        main.addLayout(box)
        self.reload()

    # ------------------------------------------------------------------ data

    def reload(self) -> None:
        from ggs.items import ItemRecord

        self.rec = ItemRecord.model_validate_json((self.dir / "item.json").read_text(encoding="utf-8"))
        k = kind(self.rec.asset_class)
        self.title.setText(self.rec.name)
        self.kind.setText(f"{k.icon} {k.label}")
        self._sizes(k.sizes)
        if self.rec.settings.style:
            i = self.adv.style.findData(self.rec.settings.style)
            self.adv.style.setCurrentIndex(max(0, i))
        self.versions.clear()
        for v in self.rec.versions:
            if not v.file:
                continue
            it = QListWidgetItem(QIcon(checker_pixmap(self.dir / v.file, 64)), f"v{v.n}")
            it.setData(256, v.n)
            it.setToolTip(v.prompt + ("" if v.passed else "\n(did not pass the quality check)"))
            self.versions.addItem(it)
        self._rebuild_chat()
        self._show_version(self.rec.current)
        first = not self.rec.versions
        self.input.setPlaceholderText(
            "Describe it, then press Send" if first else "Say what to change, e.g. 'make it blue' (keeps the rest)")
        if first and not self.input.toPlainText():
            self.input.setPlainText(self.rec.description)
        self.send.setText("Create" if first else "Refine")
        self.regen.setEnabled(not first)
        self.vars.setEnabled(bool(self.rec.description))
        self.export_btn.setEnabled(self.rec.current is not None)

    def _sizes(self, sizes) -> None:
        for b in self.size_group.buttons():
            self.size_group.removeButton(b)
        while self.size_box.count():
            it = self.size_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if len(sizes) < 2:
            return
        for s in sizes:
            b = QPushButton(SIZE_LABELS[s][0])
            b.setObjectName("tile")
            b.setCheckable(True)
            b.setFixedSize(52, 32)
            b.setStyleSheet("padding: 0px;")
            b.setToolTip(f"Export size: {SIZE_LABELS[s][1]}")
            b.setProperty("tile_size", s)
            b.setChecked(s == self.rec.settings.size)
            b.toggled.connect(lambda on, s=s: on and self._set_size(s))
            self.size_group.addButton(b)
            self.size_box.addWidget(b)

    def _set_size(self, size: str) -> None:
        self.rec.settings.size = size
        (self.dir / "item.json").write_text(self.rec.model_dump_json(indent=1), encoding="utf-8")

    def _show_version(self, n: int | None) -> None:
        self.shown = n
        v = self.rec.get_version(n) if self.rec else None
        if v is None or not v.file:
            self.preview.setText("Nothing drawn yet. Describe it below and press Create.")
            self.preview.setStyleSheet(f"color: {MUTED};")
            return
        self.preview.setStyleSheet("")
        side = max(200, min(self.preview.width(), self.preview.height()) - 10)
        self.preview.setPixmap(checker_pixmap(self.dir / v.file, side))
        for i in range(self.versions.count()):
            if self.versions.item(i).data(256) == n:
                self.versions.setCurrentRow(i)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self.rec is not None:
            self._show_version(self.shown)

    def _rebuild_chat(self) -> None:
        while self.chat_lay.count() > 1:
            it = self.chat_lay.takeAt(1)
            if it.widget():
                it.widget().deleteLater()
        for t in self.rec.chat:
            thumbs = [(n, self.dir / v.file) for n in t.versions if (v := self.rec.get_version(n)) and v.file]
            self._bubble(t.text, t.role == "user", thumbs)

    def _bubble(self, text: str, user: bool, thumbs=(), error=False, details="") -> None:
        b = Bubble(text, user, thumbs, error, details)
        b.picked.connect(self._show_version)
        row = QHBoxLayout()
        if user:
            row.addStretch(1)
        row.addWidget(b)
        if not user:
            row.addStretch(1)
        holder = QWidget()
        holder.setLayout(row)
        self.chat_lay.addWidget(holder)

    # ------------------------------------------------------------------ actions

    def _send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        self._run("new" if not self.rec.versions else "refine", text)

    def _run(self, action: str, text: str = "") -> None:
        if self.queue.busy:
            self._bubble("I'm still working on the last request; this one is queued.", False)
        cur = self.rec.get_version(self.shown) if self.shown else None
        if action == "refine" and cur is not None and cur.n != self.rec.current:  # refine the version on screen
            self.rec.current = cur.n
            (self.dir / "item.json").write_text(self.rec.model_dump_json(indent=1), encoding="utf-8")
        self._save_advanced()
        ref = f"{self.rec.asset_class}/{self.rec.name}"
        cmd = C.asset_gen(self.project, ref, action, text, count=self.adv.count.value(),
                          seed=self.adv.seed_value(), model=self.adv.model.currentData())
        if text:
            self._bubble(text, True)
        self.input.clear()
        self._busy(True, "Starting...")
        job = Job(f"{action} {self.rec.name}", [cmd], exit_ok=(0, 2))
        job.done = lambda ok, job=job: self._finished(ok, job)
        self.queue.submit(job)

    def _save_advanced(self) -> None:
        s = self.rec.settings
        s.style = self.adv.style.currentData()
        s.style_weight = self.adv.weight.value() if s.style else None
        (self.dir / "item.json").write_text(self.rec.model_dump_json(indent=1), encoding="utf-8")

    def _finished(self, ok: bool, job: Job) -> None:
        self._busy(False)
        if not ok:
            raw = "\n".join(job.output[-60:])
            self._bubble(friendly_error(raw), False, error=True, details=raw)
            return
        self.reload()
        self.changed.emit()

    def _busy(self, on: bool, text: str = "") -> None:
        for w in (self.progress, self.progress_text):
            w.setVisible(on)
        self.progress_text.setText(text)
        for b in (self.send, self.regen, self.vars):
            b.setEnabled(not on)
        if on:
            self.queue.stage.connect(self.progress_text.setText)
        else:
            try:
                self.queue.stage.disconnect(self.progress_text.setText)
            except (RuntimeError, TypeError):
                pass

    def _export(self) -> None:
        cmd = C.asset_export(self.project, f"{self.rec.asset_class}/{self.rec.name}", self.rec.settings.size,
                             self.shown)
        job = Job(f"export {self.rec.name}", [cmd])

        def done(ok: bool) -> None:
            if ok:
                out = next((ln.split("exported ", 1)[1] for ln in job.output if ln.startswith("exported ")), "")
                self._bubble(f"Exported to {out}", False)
            else:
                raw = "\n".join(job.output[-30:])
                self._bubble(friendly_error(raw), False, error=True, details=raw)

        job.done = done
        self.queue.submit(job)


class Placeholder(QWidget):
    def __init__(self, text: str):
        super().__init__()
        lay = QVBoxLayout(self)
        lab = QLabel(text)
        lab.setObjectName("muted")
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lab)



class CharacterWorkspace(ItemWorkspace):
    """A character: the item workspace plus the body & pose editor on the left."""

    def __init__(self, queue: JobQueue, project: Path, item_dir: Path, styles: list[str], models: list[str]):
        super().__init__(queue, project, item_dir, styles, models)
        from ggs.items import CharacterSettings
        from ggs.ui.character_editor import BodyPoseEditor

        self.editor = BodyPoseEditor(self.rec.settings.character or CharacterSettings())
        self.editor.changed.connect(self._character_changed)
        self.layout().insertWidget(0, self.editor)

    def _character_changed(self, cs) -> None:
        self.rec.settings.character = cs
        (self.dir / "item.json").write_text(self.rec.model_dump_json(indent=1), encoding="utf-8")
        if self.rec.versions:
            self.input.setPlaceholderText("Pose or body changed: press Regenerate to draw it in the new pose, "
                                          "or type a change to refine the current drawing")
