"""Item workspace: the picture in the middle, the prompt under it, Advanced as a side drawer.

No chat and no stored versions: the history keeps how each drawing was made (``ggs.items``), so Previous / Next
bring an earlier drawing back (from the cache, or drawn again). Regenerate sits by the picture: the same words,
a new seed. Under the picture one prompt box and two buttons: "Change it" edits the drawing on
screen, "Draw new" starts over from the words in the box.
Characters add the body & pose editor, "Use this design", and the Sliced version and Animate tabs.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ggs.ui import commands as C
from ggs.ui.icons import pixmap, set_icon
from ggs.ui.jobs import Job, JobQueue, run_on_stage
from ggs.ui.kinds import SIZE_LABELS, kind
from ggs.ui.widgets import CurrentStack, ImageStage, PromptEdit


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

    def seed_value(self) -> int | None:
        return self.seed.value() if self.fix_seed.isChecked() else None


class Placeholder(QWidget):
    def __init__(self, text: str):
        super().__init__()
        lay = QVBoxLayout(self)
        lab = QLabel(text)
        lab.setObjectName("muted")
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lab)


def _flat(text: str, tip: str = "", ic: str | None = None) -> QPushButton:
    b = QPushButton(text)
    if ic:
        set_icon(b, ic)
    b.setToolTip(tip)
    return b


class DrawPanel(QWidget):
    """Picture + history arrows + Regenerate + prompt with Change it / Draw new. Shared by every item kind."""

    def __init__(self, ws: ItemWorkspace):
        super().__init__()
        self.ws = ws
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.stage = ImageStage(empty="Nothing drawn yet.\nDescribe it below and press Draw.")
        lay.addWidget(self.stage, 1)
        row = QHBoxLayout()
        self.prev = _flat("Previous", "Previous drawing")
        self.prev.clicked.connect(lambda: ws.step(-1))
        self.where = QLabel("")
        self.where.setObjectName("muted")
        self.next = _flat("Next", "Next drawing")
        self.next.clicked.connect(lambda: ws.step(1))
        self.regen = _flat("Regenerate", "Same description, new seed", "regenerate")
        self.regen.clicked.connect(lambda: ws.run("regenerate"))

        self.delete = _flat("", "Delete this drawing from the history")
        set_icon(self.delete, "delete")
        self.delete.clicked.connect(ws.delete_drawing)
        row.addWidget(self.prev)
        row.addWidget(self.where)
        row.addWidget(self.next)
        row.addWidget(self.delete)
        row.addStretch(1)
        row.addWidget(self.regen)
        lay.addLayout(row)
        self.caption = QLabel("")
        self.caption.setObjectName("muted")
        self.caption.setWordWrap(True)
        self.caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.caption)
        self.extra = QHBoxLayout()  # characters: "Use this design" and what comes after it
        lay.addLayout(self.extra)
        box = QHBoxLayout()
        self.input = PromptEdit()
        self.input.setFixedHeight(62)
        self.input.submitted.connect(self._enter)
        box.addWidget(self.input, 1)
        col = QVBoxLayout()
        self.change = QPushButton("Refine")
        self.change.setObjectName("primary")
        set_icon(self.change, "refine")
        self.change.setToolTip("Edit the drawing on screen: type what to change, e.g. 'make the hat blue'")
        self.change.clicked.connect(self._change)
        self.new = QPushButton("Generate")
        set_icon(self.new, "generate")
        self.new.setToolTip("Start over: a new drawing from the words in the box")
        self.new.clicked.connect(self._new)
        col.addWidget(self.change)
        col.addWidget(self.new)
        box.addLayout(col)
        lay.addLayout(box)

    def _enter(self) -> None:
        (self._change if self.ws.rec.shown() else self._new)()

    def _change(self) -> None:
        text = self.input.toPlainText().strip()
        if text:
            self.ws.run("refine", text)

    def _new(self) -> None:
        text = self.input.toPlainText().strip()
        if text:
            self.ws.run("new", text)

    def set_busy(self, on: bool) -> None:
        for b in (self.prev, self.next, self.regen, self.change, self.new, self.delete):
            b.setEnabled(not on)
        if not on:
            self.ws.refresh_buttons()


class ItemWorkspace(QWidget):
    changed = Signal()  # the item changed (the tree refreshes its thumbnail)
    delete_requested = Signal(Path)  # the item's Delete button (the project view deletes it)

    def __init__(self, queue: JobQueue, project: Path, item_dir: Path, styles: list[str], models: list[str]):
        super().__init__()
        self.queue, self.project, self.dir = queue, Path(project), Path(item_dir)
        self.rec = None
        self.working = False
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.left = QVBoxLayout()  # characters put the body & pose editor here
        root.addLayout(self.left)
        main = QVBoxLayout()
        root.addLayout(main, 1)
        self.adv = Advanced(styles, models)
        self.adv.setVisible(False)
        root.addWidget(self.adv)

        head = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("h2")
        self.kind = QLabel()
        self.kind.setObjectName("chip")
        self.kind_icon = QLabel()
        head.addWidget(self.title)
        head.addWidget(self.kind_icon)
        head.addWidget(self.kind)
        head.addStretch(1)
        # second row (one long row pushed the window wider than a laptop screen): tabs, export size, Export
        row2 = QHBoxLayout()
        self.tabs_box = QHBoxLayout()  # characters: Design | Sliced version | Animate
        row2.addLayout(self.tabs_box)
        row2.addStretch(1)
        self.size_group = QButtonGroup(self)
        self.size_box = QHBoxLayout()
        row2.addLayout(self.size_box)
        self.export_btn = QPushButton("Export")
        set_icon(self.export_btn, "export")
        self.export_btn.clicked.connect(self._export)
        row2.addWidget(self.export_btn)

        self.delete_btn = QPushButton()
        set_icon(self.delete_btn, "delete")
        self.delete_btn.setObjectName("flat")
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.dir))
        head.addWidget(self.delete_btn)
        adv = QPushButton("Advanced")
        set_icon(adv, "advanced")
        adv.setObjectName("flat")
        adv.setCheckable(True)
        adv.toggled.connect(self.adv.setVisible)
        head.addWidget(adv)
        self.adv_btn = adv
        main.addLayout(head)
        main.addLayout(row2)
        self.pages = CurrentStack()
        main.addWidget(self.pages, 1)
        self.draw = DrawPanel(self)
        self.pages.addWidget(self.draw)
        self.reload()

    # ------------------------------------------------------------------ data

    def reload(self) -> None:
        from ggs.items import ItemRecord

        self.rec = ItemRecord.model_validate_json((self.dir / "item.json").read_text(encoding="utf-8"))
        k = kind(self.rec.asset_class)
        self.title.setText(self.rec.name)
        self.kind.setText(k.label)
        self.kind_icon.setPixmap(pixmap(k.icon, 22))
        self._sizes(k.sizes)
        if self.rec.settings.style:
            i = self.adv.style.findData(self.rec.settings.style)
            self.adv.style.setCurrentIndex(max(0, i))
        if not self.draw.input.toPlainText().strip():
            self.draw.input.setPlainText(self.rec.description)
        self.show_current()
        self.refresh_buttons()

    def show_current(self) -> None:
        v = self.rec.get_version()
        path = self.dir / v.file if v is not None and v.file else None
        note = ""
        if v is not None and not v.passed:
            from ggs.items import friendly

            note = "Did not pass the quality check: " + friendly(v.failures).split(". Try")[0]
        self.draw.stage.set_image(path if path is not None and path.exists() else None, note)
        if v is None:
            self.draw.caption.setText("")
        else:
            parts = self.rec.design_text()
            self.draw.caption.setText("  →  ".join(parts))

    def refresh_buttons(self) -> None:
        ns = [v.n for v in self.rec.shown()]
        cur = self.rec.current
        i = ns.index(cur) if cur in ns else -1
        has = bool(ns) and i >= 0
        d = self.draw
        d.prev.setEnabled(has and i > 0 and not self.working)
        d.next.setEnabled(has and i < len(ns) - 1 and not self.working)
        d.where.setText(f"{i + 1} of {len(ns)}" if has else "")
        d.regen.setEnabled(bool(self.rec.description) and has and not self.working)
        d.change.setVisible(has)
        d.new.setText("Generate")
        d.new.setObjectName("" if has else "primary")
        set_icon(d.new, "generate")
        d.new.style().unpolish(d.new)
        d.new.style().polish(d.new)
        d.input.setPlaceholderText("Description, or a change to refine" if has else "Description")
        self.export_btn.setEnabled(has and not self.working)
        design = has and self.rec.accepted == cur
        d.delete.setEnabled(has and not self.working and not design)
        d.delete.setToolTip("This drawing is the character's design: use another design first, or delete the "
                            "whole character" if design else "Delete this drawing from the history")
        self.delete_btn.setToolTip(f"Delete {self.rec.name}")

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
            b.setChecked(s == self.rec.settings.size)
            b.toggled.connect(lambda on, s=s: on and self._set_size(s))
            self.size_group.addButton(b)
            self.size_box.addWidget(b)

    def _save(self) -> None:
        (self.dir / "item.json").write_text(self.rec.model_dump_json(indent=1), encoding="utf-8")

    def _set_size(self, size: str) -> None:
        self.rec.settings.size = size
        self._save()

    # ------------------------------------------------------------------ actions

    def step(self, delta: int) -> None:
        ns = [v.n for v in self.rec.shown()]
        if self.rec.current not in ns:
            return
        i = ns.index(self.rec.current) + delta
        if 0 <= i < len(ns):
            self._job("goto", version=ns[i], title="Bringing it back...")

    def delete_drawing(self) -> None:
        from ggs.items import ItemError, delete_drawing
        from ggs.ui.errors import busy_refusal, confirm_delete, show_error

        if self.rec.current is None or busy_refusal(self, self.queue):
            return
        if not confirm_delete(self, "this drawing", "Previous and Next will not show it any more."):
            return
        try:
            rec, redraw = delete_drawing(self.project, f"{self.rec.asset_class}/{self.rec.name}")
        except (ItemError, OSError) as exc:
            show_error(self, str(exc), "Not deleted", str(exc)[:1].upper() + str(exc)[1:] + ".")
            return
        self.reload()
        if redraw and rec.current is not None:
            self._job("goto", version=rec.current, title="Bringing back the drawing before it...")
        self.changed.emit()

    def run(self, action: str, text: str = "") -> None:
        self._save_advanced()
        self._job(action, text)

    def _job(self, action: str, text: str = "", version: int | None = None, title: str = "") -> None:
        ref = f"{self.rec.asset_class}/{self.rec.name}"
        cmd = C.asset_gen(self.project, ref, action, text, seed=self.adv.seed_value(), model=self.adv.model.currentData(),
                          version=version)
        job = Job(f"{action} {self.rec.name}", [cmd], exit_ok=(0, 2))
        job.done = lambda ok: self._finished(ok)
        self.working = True
        self.draw.set_busy(True)
        run_on_stage(self.queue, job, self.draw.stage)
        if title:
            self.draw.stage.set_step(title)

    def _save_advanced(self) -> None:
        s = self.rec.settings
        s.style = self.adv.style.currentData()
        s.style_weight = self.adv.weight.value() if s.style else None
        self._save()

    def _finished(self, ok: bool) -> None:
        self.working = False
        self.reload()
        self.draw.set_busy(False)
        if ok:
            self.changed.emit()

    def _export(self) -> None:
        cmd = C.asset_export(self.project, f"{self.rec.asset_class}/{self.rec.name}", self.rec.settings.size)
        job = Job(f"export {self.rec.name}", [cmd])

        def done(ok: bool) -> None:
            if ok:
                out = next((ln.split("exported ", 1)[1] for ln in job.output if ln.startswith("exported ")), "")
                from ggs.ui.errors import show_saved

                show_saved(self, "Exported", Path(out) if out else None)

        job.done = done
        run_on_stage(self.queue, job, self.draw.stage)


class CharacterWorkspace(ItemWorkspace):
    """A character: the body & pose editor, "Use this design", and the Sliced version and Animate tabs."""

    def __init__(self, queue: JobQueue, project: Path, item_dir: Path, styles: list[str], models: list[str]):
        super().__init__(queue, project, item_dir, styles, models)
        from ggs.items import CharacterSettings
        from ggs.ui.character_editor import BodyPoseEditor
        from ggs.ui.character_page import AnimateTab, SlicedTab

        self.editor = BodyPoseEditor(self.rec.settings.character or CharacterSettings(), self.project)
        self.editor.changed.connect(self._character_changed)
        self.left.addWidget(self.editor)
        self.tab_group = QButtonGroup(self)
        for i, label in enumerate(("Design", "Sliced Version", "Animation")):
            b = QPushButton(label)
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setChecked(i == 0)
            self.tab_group.addButton(b, i)
            self.tabs_box.addWidget(b)
        self.tab_group.idClicked.connect(self.show_tab)
        self.sliced = SlicedTab(self)
        self.animate = AnimateTab(self)
        self.pages.addWidget(self.sliced)
        self.pages.addWidget(self.animate)
        # "Use this design" and what opens after it
        self.accept_btn = QPushButton("Accept Design")
        self.accept_btn.setObjectName("primary")
        set_icon(self.accept_btn, "accept")
        self.accept_btn.setToolTip("Keep this drawing as the character. Then you can make its sliced version, "
                                   "animate it and export it.")
        self.accept_btn.clicked.connect(self._accept)
        self.kept = QLabel("Accepted Design")
        self.kept.setStyleSheet("color: #3fb27f; font-weight: 600;")
        self.go_sliced = QPushButton("Sliced Version")
        set_icon(self.go_sliced, "layout")
        self.go_sliced.clicked.connect(lambda: self.show_tab(1))
        self.go_anim = QPushButton("Animate")
        set_icon(self.go_anim, "animations")
        self.go_anim.clicked.connect(lambda: self.show_tab(2))
        for w in (self.accept_btn, self.kept, self.go_sliced, self.go_anim):
            self.draw.extra.addWidget(w)
        self.draw.extra.addStretch(1)
        self._ready = True
        self.refresh_buttons()

    def refresh_buttons(self) -> None:
        super().refresh_buttons()
        if not getattr(self, "_ready", False):
            return
        has = self.rec.get_version() is not None and bool(self.rec.get_version().file)
        accepted = self.rec.accepted is not None
        on_design = accepted and self.rec.current == self.rec.accepted
        self.accept_btn.setVisible(has and not on_design)
        self.accept_btn.setEnabled(not self.working)
        self.accept_btn.setText("Accept Design")
        self.kept.setVisible(on_design)
        self.go_sliced.setVisible(accepted)
        self.go_anim.setVisible(accepted)
        for i in (1, 2):
            b = self.tab_group.button(i)
            b.setEnabled(accepted)
            b.setToolTip("" if accepted else "Press 'Use this design' first")
        self.export_btn.setToolTip("Export the character's design (one picture + skeleton.json)" if accepted
                                   else "Export the drawing on screen")

    def show_tab(self, i: int) -> None:
        self.tab_group.button(i).setChecked(True)
        self.pages.setCurrentIndex(i)
        self.editor.setVisible(i == 0)
        if i == 1:
            self.sliced.refresh()
        elif i == 2:
            self.animate.refresh()

    def _character_changed(self, cs) -> None:
        self.rec.settings.character = cs
        self._save()
        if self.rec.shown():
            self.draw.caption.setText(self.draw.caption.text().split("\n")[0] +
                                      "\nPose or body changed: press Regenerate or Draw new to draw it that way.")

    def _accept(self) -> None:
        job = Job(f"accept {self.rec.name}", [C.asset_accept(self.project, f"character/{self.rec.name}")])
        job.done = lambda ok: ok and (self.reload(), self.changed.emit())
        run_on_stage(self.queue, job, self.draw.stage)
        self.draw.stage.set_step("Saving the design...")
