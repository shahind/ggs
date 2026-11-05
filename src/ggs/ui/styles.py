"""Styles section: the project's styles, and one form that trains a new one.

One button runs the whole chain as one job: import the images (rights confirmed), describe them (captions, with
a trigger word made from the style's name), train, install into the project. The progress shows the step in
plain words and a percentage (training reports every step); an error pops up with the details.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ggs.ui import commands as C
from ggs.ui.jobs import Job, JobQueue
from ggs.ui.widgets import PathPicker

LENGTHS = [("Quick", 500, "about 25 min on 8 GB: a first look"), ("Normal", None, "the model's default"),
           ("Long", 2500, "learns finer detail; about twice the normal time")]
STEP = re.compile(r"step\s+(\d+)/(\d+).*?eta ([\d.]+) min")
WEIGHTS = (2, 13, 83, 2)  # import, captions, training, install: shares of the bar


def trigger_for(name: str) -> str:
    """A rare word that stands for the style in prompts: GGS + the name in capitals."""
    return "GGS" + re.sub(r"[^A-Z0-9]", "", name.upper())[:16]


class StylesSection(QWidget):
    def __init__(self, queue: JobQueue, project: Path, models: list[str], base_model: str | None):
        super().__init__()
        from ggs.style.dataset import RIGHTS_WARNING

        self.queue, self.project = queue, Path(project)
        lay = QVBoxLayout(self)
        h = QLabel("Styles")
        h.setObjectName("h1")
        lay.addWidget(h)
        tip = QLabel("A style teaches the drawing model a look from 20-40 of your images. Pick it in Advanced on "
                     "any item afterwards.")
        tip.setObjectName("muted")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        row = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Styles in this project"))
        self.installed = QListWidget()
        left.addWidget(self.installed, 1)
        row.addLayout(left, 1)
        card = QFrame()
        card.setObjectName("panel")
        f = QFormLayout(card)
        t = QLabel("Train a new style")
        t.setObjectName("h2")
        f.addRow(t)
        self.src = PathPicker(caption="Folder with your images", placeholder="a folder with 20-40 images")
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. inkwash")
        f.addRow("Images", self.src)
        f.addRow("Style name", self.name)
        self.model = QComboBox()
        self.model.addItem(f"Project's ({base_model or 'flux2-klein-4b'})", None)
        for m in models:
            self.model.addItem(m, m)
        f.addRow("Drawing model", self.model)
        self.length = QButtonGroup(self)
        lrow = QHBoxLayout()
        for i, (label, _steps, why) in enumerate(LENGTHS):
            b = QPushButton(label)
            b.setObjectName("tile")
            b.setCheckable(True)
            b.setToolTip(why)
            b.setChecked(i == 1)
            self.length.addButton(b, i)
            lrow.addWidget(b)
        f.addRow("Training", lrow)
        warn = QLabel(RIGHTS_WARNING)
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #f0a040;")
        f.addRow(warn)
        self.rights = QCheckBox("I have the rights to these images")
        f.addRow(self.rights)
        self.go = QPushButton("Train style")
        self.go.setObjectName("primary")
        self.go.clicked.connect(self._train)
        f.addRow(self.go)
        self.step = QLabel("")
        self.step.setObjectName("muted")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(True)
        self.bar.setFormat("%p%")
        self.bar.setStyleSheet("QProgressBar { height: 18px; max-height: 18px; }")
        for w in (self.step, self.bar):
            w.setVisible(False)
            f.addRow(w)
        for w in (self.rights, self.name):
            (w.toggled if w is self.rights else w.textChanged).connect(self._ready)
        self.src.changed.connect(self._ready)
        row.addWidget(card, 2)
        lay.addLayout(row, 1)
        self.refresh()
        self._ready()

    def refresh(self) -> None:
        from ggs.style.package import list_styles

        self.installed.clear()
        try:
            names = sorted({m.name for m, _ in list_styles(self.project)})
        except Exception:  # noqa: BLE001 - a broken style must not break the page
            names = []
        for n in names:
            self.installed.addItem(n)
        if not names:
            self.installed.addItem("(none yet)")

    def _ready(self) -> None:
        self.go.setEnabled(bool(self.rights.isChecked() and self.name.text().strip() and self.src.path())
                           and not self.bar.isVisible())

    def _train(self) -> None:
        name = C.slug(self.name.text().strip(), 24)
        ds = self.project / "style_data" / name
        pkg = self.project / "style_data" / f"{name}.ggsstyle"
        steps = LENGTHS[self.length.checkedId()][1]
        job = Job(f"Train style {name}", [
            C.dataset_import(self.src.path(), ds, True),
            C.caption(ds, trigger_for(name)),
            C.train(ds, name, pkg, project=self.project, model=self.model.currentData(), steps=steps),
            C.style_install(pkg, self.project)])
        texts = ("Copying your images...", "Describing your images (a minute or two)...", "Training the style...",
                 "Adding the style to the project...")

        def pct(step: int, frac: float) -> int:
            return int(sum(WEIGHTS[:step]) + WEIGHTS[step] * frac)

        def on_start() -> None:
            self.step.setText(texts[0])

        def on_line(line: str) -> None:
            i = min(job._step, 3)
            m = STEP.search(line)
            if m:
                n, total, eta = int(m.group(1)), int(m.group(2)), float(m.group(3))
                self.bar.setValue(pct(2, n / total))
                self.step.setText(f"Training the style: step {n} of {total}, about {max(1, round(eta))} min left")
            elif not line.startswith("@"):
                self.bar.setValue(max(self.bar.value(), pct(i, 0.0)))
                if not self.step.text().startswith("Training the style: step") or i != 2:
                    self.step.setText(texts[i])

        def done(ok: bool) -> None:
            for w in (self.step, self.bar):
                w.setVisible(False)
            self._ready()
            self.refresh()
            if ok:
                from ggs.ui.errors import show_message

                show_message(self, f"The style '{name}' is ready. Pick it in Advanced on any item.")
            else:
                from ggs.ui.errors import show_error

                show_error(self, "\n".join(job.output[-80:]), "The style was not trained")

        job.on_start, job.on_line, job.done = on_start, on_line, done
        self.bar.setValue(0)
        self.step.setText("Waiting for the job before this one..." if self.queue.busy else texts[0])
        for w in (self.step, self.bar):
            w.setVisible(True)
        self.go.setEnabled(False)
        self.queue.submit(job)
