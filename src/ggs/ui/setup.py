"""First-run model setup: which AI models are ready, download the missing ones, or point at a folder that
already has them (e.g. ComfyUI's models folder). Opens on start when a required model is missing, and from
Settings."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ggs.ui import commands as C
from ggs.ui.jobs import Job, JobQueue

PROGRESS = re.compile(r"@progress (\d+)")


def pack_status() -> list[tuple[str, object, bool]]:
    """(name, pack, ready) for every model pack."""
    from ggs.pipeline.download import load_packs, pack_ready

    return [(n, p, pack_ready(p)) for n, p in load_packs().items()]


def missing_required() -> bool:
    try:
        return any(p.required and not ok for _, p, ok in pack_status())
    except Exception:  # noqa: BLE001 - a broken manifest must not stop the app from opening
        return False


class PackRow(QFrame):
    def __init__(self, name: str, pack, ready: bool):
        super().__init__()
        self.name, self.pack = name, pack
        self.setObjectName("card")
        lay = QHBoxLayout(self)
        self.pick = QCheckBox()
        self.pick.setChecked(not ready and pack.required)
        self.pick.setEnabled(not ready)
        lay.addWidget(self.pick)
        text = QVBoxLayout()
        t = QLabel(f"<b>{pack.label}</b>")
        why = QLabel(f"{pack.why}  ({pack.size_gb:.1f} GB{'' if pack.required else ', optional'})")
        why.setObjectName("muted")
        why.setWordWrap(True)
        text.addWidget(t)
        text.addWidget(why)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.hide()
        text.addWidget(self.bar)
        lay.addLayout(text, 1)
        self.state = QLabel()
        self.state.setObjectName("chip")
        lay.addWidget(self.state, 0, Qt.AlignmentFlag.AlignTop)
        self.set_ready(ready)

    def set_ready(self, ready: bool) -> None:
        self.ready = ready
        self.state.setText("Ready" if ready else "Missing")
        self.pick.setEnabled(not ready)
        if ready:
            self.pick.setChecked(False)
            self.bar.hide()


class ModelSetupDialog(QDialog):
    def __init__(self, queue: JobQueue, parent=None):
        super().__init__(parent)
        self.queue = queue
        self.setWindowTitle("AI models")
        self.setMinimumWidth(640)
        lay = QVBoxLayout(self)
        h = QLabel("Set up the AI models")
        h.setObjectName("h1")
        lay.addWidget(h)
        hint = QLabel("GGS draws with AI models that run on your own graphics card. They are large, so they are "
                      "not part of the installer: download them once here (they go to your user folder and can "
                      "be resumed if the connection drops), or point GGS at a folder that already has them.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        lay.addWidget(hint)
        self.rows: list[PackRow] = []
        for name, pack, ready in pack_status():
            row = PackRow(name, pack, ready)
            row.pick.toggled.connect(self._update)
            self.rows.append(row)
            lay.addWidget(row)
        self.note = QLabel()
        self.note.setObjectName("muted")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        btns = QHBoxLayout()
        have = QPushButton("I already have them...")
        have.setToolTip("Choose a folder with the model files, for example ComfyUI's 'models' folder")
        have.clicked.connect(self._add_folder)
        btns.addWidget(have)
        btns.addStretch(1)
        self.later = QPushButton("Later")
        self.later.clicked.connect(self.reject)
        self.go = QPushButton("Download")
        self.go.setObjectName("primary")
        self.go.clicked.connect(self._download)
        btns.addWidget(self.later)
        btns.addWidget(self.go)
        lay.addLayout(btns)
        self._update()

    def _update(self) -> None:
        picked = [r for r in self.rows if r.pick.isChecked()]
        gb = sum(r.pack.size_gb for r in picked)
        self.go.setEnabled(bool(picked) and not self.queue.busy)
        self.go.setText(f"Download {gb:.1f} GB" if picked else "Download")
        if all(r.ready for r in self.rows if r.pack.required):
            self.note.setText("Everything needed to draw is ready.")
            self.later.setText("Close")

    def refresh(self) -> None:
        ready = {n: ok for n, _, ok in pack_status()}
        for r in self.rows:
            r.set_ready(ready.get(r.name, False))
        self._update()

    def _add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder with AI models (e.g. ComfyUI/models)")
        if d:
            from pathlib import Path

            self.queue.submit(Job("add models folder", [C.models_add_dir(Path(d))], done=lambda ok: self.refresh()))

    def _download(self) -> None:
        self.go.setEnabled(False)
        for r in [r for r in self.rows if r.pick.isChecked()]:
            r.bar.setValue(0)
            r.bar.show()
            job = Job(f"download {r.name}", [C.models_fetch(r.name)])

            def on_line(line: str, r=r) -> None:
                m = PROGRESS.search(line)
                if m:
                    r.bar.setValue(int(m.group(1)))

            def done(ok: bool, r=r, job=job) -> None:
                self.refresh()
                if not ok:
                    from ggs.ui.errors import friendly_error

                    self.note.setText(f"{r.pack.label}: " + friendly_error("\n".join(job.output)))

            job.on_line, job.done = on_line, done
            self.queue.submit(job)
        self.note.setText("Downloading... you can close this window; the download continues while GGS is open.")
