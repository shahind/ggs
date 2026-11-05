"""Settings: where the AI models are, and a check of this computer, in plain words."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ggs.ui import commands as C
from ggs.ui.jobs import Job, JobQueue
from ggs.ui.widgets import PathPicker


class SettingsDialog(QDialog):
    def __init__(self, queue: JobQueue, parent=None):
        super().__init__(parent)
        self.queue = queue
        self.setWindowTitle("Settings")
        self.setMinimumSize(620, 460)
        lay = QVBoxLayout(self)
        t = QLabel("Model Folders")
        t.setObjectName("h2")
        lay.addWidget(t)
        hint = QLabel("GGS looks for model files here. Add a folder you already have, for example ComfyUI's "
                      "'models' folder, so nothing is downloaded twice.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        lay.addWidget(hint)
        self.dirs = QListWidget()
        self.dirs.setFixedHeight(110)
        lay.addWidget(self.dirs)
        row = QHBoxLayout()
        self.pick = PathPicker(caption="Models folder")
        add = QPushButton("Add Folder...")
        add.clicked.connect(self._add)
        row.addWidget(self.pick, 1)
        row.addWidget(add)
        lay.addLayout(row)
        t2 = QLabel("System")
        t2.setObjectName("h2")
        lay.addWidget(t2)
        row2 = QHBoxLayout()
        check = QPushButton("Run Diagnostics")
        check.clicked.connect(self._check)
        models = QPushButton("Download Models...")
        models.clicked.connect(self._models)
        row2.addWidget(check)
        row2.addWidget(models)
        lay.addLayout(row2)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        lay.addWidget(self.report, 1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        lay.addWidget(close)
        self._load_dirs()

    def _load_dirs(self) -> None:
        from ggs.pipeline.models import model_dirs

        self.dirs.clear()
        for d in model_dirs():
            self.dirs.addItem(str(d))

    def _add(self) -> None:
        p = self.pick.path()
        if p:
            self.queue.submit(Job("add models folder", [C.models_add_dir(p)], done=lambda ok: self._load_dirs()))

    def _models(self) -> None:
        from ggs.ui.setup import ModelSetupDialog

        ModelSetupDialog(self.queue, self).exec()
        self._load_dirs()

    def _check(self) -> None:
        self.report.setPlainText("Checking...")
        job = Job("check", [C.doctor(), C.models_check()], exit_ok=(0, 1))

        def done(ok: bool) -> None:
            self.report.setPlainText(plain_report(job.output))

        job.done = done
        self.queue.submit(job)


def plain_report(lines: list[str]) -> str:
    """doctor + models check output -> a short readable report."""
    out = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("$ "):
            continue
        if "VRAM" in s or "GPU" in s or "profile" in s.lower():
            out.append(s)
        elif s.endswith(": OK") or ": OK " in s or s.startswith("matting"):
            out.append("✓ " + s.split(":")[0] + " is ready" if "OK" in s else "✗ " + s)
        elif "none of" in s or "MissingModel" in s:
            out.append("✗ missing: " + s.split(":")[0])
    return "\n".join(out) or "\n".join(lines[-20:])
