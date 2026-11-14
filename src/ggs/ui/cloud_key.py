"""Google AI: the API key for the Nano Banana cloud models, and what each model costs."""

from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ggs.ui import commands as C
from ggs.ui.icons import set_icon
from ggs.ui.jobs import Job, JobQueue

KEY_URL = "https://aistudio.google.com/apikey"


def cloud_models() -> dict:
    from ggs.pipeline.models import load_pipelines

    return load_pipelines().cloud


def key_status() -> tuple[bool, str]:
    """(a key is available, one line about it)."""
    from ggs.pipeline.cloud import CloudError, key_source, mask, saved_key

    try:
        src = key_source()
        if src is None:
            return False, "No key yet."
        if src == "saved":
            return True, f"Key saved ({mask(saved_key() or '')}), encrypted for this Windows user."
        return True, f"Using the key in the {src} environment variable."
    except CloudError as exc:
        return False, str(exc)


def has_key() -> bool:
    return key_status()[0]


class GoogleKeyDialog(QDialog):
    def __init__(self, queue: JobQueue, parent=None, model: str | None = None):
        super().__init__(parent)
        self.queue = queue
        self.model = model or "nano-banana-2.1"
        self.setWindowTitle("Google AI")
        self.setMinimumWidth(600)
        lay = QVBoxLayout(self)
        t = QLabel("Google AI (Nano Banana)")
        t.setObjectName("h2")
        lay.addWidget(t)
        hint = QLabel("Draw on Google's servers instead of this computer's GPU. Needs your own API key from "
                      "Google AI Studio; Google bills that account per image. Cut-out and checks still run here.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        lay.addWidget(hint)

        row = QHBoxLayout()
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("Paste API Key")
        self.key.returnPressed.connect(self._save)
        get = QPushButton("Get a Key")
        set_icon(get, "open_with")
        get.setToolTip(KEY_URL)
        get.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(KEY_URL)))
        row.addWidget(self.key, 1)
        row.addWidget(get)
        lay.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        for c, h in enumerate(("Model", "Per Image", "")):
            lab = QLabel(h)
            lab.setObjectName("muted")
            grid.addWidget(lab, 0, c)
        for r, (name, s) in enumerate(cloud_models().items(), 1):
            grid.addWidget(QLabel(s.label), r, 0)
            low = min(s.price.values()) if s.price else None
            grid.addWidget(QLabel("" if low is None else f"from ${low:.3f}"), r, 1)
            note = QLabel(s.note)
            note.setObjectName("muted")
            grid.addWidget(note, r, 2)
        grid.setColumnStretch(2, 1)
        lay.addLayout(grid)
        foot = QLabel("List prices from Google; check Google's pricing page. Images carry Google's SynthID "
                      "watermark.")
        foot.setWordWrap(True)
        foot.setObjectName("muted")
        lay.addWidget(foot)

        btns = QHBoxLayout()
        self.remove = QPushButton("Remove Key")
        set_icon(self.remove, "delete")
        self.remove.clicked.connect(self._remove)
        self.check = QPushButton("Check")
        set_icon(self.check, "confirm")
        self.check.setToolTip("Asks Google whether the key works. Draws nothing, costs nothing.")
        self.check.clicked.connect(self._check)
        save = QPushButton("Save")
        save.setObjectName("primary")
        set_icon(save, "key")
        save.clicked.connect(self._save)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        btns.addWidget(self.remove)
        btns.addStretch(1)
        btns.addWidget(self.check)
        btns.addWidget(save)
        btns.addWidget(close)
        lay.addLayout(btns)
        self._refresh()

    def _refresh(self, extra: str = "") -> None:
        ok, text = key_status()
        self.status.setText(text + (f"\n{extra}" if extra else ""))
        self.check.setEnabled(ok)
        from ggs.pipeline.cloud import key_file

        self.remove.setEnabled(key_file().exists())

    def _save(self) -> None:
        from ggs.pipeline.cloud import CloudError, save_key

        text = self.key.text().strip()
        if not text:
            self.key.setFocus()
            return
        try:
            save_key(text)
        except CloudError as exc:
            self._refresh(str(exc))
            return
        self.key.clear()
        self._refresh()
        self._check()

    def _remove(self) -> None:
        from ggs.pipeline.cloud import clear_key

        clear_key()
        self._refresh()

    def _check(self) -> None:
        self.status.setText("Checking with Google...")
        job = Job("check Google key", [C.cloud_key_check(self.model)], exit_ok=(0, 1))

        def done(_ok: bool) -> None:
            lines = [ln.strip() for ln in job.output if ln.strip() and not ln.startswith("$ ")]
            said = next((ln for ln in reversed(lines) if ln.startswith(("OK", "ERROR"))), lines[-1] if lines else "")
            self._refresh(said.replace("ERROR: ", "Not working: ").replace("OK ", "Works: "))

        job.done = done
        self.queue.submit(job)


def ensure_key(queue: JobQueue, parent, model: str) -> bool:
    """True when a key is ready; otherwise ask for one (False when the user closes without a key)."""
    if has_key():
        return True
    GoogleKeyDialog(queue, parent, model).exec()
    return has_key()

