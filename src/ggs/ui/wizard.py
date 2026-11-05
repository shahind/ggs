"""The "New" page: pick what to make from picture tiles, pick a size, name it, describe it."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ggs.ui.icons import icon
from ggs.ui.kinds import KINDS, SIZE_LABELS, kind


def tile(text: str, sub: str = "", w: int = 150, h: int = 92) -> QPushButton:
    b = QPushButton(f"{text}\n{sub}" if sub else text)
    b.setObjectName("tile")
    b.setCheckable(True)
    b.setFixedSize(w, h)
    return b


class NewItemDialog(QDialog):
    """Returns (kind id, name, size, description) via ``chosen``."""

    chosen = Signal(str, str, str, str)

    def __init__(self, parent=None, preselect: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("New")
        self.setMinimumWidth(720)
        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        title = QLabel("New Item")
        title.setObjectName("h2")
        lay.addWidget(title)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.kinds = QButtonGroup(self)
        self.kinds.setExclusive(True)
        for i, k in enumerate(KINDS):
            b = tile(k.label, "", 200, 64)
            b.setIcon(icon(k.icon))
            b.setIconSize(QSize(36, 36))
            b.setToolTip(k.hint)
            b.setProperty("kind", k.id)
            self.kinds.addButton(b)
            grid.addWidget(b, i // 4, i % 4)
        lay.addLayout(grid)
        self.hint = QLabel("")
        self.hint.setObjectName("muted")
        lay.addWidget(self.hint)
        self.size_title = QLabel("Size")
        self.size_title.setObjectName("h2")
        lay.addWidget(self.size_title)
        self.size_row = QHBoxLayout()
        self.sizes = QButtonGroup(self)
        lay.addLayout(self.size_row)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Name, e.g. Health potion")
        self.describe = QPlainTextEdit()
        self.describe.setPlaceholderText("Describe it, e.g. a small red health potion in a round glass bottle")
        self.describe.setFixedHeight(70)
        lay.addWidget(self.name)
        lay.addWidget(self.describe)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.ok = QPushButton("Create")
        self.ok.setObjectName("primary")
        self.ok.clicked.connect(self._accept)
        row.addWidget(cancel)
        row.addWidget(self.ok)
        lay.addLayout(row)
        self.kinds.buttonToggled.connect(self._kind_changed)
        self.name.textChanged.connect(self._validate)
        self.describe.textChanged.connect(self._validate)
        first = next((b for b in self.kinds.buttons() if b.property("kind") == (preselect or "sprite")),
                     self.kinds.buttons()[0])
        first.setChecked(True)
        self._validate()

    def kind_id(self) -> str:
        b = self.kinds.checkedButton()
        return b.property("kind") if b else "sprite"

    def size(self) -> str:
        b = self.sizes.checkedButton()
        return b.property("tile_size") if b else "medium"

    def _kind_changed(self, button, checked: bool) -> None:
        if not checked:
            return
        k = kind(button.property("kind"))
        self.hint.setText(k.hint)
        for b in self.sizes.buttons():
            self.sizes.removeButton(b)
        while self.size_row.count():  # clear the old tiles and stretch
            it = self.size_row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for s in k.sizes:
            short, sub = SIZE_LABELS[s]
            b = tile(short, sub, 96, 60)
            b.setProperty("tile_size", s)
            self.sizes.addButton(b)
            self.size_row.addWidget(b)
        self.size_row.addStretch(1)
        default = "medium" if "medium" in k.sizes else k.sizes[0]
        next(b for b in self.sizes.buttons() if b.property("tile_size") == default).setChecked(True)
        self.size_title.setVisible(len(k.sizes) > 1)
        for b in self.sizes.buttons():
            b.setVisible(len(k.sizes) > 1)
        self._validate()

    def _validate(self) -> None:
        self.ok.setEnabled(bool(self.name.text().strip()) and bool(self.describe.toPlainText().strip()))

    def _accept(self) -> None:
        self.chosen.emit(self.kind_id(), self.name.text().strip(), self.size(), self.describe.toPlainText().strip())
        self.accept()

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.ok.isEnabled() and not self.describe.hasFocus():
            self._accept()
        else:
            super().keyPressEvent(e)
