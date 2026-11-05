"""Small widgets shared by the app's pages."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QWidget,
)

THUMB = 220


def checker_pixmap(path: Path, size: int = THUMB, badge: str = "", ok: bool | None = None) -> QPixmap:
    """The image scaled into a square on a light checkerboard, so transparency shows; ``badge`` is drawn as a
    strip along the bottom (green when ``ok``, red when not)."""
    src = QPixmap(str(path))
    out = QPixmap(size, size)
    out.fill(QColor(255, 255, 255))
    p = QPainter(out)
    tile = 12
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            if (x // tile + y // tile) % 2:
                p.fillRect(x, y, tile, tile, QColor(232, 232, 236))
    if not src.isNull():
        scaled = src.scaled(size - 8, size - 8, Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        p.drawPixmap((size - scaled.width()) // 2, (size - scaled.height()) // 2, scaled)
    if badge:
        h = 22
        colour = QColor(40, 140, 70, 225) if ok else QColor(190, 50, 50, 225) if ok is False else QColor(60, 60,
                                                                                                         70, 210)
        p.fillRect(0, size - h, size, h, colour)
        p.setPen(QColor(255, 255, 255))
        p.drawText(6, size - h, size - 12, h, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                   badge)
    p.end()
    return out


class ImageGrid(QListWidget):
    """Thumbnails with a caption; double-click opens the file, a click emits ``picked``."""

    picked = Signal(str)

    def __init__(self, parent=None, thumb: int = THUMB):
        super().__init__(parent)
        self.thumb = thumb
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(thumb, thumb))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setSpacing(8)
        self.setWordWrap(True)
        self.setUniformItemSizes(True)
        self.setGridSize(QSize(thumb + 24, thumb + 64))
        self.itemDoubleClicked.connect(lambda it: QDesktopServices.openUrl(QUrl.fromLocalFile(it.data(256))))
        self.itemClicked.connect(lambda it: self.picked.emit(it.data(256)))

    def add(self, path: Path, caption: str, ok: bool | None = None) -> None:
        """``caption``: first line goes on the thumbnail's badge, the whole text in the tooltip and below."""
        badge = caption.splitlines()[-1] if caption else ""
        it = QListWidgetItem(QIcon(checker_pixmap(Path(path), self.thumb, badge[:40], ok)), caption.splitlines()[0])
        it.setData(256, str(path))
        it.setToolTip(f"{path}\n{caption}")
        self.insertItem(0, it)  # newest first


class LogView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setPlaceholderText("Job output appears here.")

    def append_line(self, text: str) -> None:
        self.appendPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class PathPicker(QWidget):
    """A line edit with a Browse button (folder or file)."""

    changed = Signal(str)

    def __init__(self, parent=None, *, folder: bool = True, caption: str = "Choose", filter: str = "",
                 placeholder: str = ""):
        super().__init__(parent)
        self.folder, self.caption, self.filter = folder, caption, filter
        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed.emit)
        btn = QPushButton("Browse...", self)
        btn.clicked.connect(self._browse)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def _browse(self) -> None:
        if self.folder:
            p = QFileDialog.getExistingDirectory(self, self.caption, self.edit.text())
        else:
            p, _ = QFileDialog.getOpenFileName(self, self.caption, self.edit.text(), self.filter)
        if p:
            self.edit.setText(p)

    def path(self) -> Path | None:
        t = self.edit.text().strip()
        return Path(t) if t else None

    def set_path(self, p) -> None:
        self.edit.setText("" if p is None else str(p))
