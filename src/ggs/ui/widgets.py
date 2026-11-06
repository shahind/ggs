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
    QStackedWidget,
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


class ImageStage(QWidget):
    """The big picture in a workspace: the image, a placeholder when there is none yet, or a loading placeholder
    (spinner, the step in plain words and a percentage) while it is being made."""

    def __init__(self, parent=None, empty: str = "Nothing here yet"):
        from PySide6.QtCore import QTimer

        super().__init__(parent)
        self.setMinimumSize(240, 240)
        from PySide6.QtWidgets import QSizePolicy

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.path: Path | None = None
        self.pix = None
        self.empty = empty
        self.busy = False
        self.pct = -1
        self.step = ""
        self.note = ""
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._spin)

    def set_image(self, path: Path | None, note: str = "") -> None:
        self.path = Path(path) if path else None
        self.pix = QPixmap(str(path)) if path and Path(path).exists() else None
        self.note = note
        self.update()

    def set_empty(self, text: str) -> None:
        self.empty = text
        self.update()

    def set_busy(self, on: bool, step: str = "") -> None:
        self.busy, self.step, self.pct = on, step or "Starting...", -1
        self.timer.start() if on else self.timer.stop()
        self.update()

    def set_step(self, text: str) -> None:
        self.step = text
        self.update()

    def set_progress(self, pct: int) -> None:
        self.pct = pct
        self.update()

    def _spin(self) -> None:
        self.angle = (self.angle + 9) % 360
        self.update()

    def paintEvent(self, _e) -> None:
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QFont, QPen

        from ggs.ui.theme import ACCENT, CARD, LINE, MUTED, NOTICE

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = self.rect().adjusted(4, 4, -4, -4)
        side = min(r.width(), r.height())
        box = QRectF(r.x() + (r.width() - side) / 2, r.y() + (r.height() - side) / 2, side, side)
        if self.pix is not None and not self.pix.isNull():
            tile = 14
            p.save()
            p.setClipRect(box)
            p.fillRect(box, QColor(255, 255, 255))
            for y in range(int(box.top()), int(box.bottom()), tile):
                for x in range(int(box.left()), int(box.right()), tile):
                    if ((x - int(box.left())) // tile + (y - int(box.top())) // tile) % 2:
                        p.fillRect(x, y, tile, tile, QColor(234, 234, 238))
            p.restore()
            scaled = self.pix.scaled(int(side) - 16, int(side) - 16, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
            p.drawPixmap(int(box.center().x() - scaled.width() / 2), int(box.center().y() - scaled.height() / 2),
                         scaled)
            if self.busy:
                p.fillRect(box, QColor(27, 28, 33, 190))
        else:
            p.setPen(QPen(QColor(LINE), 2, Qt.PenStyle.DashLine))
            p.setBrush(QColor(CARD))
            p.drawRoundedRect(box, 14, 14)
        if self.busy:
            c = box.center()
            rad = max(22.0, side * 0.08)
            ring = QRectF(c.x() - rad, c.y() - rad - 26, 2 * rad, 2 * rad)
            p.setPen(QPen(QColor(LINE), 6))
            p.drawEllipse(ring)
            pen = QPen(QColor(ACCENT), 6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            if self.pct >= 0:
                p.drawArc(ring, 90 * 16, -int(360 * 16 * self.pct / 100))
            else:
                p.drawArc(ring, -self.angle * 16, 90 * 16)
            f = QFont(self.font())
            f.setPointSize(16)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor("#ffffff"))
            p.drawText(QRectF(box.left(), ring.bottom() + 10, box.width(), 30), int(Qt.AlignmentFlag.AlignCenter),
                       f"{self.pct}%" if self.pct >= 0 else "")
            f.setPointSize(10)
            f.setBold(False)
            p.setFont(f)
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(box.left() + 10, ring.bottom() + 42, box.width() - 20, 40),
                       int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
                       self.step)
        elif self.pix is None or self.pix.isNull():
            p.setPen(QColor(MUTED))
            p.drawText(box.adjusted(30, 30, -30, -30), int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
                       self.empty)
        if self.note and not self.busy:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(NOTICE))
            strip = QRectF(box.left(), box.bottom() - 30, box.width(), 30)
            p.drawRect(strip)
            p.setPen(QColor("#1b1c21"))
            p.drawText(strip.adjusted(10, 0, -10, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap),
                       self.note)
        p.end()


class PromptEdit(QPlainTextEdit):
    """A short prompt box. Enter sends (``submitted``), Shift+Enter makes a new line."""

    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabChangesFocus(True)

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(e)


class CurrentStack(QStackedWidget):
    """A stacked widget sized by the page on show only, so a wide hidden page cannot widen the window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _i: self.updateGeometry())

    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()
