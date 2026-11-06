"""A small paint editor for touch-ups: brush, eraser, colour picker, fill, undo/redo, zoom.

Used on the sliced version's pieces and on the picture a bend rig is made from. "Open in my image editor" hands
the file to the system's default app (Krita, Paint.NET...) for anything bigger; "Reload" picks the changes up.
Parts cannot be moved here (that would break the joints).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ggs.ui.icons import icon, set_icon

UNDO = 40


class PaintCanvas(QWidget):
    changed = Signal()
    picked = Signal(QColor)

    def __init__(self, image: QImage):
        super().__init__()
        self.image = image.convertToFormat(QImage.Format.Format_ARGB32)
        self.tool = "brush"
        self.colour = QColor(0, 0, 0)
        self.size = 6
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.undo: deque[QImage] = deque(maxlen=UNDO)
        self.redo: list[QImage] = []
        self.last: QPointF | None = None
        self.panning: QPoint | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(400, 400)
        self.cursor_pos: QPointF | None = None

    def fit(self) -> None:
        r = self.rect()
        self.zoom = min(r.width() / max(self.image.width(), 1), r.height() / max(self.image.height(), 1)) * 0.95
        self.pan = QPointF((r.width() - self.image.width() * self.zoom) / 2,
                           (r.height() - self.image.height() * self.zoom) / 2)
        self.update()

    def to_image(self, p: QPointF) -> QPointF:
        return QPointF((p.x() - self.pan.x()) / self.zoom, (p.y() - self.pan.y()) / self.zoom)

    def _snapshot(self) -> None:
        self.undo.append(self.image.copy())
        self.redo.clear()

    def do_undo(self) -> None:
        if self.undo:
            self.redo.append(self.image.copy())
            self.image = self.undo.pop()
            self.changed.emit()
            self.update()

    def do_redo(self) -> None:
        if self.redo:
            self.undo.append(self.image.copy())
            self.image = self.redo.pop()
            self.changed.emit()
            self.update()

    def _stroke(self, a: QPointF, b: QPointF) -> None:
        p = QPainter(self.image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.tool == "eraser":
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        pen = QPen(self.colour, self.size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawLine(a, b)
        p.end()
        self.changed.emit()

    def _fill(self, at: QPointF, tolerance: int = 32) -> None:
        """Flood-fill the area of similar colour under ``at`` (flat game art: a sleeve, a boot)."""
        from scipy import ndimage

        x, y = int(at.x()), int(at.y())
        if not (0 <= x < self.image.width() and 0 <= y < self.image.height()):
            return
        ptr = self.image.constBits()
        a = np.frombuffer(ptr, np.uint8).reshape(self.image.height(), self.image.bytesPerLine() // 4, 4)
        a = a[:, :self.image.width()].astype(np.int16)  # BGRA
        seed = a[y, x]
        near = (np.abs(a - seed).max(axis=2) <= tolerance)
        lab, _ = ndimage.label(near)
        region = lab == lab[y, x]
        out = a.astype(np.uint8).copy()
        c = self.colour
        out[region] = (c.blue(), c.green(), c.red(), 255)
        img = QImage(out.data, out.shape[1], out.shape[0], out.shape[1] * 4, QImage.Format.Format_ARGB32).copy()
        self.image = img
        self.changed.emit()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(60, 62, 70))
        r = QRectF(self.pan.x(), self.pan.y(), self.image.width() * self.zoom, self.image.height() * self.zoom)
        tile = 12
        p.save()
        p.setClipRect(r)
        p.fillRect(r, QColor(255, 255, 255))
        for yy in range(int(r.top()), int(r.bottom()), tile):
            for xx in range(int(r.left()), int(r.right()), tile):
                if ((xx - int(r.left())) // tile + (yy - int(r.top())) // tile) % 2:
                    p.fillRect(xx, yy, tile, tile, QColor(232, 232, 236))
        p.restore()
        p.drawImage(r, self.image)
        if self.cursor_pos is not None and self.tool in ("brush", "eraser"):
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(self.cursor_pos, self.size * self.zoom / 2, self.size * self.zoom / 2)

    def wheelEvent(self, e) -> None:
        old = self.to_image(e.position())
        self.zoom = max(0.1, min(16.0, self.zoom * (1.15 if e.angleDelta().y() > 0 else 1 / 1.15)))
        self.pan = QPointF(e.position().x() - old.x() * self.zoom, e.position().y() - old.y() * self.zoom)
        self.update()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.MiddleButton:
            self.panning = e.position().toPoint()
            return
        at = self.to_image(e.position())
        if e.button() == Qt.MouseButton.RightButton or self.tool == "picker":
            x, y = int(at.x()), int(at.y())
            if 0 <= x < self.image.width() and 0 <= y < self.image.height():
                self.picked.emit(self.image.pixelColor(x, y))
            return
        self._snapshot()
        if self.tool == "fill":
            self._fill(at)
            self.update()
            return
        self.last = at
        self._stroke(at, at)
        self.update()

    def mouseMoveEvent(self, e) -> None:
        self.cursor_pos = e.position()
        if self.panning is not None:
            d = e.position().toPoint() - self.panning
            self.pan += QPointF(d)
            self.panning = e.position().toPoint()
        elif self.last is not None:
            at = self.to_image(e.position())
            self._stroke(self.last, at)
            self.last = at
        self.update()

    def mouseReleaseEvent(self, _e) -> None:
        self.last = None
        self.panning = None


class ImageEditor(QWidget):
    """A PNG open for editing: tools, canvas, Open With (installed open-source editors, reloaded when saved
    there), Revert and Save. ``saved`` is emitted after Save."""

    saved = Signal()

    def __init__(self, path: Path | None = None, parent=None):
        super().__init__(parent)
        from PySide6.QtCore import QFileSystemWatcher
        from PySide6.QtWidgets import QMenu, QToolButton

        from ggs.ui.external import find_editors, open_with

        self.path = Path(path) if path else None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.setSpacing(2)

        def tool_button(key: str, tip: str, checkable: bool = False) -> QToolButton:
            b = QToolButton()
            b.setIcon(icon(key))
            b.setIconSize(QSize(18, 18))
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setAutoRaise(True)
            return b

        self.tools = QButtonGroup(self)
        for i, (key, tip) in enumerate((("brush", "Brush (paint)"), ("eraser", "Eraser (erase to transparent)"),
                                        ("fill", "Fill (an area of one colour)"),
                                        ("picker", "Colour Picker (or right-click)"))):
            b = tool_button(key, tip, checkable=True)
            b.setProperty("tool", key)
            b.setChecked(i == 0)
            self.tools.addButton(b, i)
            bar.addWidget(b)
        bar.addSpacing(8)
        self.swatch = QPushButton()
        self.swatch.setFixedSize(26, 26)
        self.swatch.setToolTip("Colour")
        self.swatch.clicked.connect(self._choose)
        bar.addWidget(self.swatch)
        self.size = QSlider(Qt.Orientation.Horizontal)
        self.size.setRange(1, 60)
        self.size.setValue(6)
        self.size.setFixedWidth(90)
        self.size.setToolTip("Brush Size")
        bar.addWidget(self.size)
        bar.addSpacing(8)
        for key, tip, fn in (("undo", "Undo (Ctrl+Z)", lambda: self.canvas.do_undo()),
                             ("redo", "Redo (Ctrl+Y)", lambda: self.canvas.do_redo()),
                             ("fit", "Fit to Window", lambda: self.canvas.fit())):
            b = tool_button(key, tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch(1)
        self.open_with = QToolButton()
        set_icon(self.open_with, "open_with")
        self.open_with.setText("Open With")
        self.open_with.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.open_with.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.open_with)
        for name, exe in find_editors():
            menu.addAction(name, lambda exe=exe: self.path and open_with(exe, self.path))
        menu.addAction("Default App", lambda: self.path and open_with(None, self.path))
        self.open_with.setMenu(menu)
        self.open_with.setToolTip("Edit in an installed editor; changes saved there are loaded here")
        bar.addWidget(self.open_with)
        lay.addLayout(bar)
        self.revert = QPushButton("Revert")
        set_icon(self.revert, "revert")
        self.revert.clicked.connect(self.reload)
        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("primary")
        set_icon(self.save_btn, "save")
        self.save_btn.clicked.connect(self.save)
        foot = QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(self.revert)
        foot.addWidget(self.save_btn)
        self.canvas = PaintCanvas(QImage(str(self.path)) if self.path else QImage(8, 8, QImage.Format.Format_ARGB32))
        self.canvas.picked.connect(self._set_colour)
        self.tools.idClicked.connect(lambda i: setattr(self.canvas, "tool", self.tools.button(i).property("tool")))
        self.size.valueChanged.connect(lambda v: setattr(self.canvas, "size", v))
        lay.addWidget(self.canvas, 1)
        lay.addLayout(foot)
        self.watch = QFileSystemWatcher(self)
        self.watch.fileChanged.connect(lambda _p: self.reload())
        self._set_colour(QColor(0, 0, 0))
        if self.path:
            self.watch.addPath(str(self.path))

    def load(self, path: Path | None) -> None:
        if self.path:
            self.watch.removePath(str(self.path))
        self.path = Path(path) if path else None
        if self.path:
            self.watch.addPath(str(self.path))
        self.reload()

    def reload(self) -> None:
        img = QImage(str(self.path)) if self.path and self.path.exists() else QImage(8, 8, QImage.Format.Format_ARGB32)
        if img.isNull():
            return
        self.canvas.image = img.convertToFormat(QImage.Format.Format_ARGB32)
        self.canvas.undo.clear()
        self.canvas.redo.clear()
        self.canvas.fit()

    def save(self) -> None:
        if not self.path:
            return
        self.watch.removePath(str(self.path))
        self.canvas.image.save(str(self.path))
        self.watch.addPath(str(self.path))
        self.saved.emit()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.canvas.fit()

    def _set_colour(self, c: QColor) -> None:
        self.canvas.colour = QColor(c.red(), c.green(), c.blue())
        self.swatch.setStyleSheet(f"background: {self.canvas.colour.name()};")

    def _choose(self) -> None:
        c = QColorDialog.getColor(self.canvas.colour, self, "Colour")
        if c.isValid():
            self._set_colour(c)


class PaintDialog(QDialog):
    """``ImageEditor`` in a window (the warp's picture). ``saved`` is True after Save."""

    def __init__(self, path: Path, parent=None, title: str = "Edit Picture"):
        super().__init__(parent)
        self.path = Path(path)
        self.saved = False
        self.setWindowTitle(title)
        self.resize(1000, 760)
        lay = QVBoxLayout(self)
        self.editor = ImageEditor(self.path, self)
        self.editor.saved.connect(self._done)
        self.canvas = self.editor.canvas
        lay.addWidget(self.editor)

    def _done(self) -> None:
        self.saved = True
        self.accept()

    def _save(self) -> None:
        self.editor.save()
