"""A small paint editor for touch-ups: brush, eraser, colour picker, fill, undo/redo, zoom.

Used on the sliced version's pieces and on the picture a bend rig is made from. "Open in my image editor" hands
the file to the system's default app (Krita, Paint.NET...) for anything bigger; "Reload" picks the changes up.
Parts cannot be moved here (that would break the joints).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

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


class PaintDialog(QDialog):
    """Edit a PNG in place. ``saved`` is True after Save."""

    def __init__(self, path: Path, parent=None, title: str = "Touch up"):
        super().__init__(parent)
        self.path = Path(path)
        self.saved = False
        self.setWindowTitle(title)
        self.resize(1000, 760)
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.tools = QButtonGroup(self)
        for i, (key, label, tip) in enumerate((
                ("brush", "Brush", "Paint with the colour"), ("eraser", "Eraser", "Make pixels transparent"),
                ("fill", "Fill", "Recolour an area of one colour (click inside it)"),
                ("picker", "Pick colour", "Click to take a colour from the picture (right-click works anytime)"))):
            b = QPushButton(label)
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setProperty("tool", key)
            b.setChecked(i == 0)
            self.tools.addButton(b, i)
            bar.addWidget(b)
        self.swatch = QPushButton("   ")
        self.swatch.setToolTip("Brush colour")
        self.swatch.clicked.connect(self._choose)
        bar.addWidget(self.swatch)
        bar.addWidget(QLabel("Size"))
        self.size = QSlider(Qt.Orientation.Horizontal)
        self.size.setRange(1, 60)
        self.size.setValue(6)
        self.size.setFixedWidth(140)
        bar.addWidget(self.size)
        bar.addStretch(1)
        for label, fn in (("Undo", lambda: self.canvas.do_undo()), ("Redo", lambda: self.canvas.do_redo()),
                          ("Fit", lambda: self.canvas.fit())):
            b = QPushButton(label)
            b.clicked.connect(fn)
            bar.addWidget(b)
        lay.addLayout(bar)
        self.canvas = PaintCanvas(QImage(str(self.path)))
        self.canvas.picked.connect(self._set_colour)
        self.tools.idClicked.connect(lambda i: setattr(self.canvas, "tool", self.tools.button(i).property("tool")))
        self.size.valueChanged.connect(lambda v: setattr(self.canvas, "size", v))
        lay.addWidget(self.canvas, 1)
        tip = QLabel("Mouse wheel zooms, middle button moves the picture, right-click takes a colour.")
        tip.setObjectName("muted")
        lay.addWidget(tip)
        row = QHBoxLayout()
        ext = QPushButton("Open in my image editor")
        ext.setToolTip("Edit the file in your usual image app; press Reload here when you have saved it there")
        ext.clicked.connect(self._external)
        reload_ = QPushButton("Reload")
        reload_.clicked.connect(self._reload)
        row.addWidget(ext)
        row.addWidget(reload_)
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save")
        save.setObjectName("primary")
        save.clicked.connect(self._save)
        row.addWidget(cancel)
        row.addWidget(save)
        lay.addLayout(row)
        self._set_colour(QColor(0, 0, 0))

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.canvas.fit()

    def _set_colour(self, c: QColor) -> None:
        self.canvas.colour = QColor(c.red(), c.green(), c.blue())
        self.swatch.setStyleSheet(f"background: {self.canvas.colour.name()};")

    def _choose(self) -> None:
        c = QColorDialog.getColor(self.canvas.colour, self, "Brush colour")
        if c.isValid():
            self._set_colour(c)

    def _external(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path)))

    def _reload(self) -> None:
        self.canvas._snapshot()
        self.canvas.image = QImage(str(self.path)).convertToFormat(QImage.Format.Format_ARGB32)
        self.canvas.update()

    def _save(self) -> None:
        self.canvas.image.save(str(self.path))
        self.saved = True
        self.accept()
