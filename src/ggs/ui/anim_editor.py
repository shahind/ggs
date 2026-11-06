"""Live animation editor for a character: plays an animation as it will look, without writing any file.

Frames are drawn on demand at screen size (the warp renders a frame in milliseconds; the cut-out puppet in a
few more), so the preview plays at the clip's speed. Files are only written on export.

- Play / pause, a timeline, and a strip of key frames to jump to.
- Drag a hand, elbow, knee, foot or the head in a frame to turn that bone; the change fades into the frames
  around it. The touched-up animation is kept with the character (``ggs.rig.edits``).
- Joints mode (bend-the-drawing only): the design with its skeleton; drag a joint where it really is in the
  drawing and the rig is rebuilt.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ggs.ui.icons import set_icon

PREVIEW_HEIGHT = 360  # px per standing height in the editor
HANDLE = 7


def _qimage_premultiplied(pre: np.ndarray) -> QImage:
    a = np.ascontiguousarray((np.clip(pre, 0, 1) * 255).round().astype(np.uint8))
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 4 * w, QImage.Format.Format_RGBA8888_Premultiplied).copy()


def _qimage_pil(im) -> QImage:
    a = np.ascontiguousarray(np.asarray(im.convert("RGBA")))
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


class WarpRenderer:
    """Frames of a warp rig (``ggs.rig.skin``)."""

    joints_editable = True

    def __init__(self, skin_json: Path):
        from ggs.rig.skin import load_skin

        self.path = Path(skin_json)
        self.skin = load_skin(skin_json)
        self.rig = self.skin.rig
        self.scale = PREVIEW_HEIGHT / self.skin.height_px

    def prepare(self, clip):
        return self.skin.prepare(clip)

    def layout(self, clip) -> tuple[tuple[int, int], tuple[float, float]]:
        lo, hi = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
        step = max(1, clip.frame_count // 40)
        for f in range(0, clip.frame_count, step):
            start, ang = self.skin.pose(clip, f)
            for layer in self.skin.layers:
                d = self.skin.deform(layer, start, ang) * self.scale
                lo, hi = np.minimum(lo, d.min(0)), np.maximum(hi, d.max(0))
        pad = 0.08 * PREVIEW_HEIGHT
        size = (int(hi[0] - lo[0] + 2 * pad), int(hi[1] - lo[1] + 2 * pad))
        return size, (float(-lo[0] + pad), float(-lo[1] + pad))

    def frame(self, clip, f: int, size, offset) -> QImage:
        return _qimage_premultiplied(self.skin.render_frame(clip, f, self.scale, size, offset))

    def bones(self, clip, f: int, offset) -> dict[str, tuple[QPointF, QPointF]]:
        start, ang = self.skin.pose(clip, f)
        lengths = self.skin.lengths()
        out = {}
        for i, b in enumerate(self.skin.bones):
            s = start[i] * self.scale + offset
            L = lengths[b] * self.skin.height_px * self.scale
            e = s + L * np.array([math.cos(ang[i]), math.sin(ang[i])])
            out[b] = (QPointF(*s), QPointF(*e))
        return out

    def rest(self) -> tuple[QImage, dict[str, tuple[QPointF, QPointF]], float]:
        """The rig's own picture, its joints (picture pixels) and the picture's scale on screen."""
        from PIL import Image

        im = Image.fromarray(self.skin.source, "RGBA")
        return _qimage_pil(im), {b: (QPointF(*s), QPointF(*e)) for b, (s, e) in self.skin.joints.items()}, 1.0


class PuppetRenderer:
    """Frames of the cut-out puppet (the sliced version)."""

    joints_editable = False

    def __init__(self, character_yaml: Path):
        from ggs.rig.character import load_character

        self.character = load_character(character_yaml)
        self.rig = self.character.rig()
        self.puppet = self.character.puppet(self.rig)
        self.lengths = dict(self.character.spec.bone_lengths)

    def prepare(self, clip):
        return self.character.clip(clip)

    def _pose(self, clip, f: int):
        from ggs.rig.pose import bone_defs, pose_bones, root_scale

        bones = bone_defs(self.rig)
        return pose_bones(bones, clip, f, self.lengths, root_scale(bones, clip, self.lengths), True)

    def layout(self, clip):
        h = PREVIEW_HEIGHT
        xs, ys = [], []
        for f in range(0, clip.frame_count, max(1, clip.frame_count // 40)):
            for s, e in self._pose(clip, f).values():
                xs += [s[0], e[0]]
                ys += [s[1], e[1]]
        pad = 0.25
        x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
        return (int((x1 - x0) * h), int((y1 - y0) * h)), (float(-x0 * h), float(-y0 * h))

    def frame(self, clip, f: int, size, offset) -> QImage:
        return _qimage_pil(self.puppet.render_frame(clip, f, PREVIEW_HEIGHT, size, offset, True))

    def bones(self, clip, f: int, offset):
        h = PREVIEW_HEIGHT
        return {b: (QPointF(s[0] * h + offset[0], s[1] * h + offset[1]), QPointF(e[0] * h + offset[0],
                                                                                   e[1] * h + offset[1]))
                for b, (s, e) in self._pose(clip, f).items()}


class Canvas(QWidget):
    """Draws one image (scaled to fit) with the skeleton over it; joint ends can be dragged."""

    dragged = Signal(str, QPointF)  # handle key, new position in image pixels (on release)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(260, 260)
        self.setMouseTracking(True)
        self.image: QImage | None = None
        self.bones: dict[str, tuple[QPointF, QPointF]] = {}
        self.handles: dict[str, QPointF] = {}  # key -> image px
        self.show_bones = True
        self.drag: str | None = None
        self.drag_pos: QPointF | None = None
        self.text = ""

    def set_content(self, image: QImage | None, bones=None, handles=None, text: str = "") -> None:
        self.image, self.bones, self.handles, self.text = image, bones or {}, handles or {}, text
        self.update()

    def _fit(self) -> tuple[float, QPointF]:
        if self.image is None or self.image.isNull():
            return 1.0, QPointF(0, 0)
        r = self.rect().adjusted(8, 8, -8, -8)
        k = min(r.width() / self.image.width(), r.height() / self.image.height())
        return k, QPointF(r.x() + (r.width() - self.image.width() * k) / 2,
                          r.y() + (r.height() - self.image.height() * k) / 2)

    def to_image(self, p: QPointF) -> QPointF:
        k, o = self._fit()
        return QPointF((p.x() - o.x()) / k, (p.y() - o.y()) / k)

    def _at(self, pos: QPointF) -> str | None:
        k, o = self._fit()
        best, d0 = None, (HANDLE + 4) ** 2
        for key, p in self.handles.items():
            q = QPointF(o.x() + p.x() * k, o.y() + p.y() * k)
            d = (q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2
            if d < d0:
                best, d0 = key, d
        return best

    def paintEvent(self, _e) -> None:
        from ggs.ui.theme import ACCENT, INK, MUTED, NOTICE

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        if self.image is None or self.image.isNull():
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self.text)
            return
        k, o = self._fit()
        p.drawImage(QRectF(o.x(), o.y(), self.image.width() * k, self.image.height() * k), self.image)
        if self.show_bones:
            line = QColor(INK)
            line.setAlpha(170)
            p.setPen(QPen(line, 2))
            for s, e in self.bones.values():
                p.drawLine(QPointF(o.x() + s.x() * k, o.y() + s.y() * k), QPointF(o.x() + e.x() * k, o.y() + e.y() * k))
            for key, h in self.handles.items():
                pt = self.drag_pos if key == self.drag and self.drag_pos is not None else h
                q = QPointF(o.x() + pt.x() * k, o.y() + pt.y() * k)
                p.setPen(QPen(QColor(INK), 1.5))
                p.setBrush(QColor(NOTICE if key == self.drag else ACCENT))
                p.drawEllipse(q, HANDLE, HANDLE)
        if self.text:
            p.setPen(QColor(MUTED))
            p.drawText(self.rect().adjusted(8, 4, -8, -4), int(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft),
                       self.text)

    def mousePressEvent(self, e) -> None:
        key = self._at(e.position())
        if key and e.button() == Qt.MouseButton.LeftButton:
            self.drag, self.drag_pos = key, self.to_image(e.position())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()

    def mouseMoveEvent(self, e) -> None:
        if self.drag:
            self.drag_pos = self.to_image(e.position())
            self.update()
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._at(e.position()) else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, _e) -> None:
        if self.drag and self.drag_pos is not None:
            key, pos = self.drag, self.drag_pos
            self.drag, self.drag_pos = None, None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.dragged.emit(key, pos)
        self.update()


TIPS = {"hand", "forearm", "shin", "foot", "head"}  # bones whose end is a handle in animation mode


class AnimEditor(QWidget):
    """Plays a clip on a renderer; touch-ups make an edited copy of the clip (``edited`` emits its path)."""

    edited = Signal(object)  # Path of the saved edited clip (or None after "undo all")
    rig_changed = Signal()

    def __init__(self):
        super().__init__()
        self.renderer = None
        self.clip = None
        self.clip_name = ""
        self.save_dir: Path | None = None
        self.frame_i = 0
        self.lead_n = 0
        self.base_clip = None
        self.size = (1, 1)
        self.offset = (0.0, 0.0)
        self.cache: dict[int, QImage] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.modes = QButtonGroup(self)
        for i, t in enumerate(("Animation", "Joints")):
            b = QPushButton(t)
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setChecked(i == 0)
            self.modes.addButton(b, i)
            top.addWidget(b)
        self.modes.idClicked.connect(lambda _i: self._show())
        top.addStretch(1)
        self.confirm = QPushButton("Confirm Joints")
        self.confirm.setObjectName("primary")
        set_icon(self.confirm, "confirm")
        self.confirm.setToolTip("Use these joint positions and continue to the animations")
        self.confirm.clicked.connect(self._confirm)
        top.addWidget(self.confirm)
        self.bones_btn = QPushButton("Skeleton")
        self.bones_btn.setCheckable(True)
        set_icon(self.bones_btn, "skeleton")
        self.bones_btn.setChecked(True)
        self.bones_btn.toggled.connect(self._toggle_bones)
        top.addWidget(self.bones_btn)
        self.undo = QPushButton("Reset Edits")
        set_icon(self.undo, "revert")
        self.undo.setToolTip("Discard the touch-ups of this animation")
        self.undo.clicked.connect(self._undo)
        top.addWidget(self.undo)
        lay.addLayout(top)
        self.canvas = Canvas()
        self.canvas.dragged.connect(self._dragged)
        lay.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.play = QPushButton("Play")
        self.play.setCheckable(True)
        set_icon(self.play, "play")
        self.play.toggled.connect(self._play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._seek)
        self.where = QLabel("")
        self.where.setObjectName("muted")
        row.addWidget(self.play)
        row.addWidget(self.slider, 1)
        row.addWidget(self.where)
        lay.addLayout(row)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._show()

    # ------------------------------------------------------------------ content

    def set_renderer(self, renderer) -> None:
        self.renderer = renderer
        editable = bool(renderer and renderer.joints_editable)
        self.modes.button(1).setVisible(editable)
        if not editable:
            self.modes.button(0).setChecked(True)
        elif not getattr(renderer.skin, "confirmed", False):  # joints first: check them before animating
            self.modes.button(1).setChecked(True)
        if self.clip is not None:
            self.set_clip(self.clip, self.clip_name, self.save_dir)
        else:
            self._show()

    def set_clip(self, clip, name: str, save_dir: Path | None) -> None:
        self.base_clip = clip
        self.lead_n = 0
        if self.renderer is not None and clip is not None:
            clip = self.renderer.prepare(clip)  # tracks for the layout's extra bones
        self.clip, self.clip_name, self.save_dir = clip, name, save_dir
        self.cache.clear()
        if self.renderer is not None and clip is not None:
            self.size, self.offset = self.renderer.layout(clip)
            self.slider.blockSignals(True)
            self.slider.setRange(0, clip.frame_count - 1)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            self.frame_i = 0
            self.timer.setInterval(max(15, round(1000 / clip.fps)))
        self._show()

    # ------------------------------------------------------------------ drawing

    def _frame(self, f: int) -> QImage:
        if f not in self.cache:
            self.cache[f] = self.renderer.frame(self.clip, f, self.size, self.offset)
        return self.cache[f]

    def _show(self) -> None:
        joints_mode = self.modes.checkedId() == 1
        for w in (self.play, self.slider, self.where, self.undo):
            w.setVisible(not joints_mode)
        self.confirm.setVisible(joints_mode)
        if joints_mode and self.play.isChecked():
            self.play.setChecked(False)
        if self.renderer is None:
            self.canvas.set_content(None, text="Nothing to show yet.")
            return
        if joints_mode:
            img, bones, _ = self.renderer.rest()
            handles = {}
            for b, (s, e) in bones.items():
                handles.setdefault(b, s)
                if not any(c for c, (cs, _ce) in bones.items() if (cs - e).manhattanLength() < 1.0):
                    handles[f"{b}:end"] = e
            self.canvas.set_content(img, bones, self._unique(handles))
            return
        if self.clip is None:
            self.canvas.set_content(None, text="Select an animation")
            return
        f = self.frame_i
        bones = self.renderer.bones(self.clip, f, np.array(self.offset))
        handles = {b: e for b, (_s, e) in bones.items() if b.split("_")[0] in TIPS}
        self.canvas.set_content(self._frame(f), bones, handles)
        self.where.setText(f"{f + 1} / {self.clip.frame_count}")
        self.undo.setEnabled(self._saved_path() is not None and self._saved_path().exists())

    @staticmethod
    def _unique(handles: dict) -> dict:
        out, seen = {}, []
        for k, p in handles.items():
            if any((p - q).manhattanLength() < 1.0 for q in seen):
                continue
            seen.append(p)
            out[k] = p
        return out

    def _toggle_bones(self, on: bool) -> None:
        self.canvas.show_bones = on
        self.canvas.update()

    # ------------------------------------------------------------------ playback

    def _play(self, on: bool) -> None:
        self.play.setText("Pause" if on else "Play")
        set_icon(self.play, "pause" if on else "play")
        self.timer.start() if on and self.clip is not None else self.timer.stop()

    def _tick(self) -> None:
        if self.clip is None:
            return
        self.slider.setValue((self.frame_i + 1) % self.clip.frame_count)

    def _seek(self, f: int) -> None:
        self.frame_i = f
        self._show()

    # ------------------------------------------------------------------ editing

    def _saved_path(self) -> Path | None:
        return self.save_dir / f"{self.clip_name}.json" if self.save_dir and self.clip_name else None

    def _dragged(self, key: str, pos: QPointF) -> None:
        if self.modes.checkedId() == 1:
            self._move_joint(key, pos)
            return
        from ggs.rig.edits import turn_bone

        bones = self.renderer.bones(self.clip, self.frame_i, np.array(self.offset))
        s, e = bones[key]
        a0 = math.atan2(e.y() - s.y(), e.x() - s.x())
        a1 = math.atan2(pos.y() - s.y(), pos.x() - s.x())
        delta = math.degrees(a1 - a0)
        delta = (delta + 180.0) % 360.0 - 180.0
        parents = {b.name: b.parent for b in self.renderer.rig.bones}
        f = self.frame_i - self.lead_n  # edits go into the animation itself, not the lead-in
        if f < 0:
            return
        # screen angles grow clockwise like the clip's, so the turn is the same number of degrees
        edited = turn_bone(self.renderer.prepare(self.base_clip), parents, key, f, delta)
        path = self._saved_path()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            edited.save(path)
        keep = self.frame_i
        self.set_clip(edited, self.clip_name, self.save_dir)
        self.slider.setValue(keep)
        self.edited.emit(path)

    def _confirm(self) -> None:
        skin = self.renderer.skin
        skin.confirmed = True
        skin.save(self.renderer.path.parent)
        self.modes.button(0).setChecked(True)
        self._show()

    def _undo(self) -> None:
        path = self._saved_path()
        if path is not None and path.exists():
            path.unlink()
        self.edited.emit(None)

    def _move_joint(self, key: str, pos: QPointF) -> None:
        from PySide6.QtWidgets import QApplication

        from ggs.rig.warp import move_joints

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            move_joints(self.renderer.path, {key: (pos.x(), pos.y())})
            self.set_renderer(WarpRenderer(self.renderer.path))
            self.rig_changed.emit()
        finally:
            QApplication.restoreOverrideCursor()
        self._show()
