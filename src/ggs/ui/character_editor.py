"""Body & pose editor: view, pose and body tiles, proportion sliders, and a live figure whose limbs can be dragged.

The figure is the pose guide the generator draws over, so what you see is the pose and proportions you get.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from ggs.ui.theme import ACCENT

BODY_TILES = [("standard", "Standard"), ("heroic", "Heroic"), ("chibi", "Chibi"), ("stocky", "Stocky"),
              ("lanky", "Lanky"), ("child", "Child")]
SLIDERS = [("head", "Head"), ("torso", "Body"), ("arms", "Arms"), ("legs", "Legs"), ("width", "Build")]
DRAG_BONES = ("hand", "forearm", "upper_arm", "shin", "thigh", "foot", "head")
PREVIEW = 320


def _small_tile(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("tile")
    b.setCheckable(True)
    b.setStyleSheet("padding: 4px 6px;")
    return b


class FigureView(QLabel):
    """The guide render; dragging near a limb's end turns that limb toward the mouse."""

    angle_changed = Signal(str, float)

    def __init__(self):
        super().__init__()
        self.setFixedSize(PREVIEW, PREVIEW)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: white; border-radius: 10px;")
        self.guide = None
        self.drag: str | None = None

    def show_guide(self, guide) -> None:
        self.guide = guide
        img = guide.image
        qimg = QImage(img.tobytes("raw", "RGBA"), img.width, img.height, QImage.Format.Format_RGBA8888).copy()
        pm = QPixmap(PREVIEW, PREVIEW)
        pm.fill(QColor(255, 255, 255))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(pm.rect(), qimg)
        p.setPen(QPen(QColor(ACCENT), 2))
        p.setBrush(QColor(ACCENT))
        k = PREVIEW / img.width
        for name, (_, (ex, ey)) in guide.spec.joints.items():
            if name.rsplit("_", 1)[0] in DRAG_BONES or name == "head":
                p.drawEllipse(QPointF(ex * k, ey * k), 4, 4)
        p.end()
        self.setPixmap(pm)

    def _to_img(self, pos) -> tuple[float, float]:
        k = self.guide.image.width / PREVIEW
        return pos.x() * k, pos.y() * k

    def mousePressEvent(self, e) -> None:
        if self.guide is None:
            return
        x, y = self._to_img(e.position())
        best, dist = None, 30 * self.guide.image.width / PREVIEW
        for name, (_, (ex, ey)) in self.guide.spec.joints.items():
            if name.rsplit("_", 1)[0] not in DRAG_BONES and name != "head":
                continue
            d = math.hypot(ex - x, ey - y)
            if d < dist:
                best, dist = name, d
        self.drag = best

    def mouseMoveEvent(self, e) -> None:
        if self.drag is None or self.guide is None:
            return
        x, y = self._to_img(e.position())
        (sx, sy), _ = self.guide.spec.joints[self.drag]
        if math.hypot(x - sx, y - sy) < 3:
            return
        self.angle_changed.emit(self.drag, math.degrees(math.atan2(-(x - sx), y - sy)))

    def mouseReleaseEvent(self, e) -> None:
        self.drag = None


class BodyPoseEditor(QFrame):
    """Edits a CharacterSettings; ``changed`` fires (debounced) with the new settings."""

    changed = Signal(object)

    def __init__(self, settings):
        super().__init__()
        from ggs.rig.body import PRESETS

        self.setObjectName("panel")
        self.setFixedWidth(PREVIEW + 24)
        self.cs = settings.model_copy(deep=True)
        self.presets = PRESETS
        lay = QVBoxLayout(self)
        t = QLabel("Body & pose")
        t.setObjectName("h2")
        lay.addWidget(t)
        self.figure = FigureView()
        self.figure.angle_changed.connect(self._dragged)
        lay.addWidget(self.figure, 0, Qt.AlignmentFlag.AlignHCenter)
        hint = QLabel("Drag a hand, foot, elbow or knee to bend it.")
        hint.setObjectName("muted")
        lay.addWidget(hint)

        self.views = QButtonGroup(self)
        row = QHBoxLayout()
        for v, label in (("front", "Front"), ("side", "Side")):
            b = _small_tile(label)
            b.setProperty("view", v)
            b.setChecked(v == self.cs.view)
            self.views.addButton(b)
            row.addWidget(b)
        self.views.buttonClicked.connect(self._view)
        lay.addLayout(row)
        self.pose_box = QHBoxLayout()
        self.poses = QButtonGroup(self)
        self.poses.buttonClicked.connect(self._pose)
        lay.addLayout(self.pose_box)

        grid = QGridLayout()
        self.bodies = QButtonGroup(self)
        for i, (bid, label) in enumerate(BODY_TILES):
            b = _small_tile(label)
            b.setProperty("body", bid)
            b.setChecked(bid == self.cs.body_preset and not self.cs.body)
            self.bodies.addButton(b)
            grid.addWidget(b, i // 3, i % 3)
        self.bodies.buttonClicked.connect(self._body)
        lay.addLayout(grid)
        self.sliders = {}
        for key, label in SLIDERS:
            r = QHBoxLayout()
            lab = QLabel(label)
            lab.setFixedWidth(44)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(40, 220)
            sl.valueChanged.connect(self._slid)
            r.addWidget(lab)
            r.addWidget(sl, 1)
            lay.addLayout(r)
            self.sliders[key] = sl
        lay.addStretch(1)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self._render)
        self._fill_poses()
        self._set_sliders()
        self._render()

    # ------------------------------------------------------------------ state

    def body_shape(self):
        from ggs.rig.body import BodyShape

        return BodyShape(**self.cs.body) if self.cs.body else self.presets.get(self.cs.body_preset,
                                                                             self.presets["standard"])

    def _set_sliders(self) -> None:
        b = self.body_shape()
        for key, sl in self.sliders.items():
            sl.blockSignals(True)
            sl.setValue(round(getattr(b, key) * 100))
            sl.blockSignals(False)

    def _fill_poses(self) -> None:
        from ggs.rig.poses import LABELS, POSES

        for b in self.poses.buttons():
            self.poses.removeButton(b)
        while self.pose_box.count():
            it = self.pose_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        current = self.cs.pose if isinstance(self.cs.pose, str) else None
        for pid in POSES[self.cs.view]:
            b = _small_tile(LABELS.get(pid, pid))
            b.setProperty("pose", pid)
            b.setChecked(pid == current)
            self.poses.addButton(b)
            self.pose_box.addWidget(b)

    def _view(self, b) -> None:
        from ggs.rig.poses import DEFAULT

        self.cs.view = b.property("view")
        self.cs.pose = DEFAULT[self.cs.view]
        self._fill_poses()
        self._changed()

    def _pose(self, b) -> None:
        self.cs.pose = b.property("pose")
        self._changed()

    def _body(self, b) -> None:
        self.cs.body_preset = b.property("body")
        self.cs.body = {}
        self._set_sliders()
        self._changed()

    def _slid(self) -> None:
        b = self.body_shape().model_dump()
        b.update({k: sl.value() / 100 for k, sl in self.sliders.items()})
        b["name"], b["words"] = "custom", ""
        self.cs.body = b
        if self.bodies.checkedButton():
            self.bodies.setExclusive(False)
            self.bodies.checkedButton().setChecked(False)
            self.bodies.setExclusive(True)
        self._changed()

    def _dragged(self, bone: str, angle: float) -> None:
        from ggs.rig.poses import pose_angles

        angles = pose_angles(self.cs.view, self.cs.pose)
        delta = angle - angles[bone]
        angles[bone] = angle
        # children follow the bone they hang from, so the limb bends as one piece below the joint
        from ggs.rig.poses import rig_for

        rig = rig_for(self.cs.view)
        kids = [b.name for b in rig.bones if b.parent == bone]
        while kids:
            k = kids.pop()
            angles[k] = angles[k] + delta
            kids += [b.name for b in rig.bones if b.parent == k]
        self.cs.pose = angles
        if self.poses.checkedButton():
            self.poses.setExclusive(False)
            self.poses.checkedButton().setChecked(False)
            self.poses.setExclusive(True)
        self._changed()

    def _changed(self) -> None:
        self.timer.start()

    def _render(self) -> None:
        from ggs.items import character_guide

        _, guide, _ = character_guide(self.cs, size=PREVIEW * 2)
        self.figure.show_guide(guide)
        self.changed.emit(self.cs.model_copy(deep=True))
