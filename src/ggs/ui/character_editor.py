"""Body & pose editor: view, pose and body tiles, proportion sliders, and a live figure whose limbs can be dragged.

The figure is the pose guide the generator draws over, so what you see is the pose and proportions you get.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QSize, Qt, QTimer, Signal
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
    QWidget,
)

from ggs.ui.icons import icon
from ggs.ui.theme import ACCENT

SLIDERS = [("head", "Head"), ("torso", "Body"), ("arms", "Arms"), ("legs", "Legs"), ("width", "Build")]
DRAG_BONES = ("hand", "forearm", "upper_arm", "shin", "thigh", "foot", "head")
PREVIEW = 320


def _section(title: str, on_save) -> QWidget:
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 6, 0, 0)
    lab = QLabel(title)
    lab.setStyleSheet("font-weight: 600;")
    save = QPushButton("Save\u2026")
    save.setIcon(icon("save"))
    save.setObjectName("flat")
    save.setToolTip(f"Save the current {title.lower()} to reuse it")
    save.clicked.connect(lambda: on_save())
    row.addWidget(lab)
    row.addStretch(1)
    row.addWidget(save)
    return w


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

    def __init__(self, settings, project=None):
        super().__init__()
        from ggs.rig.body import PRESETS

        self.project = project

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
            b.setIcon(icon(v))
            b.setIconSize(QSize(22, 22))
            b.setProperty("view", v)
            b.setChecked(v == self.cs.view)
            self.views.addButton(b)
            row.addWidget(b)
        self.views.buttonClicked.connect(self._view)
        lay.addLayout(row)
        lay.addWidget(_section("Pose", self._save_pose))
        self.pose_box = QGridLayout()
        self.poses = QButtonGroup(self)
        self.poses.buttonClicked.connect(self._pose)
        lay.addLayout(self.pose_box)
        lay.addWidget(_section("Body", self._save_body))
        self.body_box = QGridLayout()
        self.bodies = QButtonGroup(self)
        self.bodies.buttonClicked.connect(self._body)
        lay.addLayout(self.body_box)
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
        self._fill_bodies()
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

    def _clear(self, group: QButtonGroup, box: QGridLayout) -> None:
        for b in group.buttons():
            group.removeButton(b)
        while box.count():
            it = box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

    def _saved_tile(self, saved, kind: str) -> QPushButton:
        b = _small_tile(saved.name)
        b.setProperty("saved", saved)
        if saved.scope != "builtin":
            where = "for all projects" if saved.scope == "global" else "in this project"
            b.setToolTip(f"Saved {where}. Right-click to delete.")
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _p, sv=saved: self._delete(kind, sv))
        return b

    def _fill_poses(self) -> None:
        from ggs.library import list_poses

        self._clear(self.poses, self.pose_box)
        current = self.cs.pose
        for i, saved in enumerate(list_poses(self.cs.view, self.project)):
            b = self._saved_tile(saved, "pose")
            if saved.scope == "builtin":
                b.setChecked(saved.data.get("id") == current)
            else:
                b.setChecked(isinstance(current, dict) and current == saved.data["angles"])
            self.poses.addButton(b)
            self.pose_box.addWidget(b, i // 3, i % 3)

    def _fill_bodies(self) -> None:
        from ggs.library import list_bodies

        self._clear(self.bodies, self.body_box)
        for i, saved in enumerate(list_bodies(self.project)):
            b = self._saved_tile(saved, "body")
            if saved.scope == "builtin":
                b.setChecked(not self.cs.body and saved.data["name"] == self.cs.body_preset)
            else:
                b.setChecked(bool(self.cs.body) and self.cs.body.get("name") == saved.name)
            self.bodies.addButton(b)
            self.body_box.addWidget(b, i // 3, i % 3)

    def _ask_name(self, what: str):
        from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QLineEdit

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Save {what}")
        v = QVBoxLayout(dlg)
        name = QLineEdit()
        name.setPlaceholderText(f"Name for this {what}")
        everywhere = QCheckBox("Use it in all my projects")
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        for w in (name, everywhere, btns):
            v.addWidget(w)
        if dlg.exec() != QDialog.DialogCode.Accepted or not name.text().strip():
            return None
        return name.text().strip(), "global" if everywhere.isChecked() or self.project is None else "project"

    def _save_pose(self, name: str | None = None, scope: str | None = None) -> None:
        from ggs.library import save_pose
        from ggs.rig.poses import pose_angles

        got = (name, scope or ("project" if self.project else "global")) if name else self._ask_name("pose")
        if not got:
            return
        try:
            saved = save_pose(got[0], self.cs.view, pose_angles(self.cs.view, self.cs.pose), self.project, got[1])
        except ValueError as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(self, "GGS", str(exc))
            return
        self.cs.pose = dict(saved.data["angles"])
        self._fill_poses()
        self._changed()

    def _save_body(self, name: str | None = None, scope: str | None = None) -> None:
        from ggs.library import save_body

        got = (name, scope or ("project" if self.project else "global")) if name else self._ask_name("body shape")
        if not got:
            return
        try:
            saved = save_body(got[0], self.body_shape().model_dump(), self.project, got[1])
        except ValueError as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(self, "GGS", str(exc))
            return
        self.cs.body = dict(saved.data)
        self._fill_bodies()
        self._changed()

    def _delete(self, kind: str, saved) -> None:
        from PySide6.QtWidgets import QMessageBox

        from ggs.library import delete_body, delete_pose

        answer = QMessageBox.question(self, "GGS", f"Delete the saved {kind} '{saved.name}'?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        (delete_pose if kind == "pose" else delete_body)(saved.name, self.project, saved.scope)
        if kind == "pose":
            self._fill_poses()
        else:
            self._fill_bodies()

    def _view(self, b) -> None:
        from ggs.rig.poses import DEFAULT

        self.cs.view = b.property("view")
        self.cs.pose = DEFAULT[self.cs.view]
        self._fill_poses()
        self._changed()

    def _pose(self, b) -> None:
        saved = b.property("saved")
        self.cs.pose = saved.data["id"] if saved.scope == "builtin" else dict(saved.data["angles"])
        self._changed()

    def _body(self, b) -> None:
        saved = b.property("saved")
        if saved.scope == "builtin":
            self.cs.body_preset, self.cs.body = saved.data["name"], {}
        else:
            self.cs.body = dict(saved.data)
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
