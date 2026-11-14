"""Body & pose editor: view, pose and body tiles, proportion sliders, and a live figure whose limbs can be dragged.

The figure shows the pose and proportions the character is drawn in. Dragging redraws from a cached puppet at
the preview's own size, so it keeps up with the mouse; the settings are emitted once the mouse is let go.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

from PySide6.QtCore import QPointF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ggs.rig.body import limits, saved_body
from ggs.ui.icons import icon, set_icon
from ggs.ui.theme import ACCENT, INK, NOTICE

# (BodyShape field, label); the four limb segments have their own sliders
SLIDERS = [("head", "Head"), ("torso", "Body"), ("upper_arm", "Upper arm"), ("forearm", "Forearm"),
           ("thigh", "Thigh"), ("shin", "Lower leg"), ("width", "Build")]
DRAG_BONES = ("hand", "forearm", "upper_arm", "shin", "thigh", "foot", "head")
PREVIEW = 320
GRAB = 22  # px (on screen) around a joint end that picks it up


@lru_cache(maxsize=2)
def _rig(view: str):
    from ggs.rig.poses import rig_for

    return rig_for(view)


@lru_cache(maxsize=8)
def _puppet(view: str, body_json: str):
    from ggs.rig.make import guide_puppet

    return guide_puppet(_rig(view), saved_body(json.loads(body_json)), look="mannequin")


def _small_tile(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("tile")
    b.setCheckable(True)
    b.setStyleSheet("padding: 4px 6px;")
    return b


class FigureView(QLabel):
    """The pose render; dragging a limb's end turns that limb toward the mouse."""

    angle_changed = Signal(str, float)
    released = Signal()

    def __init__(self):
        super().__init__()
        self.setFixedSize(PREVIEW, PREVIEW)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: white; border-radius: 10px;")
        self.setMouseTracking(True)
        self.guide = None
        self.drag: str | None = None
        self.hover: str | None = None

    def show_guide(self, guide) -> None:
        self.guide = guide
        img = guide.image
        qimg = QImage(img.tobytes("raw", "RGBA"), img.width, img.height, QImage.Format.Format_RGBA8888).copy()
        pm = QPixmap(PREVIEW, PREVIEW)
        pm.fill(QColor(255, 255, 255))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(pm.rect(), qimg)
        k = PREVIEW / img.width
        for name, (_, (ex, ey)) in guide.spec.joints.items():
            if self._draggable(name):
                on = name in (self.drag, self.hover)
                p.setPen(QPen(QColor(INK), 1.5))
                p.setBrush(QColor(ACCENT if not on else NOTICE))
                p.drawEllipse(QPointF(ex * k, ey * k), 6 if on else 4.5, 6 if on else 4.5)
        p.end()
        self.setPixmap(pm)

    @staticmethod
    def _draggable(name: str) -> bool:
        return name.rsplit("_", 1)[0] in DRAG_BONES or name == "head"

    def _joint_at(self, pos) -> str | None:
        if self.guide is None:
            return None
        k = self.guide.image.width / PREVIEW
        x, y = pos.x() * k, pos.y() * k
        best, dist = None, GRAB * k
        for name, (_, (ex, ey)) in self.guide.spec.joints.items():
            if self._draggable(name):
                d = math.hypot(ex - x, ey - y)
                if d < dist:
                    best, dist = name, d
        return best

    def mousePressEvent(self, e) -> None:
        self.drag = self._joint_at(e.position())
        if self.drag:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e) -> None:
        if self.drag is None:
            near = self._joint_at(e.position())
            if near != self.hover:
                self.hover = near
                self.setCursor(Qt.CursorShape.OpenHandCursor if near else Qt.CursorShape.ArrowCursor)
                if self.guide is not None:
                    self.show_guide(self.guide)
            return
        k = self.guide.image.width / PREVIEW
        x, y = e.position().x() * k, e.position().y() * k
        (sx, sy), _ = self.guide.spec.joints[self.drag]
        if math.hypot(x - sx, y - sy) < 3:
            return
        self.angle_changed.emit(self.drag, math.degrees(math.atan2(-(x - sx), y - sy)))

    def mouseReleaseEvent(self, e) -> None:
        if self.drag:
            self.drag = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._joint_at(e.position()) else
                           Qt.CursorShape.ArrowCursor)
            self.released.emit()

    def leaveEvent(self, e) -> None:
        if self.hover and self.drag is None:
            self.hover = None
            if self.guide is not None:
                self.show_guide(self.guide)


class _Section(QWidget):
    """A section title with a Save button that only shows when there is something new to save."""

    def __init__(self, title: str, on_save):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 6, 0, 0)
        lab = QLabel(title)
        lab.setStyleSheet("font-weight: 600;")
        self.save = QPushButton(f"Save {title}")
        set_icon(self.save, "save")
        self.save.setToolTip(f"You changed the {title.lower()}: save it to reuse it later")
        self.save.clicked.connect(lambda: on_save())
        self.save.setVisible(False)
        row.addWidget(lab)
        row.addStretch(1)
        row.addWidget(self.save)


class BodyPoseEditor(QFrame):
    """Edits a CharacterSettings; ``changed`` fires with the new settings (after a drag ends, a tile is picked
    or the sliders rest)."""

    changed = Signal(object)

    def __init__(self, settings, project=None):
        super().__init__()
        from ggs.rig.body import PRESETS

        self.project = project
        self.setObjectName("panel")
        self.setFixedWidth(PREVIEW + 40)
        self.cs = settings.model_copy(deep=True)
        self.presets = PRESETS
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        lay = QVBoxLayout(inner)
        t = QLabel("Body & pose")
        t.setObjectName("h2")
        lay.addWidget(t)
        self.figure = FigureView()
        self.figure.angle_changed.connect(self._dragged)
        self.figure.released.connect(self._emit)
        lay.addWidget(self.figure, 0, Qt.AlignmentFlag.AlignHCenter)
        hint = QLabel("")
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
        self.pose_head = _Section("Pose", self._save_pose)
        lay.addWidget(self.pose_head)
        self.pose_box = QGridLayout()
        self.poses = QButtonGroup(self)
        self.poses.buttonClicked.connect(self._pose)
        lay.addLayout(self.pose_box)
        self.body_head = _Section("Body", self._save_body)
        lay.addWidget(self.body_head)
        self.body_box = QGridLayout()
        self.bodies = QButtonGroup(self)
        self.bodies.buttonClicked.connect(self._body)
        lay.addLayout(self.body_box)
        self.sliders = {}
        for key, label in SLIDERS:
            r = QHBoxLayout()
            lab = QLabel(label)
            lab.setFixedWidth(70)
            sl = QSlider(Qt.Orientation.Horizontal)
            lo, hi = limits(key)
            sl.setRange(max(40, math.ceil(lo * 100)), min(220, math.floor(hi * 100)))
            sl.valueChanged.connect(self._slid)
            sl.sliderReleased.connect(self._emit)
            r.addWidget(lab)
            r.addWidget(sl, 1)
            lay.addLayout(r)
            self.sliders[key] = sl
        lay.addStretch(1)
        self.emit_timer = QTimer(self)  # sliders moved by keyboard / wheel: emit once they rest
        self.emit_timer.setSingleShot(True)
        self.emit_timer.setInterval(400)
        self.emit_timer.timeout.connect(self._emit)
        self._fill_poses()
        self._fill_bodies()
        self._set_sliders()
        self._render()

    # ------------------------------------------------------------------ state

    def body_shape(self):

        return saved_body(self.cs.body) if self.cs.body else self.presets.get(self.cs.body_preset,
                                                                             self.presets["standard"])

    def _set_sliders(self) -> None:
        b = self.body_shape()
        for key, sl in self.sliders.items():
            sl.blockSignals(True)
            sl.setValue(round((b.factor(key) if key not in ("width",) else b.width) * 100))
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
        any_checked = False
        for i, saved in enumerate(list_poses(self.cs.view, self.project)):
            b = self._saved_tile(saved, "pose")
            if saved.scope == "builtin":
                b.setChecked(saved.data.get("id") == current)
            else:
                b.setChecked(isinstance(current, dict) and current == saved.data["angles"])
            any_checked |= b.isChecked()
            self.poses.addButton(b)
            self.pose_box.addWidget(b, i // 3, i % 3)
        self.pose_head.save.setVisible(not any_checked)

    def _fill_bodies(self) -> None:
        from ggs.library import list_bodies

        self._clear(self.bodies, self.body_box)
        any_checked = False
        for i, saved in enumerate(list_bodies(self.project)):
            b = self._saved_tile(saved, "body")
            if saved.scope == "builtin":
                b.setChecked(not self.cs.body and saved.data["name"] == self.cs.body_preset)
            else:
                b.setChecked(bool(self.cs.body) and self.cs.body.get("name") == saved.name)
            any_checked |= b.isChecked()
            self.bodies.addButton(b)
            self.body_box.addWidget(b, i // 3, i % 3)
        self.body_head.save.setVisible(not any_checked)

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
            from ggs.ui.errors import show_message

            show_message(self, str(exc))
            return
        self.cs.pose = dict(saved.data["angles"])
        self._fill_poses()
        self._emit()

    def _save_body(self, name: str | None = None, scope: str | None = None) -> None:
        from ggs.library import save_body

        got = (name, scope or ("project" if self.project else "global")) if name else self._ask_name("body shape")
        if not got:
            return
        try:
            saved = save_body(got[0], self.body_shape().model_dump(), self.project, got[1])
        except ValueError as exc:
            from ggs.ui.errors import show_message

            show_message(self, str(exc))
            return
        self.cs.body = dict(saved.data)
        self._fill_bodies()
        self._emit()

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
        self._render()
        self._emit()

    def _pose(self, b) -> None:
        saved = b.property("saved")
        self.cs.pose = saved.data["id"] if saved.scope == "builtin" else dict(saved.data["angles"])
        self.pose_head.save.setVisible(False)
        self._render()
        self._emit()

    def _body(self, b) -> None:
        saved = b.property("saved")
        if saved.scope == "builtin":
            self.cs.body_preset, self.cs.body = saved.data["name"], {}
        else:
            self.cs.body = dict(saved.data)
        self._set_sliders()
        self.body_head.save.setVisible(False)
        self._render()
        self._emit()

    @staticmethod
    def _uncheck(group: QButtonGroup) -> None:
        if group.checkedButton():
            group.setExclusive(False)
            group.checkedButton().setChecked(False)
            group.setExclusive(True)

    def _slid(self) -> None:
        b = self.body_shape().model_dump()
        for k, sl in self.sliders.items():
            b[k] = sl.value() / 100
        b["name"], b["words"] = "custom", ""
        self.cs.body = b
        self._uncheck(self.bodies)
        self.body_head.save.setVisible(True)
        self._render()
        if not any(sl.isSliderDown() for sl in self.sliders.values()):
            self.emit_timer.start()

    def _dragged(self, bone: str, angle: float) -> None:
        from ggs.rig.poses import pose_angles

        angles = pose_angles(self.cs.view, self.cs.pose)
        delta = angle - angles[bone]
        angles[bone] = angle
        # children follow the bone they hang from, so the limb bends as one piece below the joint
        rig = _rig(self.cs.view)
        kids = [b.name for b in rig.bones if b.parent == bone]
        while kids:
            k = kids.pop()
            angles[k] = angles[k] + delta
            kids += [b.name for b in rig.bones if b.parent == k]
        self.cs.pose = angles
        self._uncheck(self.poses)
        self.pose_head.save.setVisible(True)
        self._render()

    def _emit(self) -> None:
        self.emit_timer.stop()
        self.changed.emit(self.cs.model_copy(deep=True))

    def _render(self) -> None:
        from ggs.rig.master import render_guide
        from ggs.rig.poses import pose_angles

        body = self.body_shape()
        puppet, lengths = _puppet(self.cs.view, body.model_dump_json())
        guide = render_guide(_rig(self.cs.view), puppet, lengths, size=PREVIEW,
                             angles=pose_angles(self.cs.view, self.cs.pose))
        self.figure.show_guide(guide)
