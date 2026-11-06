"""Start screen: the user's projects as cards, plus "New project"."""

from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ggs.ui.icons import set_icon
from ggs.ui.widgets import PathPicker, checker_pixmap
from ggs.ui.wizard import tile

PROJECTS_KEY = "projects"


def known_projects(settings: QSettings) -> list[Path]:
    raw = settings.value(PROJECTS_KEY) or []
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for p in raw:
        q = Path(p)
        if (q / ".ggs" / "project.yaml").exists() and q not in out:
            out.append(q)
    return out


def remember(settings: QSettings, project: Path) -> None:
    items = [str(project)] + [str(p) for p in known_projects(settings) if p != Path(project)]
    settings.setValue(PROJECTS_KEY, items[:30])


def default_location() -> Path:
    return Path.home() / "Documents" / "GGS Projects"


def forget(settings: QSettings, project: Path) -> None:
    settings.setValue(PROJECTS_KEY, [str(p) for p in known_projects(settings) if p != Path(project)])


class ProjectCard(QFrame):
    clicked = Signal(Path)
    delete = Signal(Path)

    def __init__(self, project: Path | None):
        super().__init__()
        self.setObjectName("card")
        self.setFixedSize(230, 210)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.project = project
        lay = QVBoxLayout(self)
        if project is None:
            big = QLabel("+")
            big.setAlignment(Qt.AlignmentFlag.AlignCenter)
            big.setStyleSheet("font-size: 48pt;")
            lay.addWidget(big, 1)
            lab = QLabel("New project")
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setObjectName("h2")
            lay.addWidget(lab)
            return
        from ggs.items import list_items

        items = list_items(project)
        thumb = QLabel()
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        latest = next((d / v.file for rec, d in reversed(items) if (v := rec.get_version()) and v.file), None)
        if latest is not None:
            thumb.setPixmap(checker_pixmap(latest, 120))
        else:
            from ggs.ui.icons import pixmap

            thumb.setPixmap(pixmap("projects", 48, "muted"))
        lay.addWidget(thumb, 1)
        meta = yaml.safe_load((project / ".ggs" / "project.yaml").read_text(encoding="utf-8")) or {}
        name = QLabel(meta.get("name") or project.name)
        name.setObjectName("h2")
        lay.addWidget(name)
        sub = QLabel(f"{len(items)} item{'s' if len(items) != 1 else ''}")
        sub.setObjectName("muted")
        row = QHBoxLayout()
        row.addWidget(sub, 1)

        rm = QPushButton()
        rm.setObjectName("flat")
        set_icon(rm, "delete")
        rm.setToolTip("Delete this project, or take it off this list")
        rm.setCursor(Qt.CursorShape.ArrowCursor)
        rm.clicked.connect(lambda: self.delete.emit(project))
        row.addWidget(rm)
        lay.addLayout(row)
        self.setToolTip(str(project))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda _pos: self.delete.emit(project))

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.project if self.project else Path())


class NewProjectDialog(QDialog):
    created = Signal(Path, str, str)  # folder, name, base model

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        t = QLabel("New project")
        t.setObjectName("h2")
        lay.addWidget(t)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Game name, e.g. Dungeon Run")
        lay.addWidget(self.name)
        self.where = PathPicker(caption="Where to keep it")
        self.where.set_path(default_location())
        lay.addWidget(QLabel("Saved in"))
        lay.addWidget(self.where)
        lay.addWidget(QLabel("Base Model"))
        row = QHBoxLayout()
        self.models = QButtonGroup(self)
        for mid, title, sub in (("flux2-klein-4b", "FLUX.2 klein", "best quality"),
                                ("sdxl", "SDXL", "faster style training")):
            b = tile(title, sub, 200, 64)
            b.setProperty("model", mid)
            self.models.addButton(b)
            row.addWidget(b)
        row.addStretch(1)
        self.models.buttons()[0].setChecked(True)
        lay.addLayout(row)
        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Create")
        ok.setObjectName("primary")
        set_icon(ok, "confirm")
        ok.clicked.connect(self._ok)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        lay.addLayout(btns)

    def _ok(self) -> None:
        name = self.name.text().strip()
        if not name:
            self.name.setFocus()
            return
        from ggs.items import slug

        folder = (self.where.path() or default_location()) / slug(name)
        self.created.emit(folder, name, self.models.checkedButton().property("model"))
        self.accept()


def ask_delete_project(parent, project: Path) -> str | None:
    """'delete' (to the Recycle Bin), 'forget' (off the list, files stay) or None."""
    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Delete project")
    box.setText(f"Delete the project '{project.name}'?")
    box.setInformativeText(f"{project}\n\nDelete moves the whole folder (every character, asset, animation and "
                           "export in it) to the Recycle Bin. 'Remove from list' only takes it off this screen.")
    delete = box.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
    forget = box.addButton("Remove from list", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(QMessageBox.StandardButton.Cancel)
    box.exec()
    return {delete: "delete", forget: "forget"}.get(box.clickedButton())


class StartScreen(QWidget):
    open_project = Signal(Path)
    new_project = Signal()
    delete_project = Signal(Path, str)  # project, 'delete' or 'forget'

    def __init__(self, settings: QSettings):
        super().__init__()
        self.settings = settings
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 30, 40, 30)
        h = QLabel("Game Graphic Studio")
        h.setObjectName("h1")
        lay.addWidget(h)
        sub = QLabel("")
        sub.setObjectName("muted")
        lay.addWidget(sub)
        scroll = QScrollArea()
        scroll.setObjectName("root")
        scroll.setWidgetResizable(True)
        self.inner = QWidget()
        self.inner.setObjectName("root")
        self.grid = QGridLayout(self.inner)
        self.grid.setSpacing(16)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(self.inner)
        lay.addWidget(scroll, 1)
        other = QPushButton("Open Folder...")
        other.setObjectName("flat")
        set_icon(other, "open")
        other.clicked.connect(self._open_folder)
        lay.addWidget(other, 0, Qt.AlignmentFlag.AlignLeft)
        self.refresh()

    def refresh(self) -> None:
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        cards = [ProjectCard(None)] + [ProjectCard(p) for p in known_projects(self.settings)]
        for i, c in enumerate(cards):
            c.clicked.connect(self._card)
            c.delete.connect(self._delete)
            self.grid.addWidget(c, i // 4, i % 4)

    def _card(self, p: Path) -> None:
        if p == Path():
            self.new_project.emit()
        else:
            self.open_project.emit(p)

    def _delete(self, p: Path) -> None:
        choice = ask_delete_project(self, p)
        if choice:
            self.delete_project.emit(p, choice)

    def _open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Project folder", str(default_location()))
        if d and (Path(d) / ".ggs" / "project.yaml").exists():
            self.open_project.emit(Path(d))

