"""Project window: top bar (projects, name, base model, settings), hierarchy tree on the left, workspace on the
right."""

from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ggs.ui import commands as C
from ggs.ui.jobs import Job, JobQueue
from ggs.ui.kinds import kind
from ggs.ui.widgets import checker_pixmap
from ggs.ui.workspace import ItemWorkspace, Placeholder

MODEL_LABELS = {"flux2-klein-4b": "FLUX.2 klein - best quality", "sdxl": "SDXL - faster style training"}
ROLE_DIR = 256
ROLE_NODE = 257


def _pipelines() -> list[str]:
    from ggs.pipeline.models import load_pipelines

    return sorted(load_pipelines().pipelines)


def _styles(project: Path) -> list[str]:
    from ggs.style.package import list_styles

    try:
        return sorted({m.name for m, _ in list_styles(project)})
    except Exception:  # noqa: BLE001
        return []


class ProjectView(QWidget):
    back = Signal()

    def __init__(self, queue: JobQueue, project: Path):
        super().__init__()
        self.queue, self.project = queue, Path(project)
        self.cache: dict[str, QWidget] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)

        top = QHBoxLayout()
        home = QPushButton("← Projects")
        home.setObjectName("flat")
        home.clicked.connect(self.back.emit)
        self.name = QLabel()
        self.name.setObjectName("h2")
        top.addWidget(home)
        top.addWidget(self.name)
        top.addStretch(1)
        top.addWidget(QLabel("Base model"))
        self.model = QComboBox()
        for m in _pipelines():
            self.model.addItem(MODEL_LABELS.get(m, m), m)
        self.model.currentIndexChanged.connect(self._model_changed)
        top.addWidget(self.model)
        settings = QPushButton("⚙")
        settings.setToolTip("Settings")
        settings.setObjectName("flat")
        settings.clicked.connect(self._settings)
        top.addWidget(settings)
        lay.addLayout(top)

        split = QSplitter()
        left = QWidget()
        left.setObjectName("panel")
        lv = QVBoxLayout(left)
        new = QPushButton("+  New")
        new.setObjectName("primary")
        new.clicked.connect(lambda: self.new_item())
        lv.addWidget(new)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(28, 28))
        self.tree.itemSelectionChanged.connect(self._selected)
        lv.addWidget(self.tree, 1)
        split.addWidget(left)
        self.stack = QStackedWidget()
        split.addWidget(self.stack)
        split.setSizes([260, 1000])
        lay.addWidget(split, 1)
        self.welcome = Placeholder("Pick something on the left, or press  + New  to make your first asset.")
        self.stack.addWidget(self.welcome)
        self.refresh()

    # ------------------------------------------------------------------ project data

    def meta(self) -> dict:
        return yaml.safe_load((self.project / ".ggs" / "project.yaml").read_text(encoding="utf-8")) or {}

    def refresh(self, select: Path | None = None) -> None:
        from ggs.items import list_items

        meta = self.meta()
        self.name.setText(meta.get("name") or self.project.name)
        self.model.blockSignals(True)
        i = self.model.findData(meta.get("base_model") or "flux2-klein-4b")
        self.model.setCurrentIndex(max(0, i))
        self.model.blockSignals(False)
        self.tree.clear()
        chars = QTreeWidgetItem(self.tree, ["\U0001F9CD  Characters"])
        chars.setData(0, ROLE_NODE, "characters")
        assets = QTreeWidgetItem(self.tree, ["\U0001F4E6  Assets"])
        assets.setData(0, ROLE_NODE, "assets")
        anims = QTreeWidgetItem(self.tree, ["\U0001F3AC  Animations"])
        anims.setData(0, ROLE_NODE, "animations")
        styles = QTreeWidgetItem(self.tree, ["\U0001F3A8  Styles"])
        styles.setData(0, ROLE_NODE, "styles")
        for s in _styles(self.project):
            QTreeWidgetItem(styles, [s]).setData(0, ROLE_NODE, "styles")
        groups = {}
        target = None
        for rec, d in list_items(self.project):
            k = kind(rec.asset_class)
            if k.id not in groups:
                groups[k.id] = QTreeWidgetItem(assets, [f"{k.icon}  {k.label}"])
                groups[k.id].setData(0, ROLE_NODE, "group")
            it = QTreeWidgetItem(groups[k.id], [rec.name])
            it.setData(0, ROLE_DIR, str(d))
            v = rec.get_version()
            if v and v.file:
                it.setIcon(0, QIcon(checker_pixmap(d / v.file, 28)))
            if select and Path(select) == d:
                target = it
        for node in (chars, assets, anims, styles, *groups.values()):
            node.setExpanded(True)
        if target is not None:
            self.tree.setCurrentItem(target)

    def _model_changed(self) -> None:
        meta = self.meta()
        meta["base_model"] = self.model.currentData()
        (self.project / ".ggs" / "project.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

    # ------------------------------------------------------------------ navigation

    def _selected(self) -> None:
        it = self.tree.currentItem()
        if it is None:
            return
        d = it.data(0, ROLE_DIR)
        if d:
            self._open_item(Path(d))
            return
        node = it.data(0, ROLE_NODE)
        self.stack.setCurrentWidget(self._section(node))

    def _open_item(self, d: Path) -> None:
        key = str(d)
        if key not in self.cache:
            w = ItemWorkspace(self.queue, self.project, d, _styles(self.project), _pipelines())
            w.changed.connect(lambda d=d: self._item_changed(d))
            self.cache[key] = w
            self.stack.addWidget(w)
        self.stack.setCurrentWidget(self.cache[key])

    def _item_changed(self, d: Path) -> None:
        self.tree.blockSignals(True)
        self.refresh(select=d)
        self.tree.blockSignals(False)

    def _section(self, node: str) -> QWidget:
        if node in self.cache:
            return self.cache[node]
        from ggs.ui.pages import AnimatePage, AppState, CharactersPage, StylesPage

        state = AppState(self.queue)
        state.project = self.project
        page = {"characters": CharactersPage, "animations": AnimatePage, "styles": StylesPage}.get(node)
        w = page(state) if page else self.welcome
        if page:
            self.cache[node] = w
            self.stack.addWidget(w)
        return w

    def new_item(self, preselect: str | None = None) -> None:
        from ggs.ui.wizard import NewItemDialog

        dlg = NewItemDialog(self, preselect)
        dlg.chosen.connect(self._create)
        dlg.exec()

    def _create(self, kind_id: str, name: str, size: str, describe: str) -> None:
        if kind_id == "character":
            self.stack.setCurrentWidget(self._section("characters"))
            page = self.cache["characters"]
            page.prompt.setText(describe)
            page.name.setText(name)
            return
        from ggs.items import item_dir

        d = item_dir(self.project, kind_id, name)
        job = Job(f"new {name}", [C.asset_new(self.project, kind_id, name, describe, size)])

        def done(ok: bool) -> None:
            if not ok:
                from PySide6.QtWidgets import QMessageBox

                from ggs.ui.errors import friendly_error

                QMessageBox.information(self, "GGS", friendly_error("\n".join(job.output)))
                return
            self.refresh(select=d)
            w = self.cache.get(str(d))
            if w is not None:  # start the first drawing right away
                w._run("new", describe)

        job.done = done
        self.queue.submit(job)

    def _settings(self) -> None:
        from ggs.ui.settings import SettingsDialog

        SettingsDialog(self.queue, self).exec()

