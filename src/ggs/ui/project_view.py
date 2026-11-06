"""Project window: top bar (projects, name, base model, settings), hierarchy tree on the left, workspace on the
right."""

from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ggs.ui import commands as C
from ggs.ui.icons import padded, set_icon
from ggs.ui.jobs import Job, JobQueue
from ggs.ui.kinds import kind
from ggs.ui.widgets import CurrentStack, checker_pixmap
from ggs.ui.workspace import CharacterWorkspace, ItemWorkspace, Placeholder

MODEL_LABELS = {"flux2-klein-4b": "FLUX.2 klein - best quality", "sdxl": "SDXL - faster style training"}
ROLE_DIR = 256
ROLE_NODE = 257
ROLE_CLIP = 258


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
        home = QPushButton("Projects")
        set_icon(home, "projects")
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
        settings = QPushButton()
        set_icon(settings, "settings")
        settings.setToolTip("Settings")
        settings.setObjectName("flat")
        settings.clicked.connect(self._settings)
        top.addWidget(settings)
        lay.addLayout(top)

        split = QSplitter()
        left = QWidget()
        left.setObjectName("panel")
        left.setMinimumWidth(190)
        lv = QVBoxLayout(left)
        new = QPushButton("New...")
        new.setObjectName("primary")
        set_icon(new, "new")
        new.clicked.connect(lambda: self.new_item())
        lv.addWidget(new)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(28, 28))
        self.tree.itemSelectionChanged.connect(self._selected)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        QShortcut(QKeySequence.StandardKey.Delete, self.tree, activated=self._delete_selected,
                  context=Qt.ShortcutContext.WidgetShortcut)
        lv.addWidget(self.tree, 1)
        split.addWidget(left)
        self.stack = CurrentStack()
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
        sections = {}
        for key, label in (("characters", "Characters"), ("assets", "Assets"), ("animations", "Animations"),
                           ("styles", "Styles")):
            node = QTreeWidgetItem(self.tree, [label])
            node.setIcon(0, padded(key, 28))
            node.setData(0, ROLE_NODE, key)
            sections[key] = node
        chars, assets, anims, styles = (sections[k] for k in ("characters", "assets", "animations", "styles"))
        for s in _styles(self.project):
            QTreeWidgetItem(styles, [s]).setData(0, ROLE_NODE, "styles")
        from ggs.rig.library import clip_label
        from ggs.ui.animations import _project_clips

        for c in _project_clips(self.project):
            it = QTreeWidgetItem(anims, [clip_label(c)])
            it.setData(0, ROLE_NODE, "animations")
            it.setData(0, ROLE_CLIP, str(c))
        groups = {}
        target = None
        for rec, d in list_items(self.project):
            k = kind(rec.asset_class)
            if rec.asset_class == "character":
                it = QTreeWidgetItem(chars, [rec.name])
                it.setData(0, ROLE_DIR, str(d))
                v = rec.get_version()
                if v and v.file:
                    it.setIcon(0, QIcon(checker_pixmap(d / v.file, 28)))
                if select and Path(select) == d:
                    target = it
                continue
            if k.id not in groups:
                groups[k.id] = QTreeWidgetItem(assets, [k.label])
                groups[k.id].setIcon(0, padded(k.icon, 28))
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
        w = self._section(node)
        self.stack.setCurrentWidget(w)
        clip = it.data(0, ROLE_CLIP)
        if clip and hasattr(w, "show_clip"):
            w.show_clip(Path(clip))

    def _open_item(self, d: Path) -> None:
        key = str(d)
        if key not in self.cache:
            import json

            is_char = json.loads((d / "item.json").read_text(encoding="utf-8")).get("asset_class") == "character"
            cls = CharacterWorkspace if is_char else ItemWorkspace
            w = cls(self.queue, self.project, d, _styles(self.project), _pipelines())
            w.changed.connect(lambda d=d: self._item_changed(d))
            w.delete_requested.connect(self.delete_item)
            self.cache[key] = w
            self.stack.addWidget(w)
        self.stack.setCurrentWidget(self.cache[key])

    def _anims_changed(self) -> None:
        self.tree.blockSignals(True)
        self.refresh()
        self.tree.blockSignals(False)

    def _item_changed(self, d: Path) -> None:
        self.tree.blockSignals(True)
        self.refresh(select=d)
        self.tree.blockSignals(False)
        anims = self.cache.get("animations")
        if anims is not None:
            anims.mine.refresh()

    # ------------------------------------------------------------------ deleting (to the Recycle Bin)

    def _menu(self, pos) -> None:
        it = self.tree.itemAt(pos)
        if it is None or not (it.data(0, ROLE_DIR) or it.data(0, ROLE_CLIP)):
            return
        from PySide6.QtWidgets import QMenu

        from ggs.ui.errors import trash_icon

        menu = QMenu(self)
        act = menu.addAction(trash_icon(self), f"Delete '{it.text(0)}'...")
        if menu.exec(self.tree.viewport().mapToGlobal(pos)) is act:
            self._delete_tree_item(it)

    def _delete_selected(self) -> None:
        it = self.tree.currentItem()
        if it is not None:
            self._delete_tree_item(it)

    def _delete_tree_item(self, it: QTreeWidgetItem) -> None:
        if it.data(0, ROLE_DIR):
            self.delete_item(Path(it.data(0, ROLE_DIR)))
        elif it.data(0, ROLE_CLIP):
            self.delete_clip(Path(it.data(0, ROLE_CLIP)), it.text(0))

    def delete_item(self, d: Path) -> None:
        import json

        from ggs.items import ItemError, delete_item
        from ggs.ui.errors import busy_refusal, confirm_delete, show_error

        if busy_refusal(self, self.queue):
            return
        rec = json.loads((d / "item.json").read_text(encoding="utf-8"))
        is_char = rec.get("asset_class") == "character"
        what = f"the character '{rec['name']}'" if is_char else f"'{rec['name']}'"
        detail = ("Its drawings, design, sliced version and drawn frames are deleted with it." if is_char
                  else "Its drawings are deleted with it.") + " Files you already exported stay."
        if not confirm_delete(self, what, detail):
            return
        w = self.cache.pop(str(d), None)
        if w is not None:  # close the workspace first: its animations keep their GIF files open
            from ggs.ui.character_page import Movie

            for m in w.findChildren(Movie):
                m.play(None)
            self.stack.setCurrentWidget(self.welcome)
            self.stack.removeWidget(w)
            w.deleteLater()
        try:
            delete_item(self.project, f"{rec['asset_class']}/{rec['name']}")
        except (ItemError, OSError) as exc:
            show_error(self, str(exc), "Not deleted")
        self.tree.blockSignals(True)
        self.refresh()
        self.tree.blockSignals(False)
        self.stack.setCurrentWidget(self.welcome)

    def delete_clip(self, clip: Path, label: str) -> None:
        from ggs.rig.library import delete_clip
        from ggs.ui.errors import confirm_delete, show_error

        if not confirm_delete(self, f"the animation '{label}'", "Characters already exported keep it."):
            return
        try:
            delete_clip(clip)
        except OSError as exc:
            show_error(self, str(exc), "Not deleted")
        self._anims_changed()
        anims = self.cache.get("animations")
        if anims is not None:
            anims.mine.refresh()

    def _section(self, node: str) -> QWidget:
        if node in self.cache:
            return self.cache[node]
        from ggs.ui.pages import AppState

        state = AppState(self.queue)
        state.project = self.project
        if node == "characters":
            ph = Placeholder("Pick a character on the left, or press  + New  and choose Character.")
            self.cache[node] = ph
            self.stack.addWidget(ph)
            return ph
        if node == "animations":
            from ggs.ui.animations import AnimationsPage

            w = AnimationsPage(self.queue, self.project, state)
            w.changed.connect(self._anims_changed)
            self.cache[node] = w
            self.stack.addWidget(w)
            return w
        if node == "styles":
            from ggs.ui.styles import StylesSection

            w = StylesSection(self.queue, self.project, _pipelines(), self.meta().get("base_model"))
            self.cache[node] = w
            self.stack.addWidget(w)
            return w
        return self.welcome

    def new_item(self, preselect: str | None = None) -> None:
        from ggs.ui.wizard import NewItemDialog

        dlg = NewItemDialog(self, preselect)
        dlg.chosen.connect(self._create)
        dlg.exec()

    def _create(self, kind_id: str, name: str, size: str, describe: str) -> None:
        from ggs.items import item_dir

        d = item_dir(self.project, kind_id, name)
        job = Job(f"new {name}", [C.asset_new(self.project, kind_id, name, describe, size)])

        def done(ok: bool) -> None:
            if not ok:
                from ggs.ui.errors import show_error

                show_error(self, "\n".join(job.output))
                return
            self.refresh(select=d)
            w = self.cache.get(str(d))
            if w is not None and kind_id != "character":  # assets start drawing right away; characters wait
                w.run("new", describe)                      # for the body & pose to be set

        job.done = done
        self.queue.submit(job)

    def _settings(self) -> None:
        from ggs.ui.settings import SettingsDialog

        SettingsDialog(self.queue, self).exec()

