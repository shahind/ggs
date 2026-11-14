"""The app's pages. Each page builds ``ggs`` commands (``ggs.ui.commands``) and hands them to the job queue."""

from __future__ import annotations

import json
import re
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QMovie
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ggs.paths import repo_root
from ggs.ui import commands as C
from ggs.ui.jobs import Job, JobQueue
from ggs.ui.widgets import ImageGrid, PathPicker, movie_from_file

BODIES = ["standard", "heroic", "chibi", "stocky", "lanky", "child"]
FORMATS = ["apng", "webp", "gif", "sheet", "rig", "godot", "unity", "defold"]


class AppState(QObject):
    project_changed = Signal()

    def __init__(self, queue: JobQueue):
        super().__init__()
        self.queue = queue
        self.project: Path | None = None

    def set_project(self, path: Path | None) -> None:
        self.project = Path(path) if path else None
        self.project_changed.emit()

    def base_model(self) -> str | None:
        if not self.project:
            return None
        from ggs.style.package import load_project

        try:
            return load_project(self.project).base_model
        except Exception:  # noqa: BLE001 - a broken project.yaml shows as "no base model"
            return None


def _need_project(page: QWidget, state: AppState) -> bool:
    if state.project is None:
        QMessageBox.information(page, "GGS", "Open or create a project first (Project page).")
        return False
    return True


def _pipelines() -> list[str]:
    from ggs.pipeline.models import load_pipelines

    return sorted(load_pipelines().pipelines)


def _styles(project: Path | None) -> list[str]:
    from ggs.style.package import list_styles

    try:
        return sorted({m.name for m, _ in list_styles(project)})
    except Exception:  # noqa: BLE001
        return []


def _title(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet("font-size: 18px; font-weight: 600; margin-bottom: 6px;")
    return lab


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: #666;")
    return lab


# ====================================================================== Project

class ProjectPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Project"))
        lay.addWidget(_hint("A project is a folder for one game. Its styles, assets, characters and exports are "
                            "kept inside it; models are shared by all projects (~/.ggs)."))
        self.current = QLabel("No project open.")
        self.current.setWordWrap(True)
        lay.addWidget(self.current)
        box = QGroupBox("New project")
        f = QFormLayout(box)
        self.new_dir = PathPicker(caption="Project folder", placeholder="an empty or existing game folder")
        self.new_name = QLineEdit()
        self.new_model = QComboBox()
        self.new_model.addItems(_pipelines())
        f.addRow("Folder", self.new_dir)
        f.addRow("Name", self.new_name)
        f.addRow("Base model", self.new_model)
        btn = QPushButton("Create project")
        btn.clicked.connect(self._create)
        f.addRow(btn)
        lay.addWidget(box)
        box2 = QGroupBox("Open project")
        h = QHBoxLayout(box2)
        self.open_dir = PathPicker(caption="Project folder")
        ob = QPushButton("Open")
        ob.clicked.connect(lambda: self._open(self.open_dir.path()))
        h.addWidget(self.open_dir, 1)
        h.addWidget(ob)
        lay.addWidget(box2)
        lay.addStretch(1)
        state.project_changed.connect(self._show)

    def _create(self) -> None:
        d = self.new_dir.path()
        if not d:
            return
        cmd = C.project_init(d, self.new_name.text().strip() or None, self.new_model.currentText())
        self.state.queue.submit(Job("Create project", [cmd], done=lambda ok: ok and self._open(d)))

    def _open(self, d: Path | None) -> None:
        if not d:
            return
        if not (Path(d) / ".ggs" / "project.yaml").exists():
            QMessageBox.warning(self, "GGS", f"{d} is not a GGS project (no .ggs/project.yaml).")
            return
        self.state.set_project(Path(d))

    def _show(self) -> None:
        p = self.state.project
        if p is None:
            self.current.setText("No project open.")
            return
        n_assets = len(list(C.assets_dir(p, "").glob("*/*.png"))) if C.assets_dir(p, "").exists() else 0
        n_chars = len(list(C.characters_dir(p).glob("*/seed_*"))) if C.characters_dir(p).exists() else 0
        self.current.setText(f"<b>{p}</b><br>base model: {self.state.base_model() or 'recipe default'} | "
                             f"styles: {', '.join(_styles(p)) or 'none'} | assets: {n_assets} | "
                             f"character drafts: {n_chars}")


# ====================================================================== Generate

class GeneratePage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        from ggs.recipes import available_recipes, load_recipe

        self.state = state
        self.last_out: Path | None = None
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Generate assets"))
        form = QFormLayout()
        self.kind = QComboBox()
        for r in available_recipes():
            self.kind.addItem(f"{r} - {load_recipe(r).description}", r)
        self.style = QComboBox()
        self.weight = QDoubleSpinBox()
        self.weight.setRange(0.0, 2.0)
        self.weight.setSingleStep(0.1)
        self.weight.setValue(1.0)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("What to draw, e.g. 'a red health potion'")
        self.prompt.setFixedHeight(60)
        self.count = QSpinBox()
        self.count.setRange(1, 8)
        self.count.setValue(2)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setSpecialValueText("0")
        self.size = QLineEdit()
        self.size.setPlaceholderText("optional, e.g. 768x768 (default: the asset type's)")
        self.model = QComboBox()
        form.addRow("Asset type", self.kind)
        form.addRow("Style", self.style)
        form.addRow("Style strength", self.weight)
        form.addRow("Prompt", self.prompt)
        form.addRow("Variants", self.count)
        form.addRow("Seed", self.seed)
        form.addRow("Size", self.size)
        form.addRow("Base model", self.model)
        lay.addLayout(form)
        row = QHBoxLayout()
        self.go = QPushButton("Generate")
        self.go.clicked.connect(self._run)
        row.addWidget(self.go)
        self.status = QLabel("")
        row.addWidget(self.status, 1)
        lay.addLayout(row)
        self.grid = ImageGrid()
        lay.addWidget(self.grid, 1)
        lay.addWidget(_hint("Green: passed QA and ready for the engine. Red: failed QA (kept so you can see why). "
                            "Double-click opens the file."))
        state.project_changed.connect(self._refresh_lists)
        self._refresh_lists()

    def _refresh_lists(self) -> None:
        self.style.clear()
        self.style.addItem("(none)", None)
        for s in _styles(self.state.project):
            self.style.addItem(s, s)
        self.model.clear()
        self.model.addItem(f"(project default: {self.state.base_model() or 'recipe default'})", None)
        from ggs.ui.project_view import base_models

        for m, label in base_models():
            self.model.addItem(label, m)

    def _run(self) -> None:
        if not _need_project(self, self.state):
            return
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            return
        kind = self.kind.currentData()
        out = C.new_asset_path(self.state.project, kind, prompt)
        cmd = C.generate(kind, prompt, out, project=self.state.project, style=self.style.currentData(),
                         style_weight=self.weight.value() if self.style.currentData() else None,
                         count=self.count.value(), seed=self.seed.value(), model=self.model.currentData(),
                         size=self.size.text().strip() or None)
        self.last_out = out
        self.status.setText("Queued...")
        self.state.queue.submit(Job(f"Generate {kind}", [cmd], exit_ok=(0, 2), done=lambda ok: self._show(out, ok)))

    def _show(self, out: Path, ok: bool) -> None:
        rows = C.read_generate_results(out)
        self.status.setText(f"{sum(r.passed for r in rows)}/{len(rows)} passed QA -> {out.parent}" if rows else
                            "Failed: see the log below.")
        for r in reversed(rows):  # the grid puts each new item first
            self.grid.add(r.path, f"{r.path.name}\n{r.detail}", r.passed)


# ====================================================================== Characters

class CharactersPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Characters"))
        lay.addWidget(_hint("Draws the character over a pose guide in the body shape you choose, cuts it into a "
                            "part sheet for the rig, checks it, and repaints hidden areas and expressions."))
        form = QFormLayout()
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("e.g. a knight in silver armour with a red cape")
        self.name = QLineEdit()
        self.name.setPlaceholderText("knight")
        self.body = QComboBox()
        self.body.addItems(BODIES + ["custom"])
        self.custom = QLineEdit()
        self.custom.setPlaceholderText("custom ratios, e.g. head=1.6,legs=0.7,width=1.2")
        self.stature = QDoubleSpinBox()
        self.stature.setRange(0.3, 3.0)
        self.stature.setSingleStep(0.05)
        self.stature.setValue(1.0)
        self.pose = QComboBox()
        self.pose.addItems(["walk", "spread"])
        self.count = QSpinBox()
        self.count.setRange(1, 6)
        self.count.setValue(2)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setSpecialValueText("0")
        self.style = QComboBox()
        self.refine = QCheckBox("Repaint hidden areas and draw expressions (klein)")
        self.refine.setChecked(True)
        form.addRow("Character", self.prompt)
        form.addRow("Name", self.name)
        form.addRow("Body shape", self.body)
        form.addRow("", self.custom)
        form.addRow("Stature", self.stature)
        form.addRow("Master pose", self.pose)
        form.addRow("Variants", self.count)
        form.addRow("Seed", self.seed)
        form.addRow("Style", self.style)
        form.addRow("", self.refine)
        lay.addLayout(form)
        row = QHBoxLayout()
        go = QPushButton("Make character")
        go.clicked.connect(self._run)
        row.addWidget(go)
        self.status = QLabel("")
        row.addWidget(self.status, 1)
        lay.addLayout(row)
        self.grid = ImageGrid(thumb=200)
        lay.addWidget(self.grid, 1)
        state.project_changed.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        self.style.clear()
        self.style.addItem("(none)", None)
        for s in _styles(self.state.project):
            self.style.addItem(s, s)

    def _run(self) -> None:
        if not _need_project(self, self.state):
            return
        prompt = self.prompt.text().strip()
        if not prompt:
            return
        name = C.slug(self.name.text().strip() or prompt, 24)
        body = self.custom.text().strip() if self.body.currentText() == "custom" else self.body.currentText()
        out = C.characters_dir(self.state.project) / f"{name}_{C.stamp()}"
        cmd = C.character_master(prompt, out, name, project=self.state.project, body=body or None,
                                 stature=self.stature.value(), pose=self.pose.currentText(),
                                 count=self.count.value(), seed=self.seed.value(), style=self.style.currentData(),
                                 refine=self.refine.isChecked())
        self.status.setText("Queued...")
        self.state.queue.submit(Job(f"Character {name}", [cmd], exit_ok=(0, 2), done=lambda ok: self._show(out)))

    def _show(self, out: Path) -> None:
        rows = C.read_character_results(out)
        good = [r for r in rows if r["passed"]]
        self.status.setText(f"{len(good)}/{len(rows)} variants gave a valid part sheet ({out.name}). Valid ones "
                            "appear on the Animate page." if rows else "Failed: see the log below.")
        self.grid.clear()
        for r in reversed(rows):  # the grid puts each new item first: add in reverse to read seed by seed
            verdict = f"valid ({r['score']:.3f})" if r["passed"] else "rejected: " + "; ".join(r["failures"][:2])
            if r["validation"].exists():
                self.grid.add(r["validation"], f"{r['seed']} rebuild + test poses\ntest poses", r["passed"])
            if r["parts"].exists():
                self.grid.add(r["parts"], f"{r['seed']} part sheet\npart sheet", r["passed"])
            self.grid.add(r["master"], f"{r['seed']} master\n{r['seed']}: {verdict}", r["passed"])


# ====================================================================== Animate

def valid_characters(project: Path | None) -> list[Path]:
    if not project or not C.characters_dir(project).exists():
        return []
    out = []
    for v in sorted(C.characters_dir(project).glob("*/seed_*/character/validation.json")):
        try:
            if json.loads(v.read_text(encoding="utf-8")).get("passed"):
                out.append(v.parent / "character.yaml")
        except (OSError, ValueError):
            continue
    return out


class AnimatePage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.movie: QMovie | None = None
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Animate and export"))
        form = QFormLayout()
        self.character = QComboBox()
        self.clip = QComboBox()
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.clip, 1)
        imp = QPushButton("Import BVH...")
        imp.clicked.connect(self._import_bvh)
        clip_row.addWidget(imp)
        self.upper = QComboBox()
        self.bones = QComboBox()
        self.bones.addItems(["arms", "near_arm", "upper"])
        self.item = PathPicker(caption="Item folder (with item.yaml)", placeholder="optional held item")
        item_row = QHBoxLayout()
        item_row.addWidget(self.item, 1)
        mk = QPushButton("Make item from image...")
        mk.clicked.connect(self._make_item)
        item_row.addWidget(mk)
        self.fmt = QComboBox()
        self.fmt.addItems(FORMATS)
        self.height = QSpinBox()
        self.height.setRange(32, 2048)
        self.height.setValue(256)
        form.addRow("Character", self.character)
        form.addRow("Motion", clip_row)
        form.addRow("Upper body", self.upper)
        form.addRow("Upper body bones", self.bones)
        form.addRow("Held item", item_row)
        form.addRow("Format / engine", self.fmt)
        form.addRow("Height (px)", self.height)
        lay.addLayout(form)
        row = QHBoxLayout()
        go = QPushButton("Export")
        go.clicked.connect(self._run)
        row.addWidget(go)
        refresh = QPushButton("Refresh lists")
        refresh.clicked.connect(self._refresh)
        row.addWidget(refresh)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        lay.addLayout(row)
        self.preview = QLabel("Preview appears here after an export.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview.setStyleSheet("background: white; border: 1px solid #ddd;")
        lay.addWidget(self.preview, 1)
        state.project_changed.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        p = self.state.project
        self.character.clear()
        for c in valid_characters(p):
            self.character.addItem(f"{c.parent.parent.parent.name} / {c.parent.parent.name}", c)
        self.clip.clear()
        if p and C.clips_dir(p).exists():
            for c in sorted(C.clips_dir(p).glob("*.json")):
                if not c.name.endswith(".motion.json"):  # sidecars of library / generated motions
                    self.clip.addItem(c.stem, c)
        if self.clip.count() == 0:
            self.clip.addItem("(no animations yet: add some in Animations > Library)", None)
        self.upper.clear()
        self.upper.addItem("(none: the motion's own arms)", None)
        for k in sorted((repo_root() / "clips").glob("*.yaml")):
            self.upper.addItem(k.stem, k)

    def _import_bvh(self) -> None:
        if not _need_project(self, self.state):
            return
        start = str(repo_root() / "data" / "mocap")
        f, _ = QFileDialog.getOpenFileName(self, "BVH file", start, "BVH (*.bvh)")
        if not f:
            return
        clips = C.clips_dir(self.state.project)
        clips.mkdir(parents=True, exist_ok=True)
        raw = clips / f"{Path(f).stem}_raw.json"
        cycle = clips / f"{Path(f).stem}.json"
        self.state.queue.submit(Job(f"Import {Path(f).name}", [C.retarget(Path(f), raw), C.clip_loop(raw, cycle)],
                                    done=lambda ok: self._refresh()))

    def _make_item(self) -> None:
        if not _need_project(self, self.state):
            return
        f, _ = QFileDialog.getOpenFileName(self, "Upright item image (Generate > held_item)",
                                           str(C.assets_dir(self.state.project, "held_item")), "PNG (*.png)")
        if not f:
            return
        name = C.slug(Path(f).stem.split("_2026")[0], 20)
        out = Path(self.state.project) / "items" / name
        hold = "two_hand" if QMessageBox.question(self, "GGS", "Held with two hands (spear, staff)?") == \
            QMessageBox.StandardButton.Yes else "one_hand"
        length = 1.0 if hold == "two_hand" else 0.5
        self.state.queue.submit(Job(f"Item {name}", [C.item_make(Path(f), out, name, length, hold)],
                                    done=lambda ok: ok and self.item.set_path(out)))

    def _run(self) -> None:
        if not _need_project(self, self.state):
            return
        char, clip = self.character.currentData(), self.clip.currentData()
        if not char or not clip:
            QMessageBox.information(self, "GGS", "Pick a valid character (Characters page) and a motion.")
            return
        steps = []
        work = C.exports_dir(self.state.project) / ".work"
        work.mkdir(parents=True, exist_ok=True)
        motion, label = clip, clip.stem
        over = self.upper.currentData()
        if over:
            keyed = work / f"{over.stem}.json"
            motion = work / f"{clip.stem}_{over.stem}.json"
            steps += [C.clip_keys(over, keyed, char), C.clip_layer(clip, keyed, motion, self.bones.currentText())]
            label = f"{clip.stem}_{over.stem}"
        items = [self.item.path()] if self.item.path() else []
        fmt = self.fmt.currentText()
        cname = char.parent.parent.parent.name
        base = C.exports_dir(self.state.project) / cname
        out = base / (f"{label}.png" if fmt in ("apng", "sheet") else f"{label}.{fmt}" if fmt in ("webp", "gif")
                      else f"{label}_{fmt}")
        steps.append(C.export(motion, char, fmt, out, height=self.height.value(), items=items, name=label,
                              matte="#FFFFFF" if fmt == "gif" else None))
        preview = work / f"{label}_preview.gif"
        steps.append(C.export(motion, char, "gif", preview, height=min(self.height.value(), 320), items=items,
                              matte="#FFFFFF"))
        self.status.setText("Queued...")
        self.state.queue.submit(Job(f"Export {label} ({fmt})", steps, done=lambda ok: self._show(ok, out, preview)))

    def _show(self, ok: bool, out: Path, preview: Path) -> None:
        self.status.setText(f"Wrote {out}" if ok else "Failed: see the log below.")
        if ok and preview.exists():
            self.movie = movie_from_file(preview)
            self.preview.setMovie(self.movie)
            self.movie.start()


# ====================================================================== Styles

class StylesPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        from ggs.style.dataset import RIGHTS_WARNING

        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_title("Styles"))
        lay.addWidget(_hint("Train the generator on 20-40 images in one look, then pick the style on any page."))
        b1 = QGroupBox("1. Training images")
        f1 = QFormLayout(b1)
        self.src = PathPicker(caption="Folder with your images")
        self.ds_name = QLineEdit()
        self.ds_name.setPlaceholderText("inkwash")
        warn = QLabel(RIGHTS_WARNING)
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #8a5a00;")
        self.rights = QCheckBox("I have the rights to these images")
        self.import_btn = QPushButton("Import images")
        self.import_btn.setEnabled(False)
        self.rights.toggled.connect(self.import_btn.setEnabled)
        self.import_btn.clicked.connect(self._import)
        f1.addRow("Images", self.src)
        f1.addRow("Style name", self.ds_name)
        f1.addRow(warn)
        f1.addRow(self.rights)
        f1.addRow(self.import_btn)
        lay.addWidget(b1)
        b2 = QGroupBox("2. Captions and 3. training")
        f2 = QFormLayout(b2)
        self.trigger = QLineEdit()
        self.trigger.setPlaceholderText("a rare word for the style, e.g. GGSINKWASH")
        self.model = QComboBox()
        self.steps = QSpinBox()
        self.steps.setRange(0, 10000)
        self.steps.setSpecialValueText("model default")
        cap = QPushButton("Write captions")
        cap.clicked.connect(self._caption)
        tr = QPushButton("Train style (long: about 45-100 min on 8 GB)")
        tr.clicked.connect(self._train)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        f2.addRow("Trigger word", self.trigger)
        f2.addRow(cap)
        f2.addRow("Base model", self.model)
        f2.addRow("Steps", self.steps)
        f2.addRow(tr)
        f2.addRow(self.progress)
        lay.addWidget(b2)
        self.installed = QLabel("")
        self.installed.setWordWrap(True)
        lay.addWidget(self.installed)
        lay.addStretch(1)
        state.project_changed.connect(self._refresh)
        self._refresh()

    def _ds(self) -> Path | None:
        name = C.slug(self.ds_name.text().strip(), 24)
        return Path(self.state.project) / "style_data" / name if self.state.project and name != "asset" else None

    def _refresh(self) -> None:
        self.model.clear()
        self.model.addItem(f"(project default: {self.state.base_model() or 'flux2-klein-4b'})", None)
        for m in _pipelines():
            self.model.addItem(m, m)
        self.installed.setText("Installed styles: " + (", ".join(_styles(self.state.project)) or "none"))

    def _import(self) -> None:
        if not _need_project(self, self.state) or not self.src.path() or not self._ds():
            return
        self.state.queue.submit(Job("Import style images",
                                    [C.dataset_import(self.src.path(), self._ds(), self.rights.isChecked())]))

    def _caption(self) -> None:
        if not _need_project(self, self.state) or not self._ds() or not self.trigger.text().strip():
            return
        self.state.queue.submit(Job("Caption images", [C.caption(self._ds(), self.trigger.text().strip())]))

    def _train(self) -> None:
        if not _need_project(self, self.state) or not self._ds():
            return
        name = C.slug(self.ds_name.text().strip(), 24)
        pkg = Path(self.state.project) / "style_data" / f"{name}.ggsstyle"
        self.progress.setVisible(True)
        self.progress.setValue(0)

        def on_line(line: str) -> None:
            m = re.match(r"step\s+(\d+)/(\d+)", line)
            if m:
                self.progress.setMaximum(int(m.group(2)))
                self.progress.setValue(int(m.group(1)))

        steps = [C.train(self._ds(), name, pkg, project=self.state.project, model=self.model.currentData(),
                         steps=self.steps.value() or None), C.style_install(pkg, self.state.project)]
        self.state.queue.submit(Job(f"Train style {name}", steps, on_line=on_line,
                                    done=lambda ok: (self.progress.setVisible(False), self._refresh(),
                                                     self.state.project_changed.emit())))


# ====================================================================== System

class SystemPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_title("System and models"))
        lay.addWidget(_hint("Check the GPU profile and which model files each pipeline would load. Output goes to "
                            "the log below."))
        row = QHBoxLayout()
        for label, cmd in (("Doctor (GPU / profile)", C.doctor), ("Check models", C.models_check)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, c=cmd, t=label: state.queue.submit(Job(t, [c()], exit_ok=(0, 1))))
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        box = QGroupBox("Reuse model files from another folder (e.g. ComfyUI models)")
        h = QHBoxLayout(box)
        self.dir = PathPicker(caption="Models folder")
        add = QPushButton("Add folder")
        add.clicked.connect(lambda: self.dir.path() and state.queue.submit(
            Job("Add model folder", [C.models_add_dir(self.dir.path())])))
        h.addWidget(self.dir, 1)
        h.addWidget(add)
        lay.addWidget(box)
        lay.addStretch(1)
