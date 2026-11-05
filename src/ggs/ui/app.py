"""GGS desktop app (M9): ``python -m ggs ui`` or ``ggs-studio``.

A thin shell over the CLI: pages build ``ggs`` commands, a queue runs them one at a time, and the log shows
everything they print. Project folders are the same as the CLI's.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ggs import __version__
from ggs.ui.jobs import JobQueue
from ggs.ui.pages import (
    AnimatePage,
    AppState,
    CharactersPage,
    GeneratePage,
    ProjectPage,
    StylesPage,
    SystemPage,
)
from ggs.ui.widgets import LogView

PAGES = [("Project", ProjectPage), ("Generate", GeneratePage), ("Characters", CharactersPage),
         ("Animate", AnimatePage), ("Styles", StylesPage), ("System", SystemPage)]


class MainWindow(QMainWindow):
    def __init__(self, project: Path | None = None):
        super().__init__()
        self.setWindowTitle(f"GGS - Game Graphic Studio {__version__}")
        self.resize(1280, 860)
        self.queue = JobQueue(self)
        self.state = AppState(self.queue)
        self.settings = QSettings("GGS", "GameGraphicStudio")

        self.nav = QListWidget()
        self.nav.setFixedWidth(150)
        self.stack = QStackedWidget()
        self.pages = {}
        for name, cls in PAGES:
            page = cls(self.state)
            self.pages[name] = page
            self.nav.addItem(name)
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)
        central = QWidget()
        h = QHBoxLayout(central)
        h.addWidget(self.nav)
        h.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.log = LogView()
        self.job_label = QLabel("Idle")
        cancel = QPushButton("Cancel job")
        cancel.clicked.connect(self.queue.cancel)
        dock_w = QWidget()
        v = QVBoxLayout(dock_w)
        top = QHBoxLayout()
        top.addWidget(self.job_label, 1)
        top.addWidget(cancel)
        v.addLayout(top)
        v.addWidget(self.log)
        dock = QDockWidget("Jobs and log", self)
        dock.setObjectName("log")
        dock.setWidget(dock_w)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [170], Qt.Orientation.Vertical)

        self.queue.line.connect(self.log.append_line)
        self.queue.started.connect(lambda t: self.job_label.setText(f"Running: {t}"
                                                                    + (f" (+{self.queue.pending()} queued)"
                                                                       if self.queue.pending() else "")))
        self.queue.busy_changed.connect(lambda b: None if b else self.job_label.setText("Idle"))
        self.state.project_changed.connect(self._remember)

        start = project or (Path(self.settings.value("last_project")) if self.settings.value("last_project")
                            else None)
        if start and (Path(start) / ".ggs" / "project.yaml").exists():
            self.state.set_project(Path(start))

    def _remember(self) -> None:
        if self.state.project:
            self.settings.setValue("last_project", str(self.state.project))
            self.setWindowTitle(f"GGS - {self.state.project.name}")

    def closeEvent(self, event) -> None:
        self.queue.cancel()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    project = None
    if "--project" in argv:
        project = Path(argv[argv.index("--project") + 1])
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("GGS")
    win = MainWindow(project)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
