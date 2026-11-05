"""GGS desktop app: ``python -m ggs ui`` or ``ggs-studio``.

Start screen (projects) -> project window (hierarchy tree + chat-centred workspace). Every action is a ``ggs``
command run by a one-at-a-time queue, so the app behaves exactly like the CLI and each job gets the whole GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QStackedWidget

from ggs import __version__
from ggs.ui import commands as C
from ggs.ui.jobs import Job, JobQueue
from ggs.ui.start import NewProjectDialog, StartScreen, remember
from ggs.ui.theme import QSS


class MainWindow(QMainWindow):
    def __init__(self, project: Path | None = None, settings: QSettings | None = None):
        super().__init__()
        self.setWindowTitle(f"GGS - Game Graphic Studio {__version__}")
        self.resize(1320, 860)
        self.settings = settings or QSettings("GGS", "GameGraphicStudio")
        self.queue = JobQueue(self)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.start = StartScreen(self.settings)
        self.start.open_project.connect(self.open_project)
        self.start.new_project.connect(self.new_project)
        self.stack.addWidget(self.start)
        self.view = None
        if project and (Path(project) / ".ggs" / "project.yaml").exists():
            self.open_project(Path(project))

    def check_models(self) -> None:
        """First run: offer to download the AI models when a required one is missing."""
        from ggs.ui.setup import ModelSetupDialog, missing_required

        if missing_required():
            ModelSetupDialog(self.queue, self).exec()

    def open_project(self, project: Path) -> None:
        from ggs.ui.project_view import ProjectView

        remember(self.settings, project)
        if self.view is not None:
            self.stack.removeWidget(self.view)
            self.view.deleteLater()
        self.view = ProjectView(self.queue, project)
        self.view.back.connect(self.show_start)
        self.stack.addWidget(self.view)
        self.stack.setCurrentWidget(self.view)
        self.setWindowTitle(f"GGS - {project.name}")

    def show_start(self) -> None:
        self.start.refresh()
        self.stack.setCurrentWidget(self.start)
        self.setWindowTitle(f"GGS - Game Graphic Studio {__version__}")

    def new_project(self) -> None:
        dlg = NewProjectDialog(self)
        dlg.created.connect(self.create_project)
        dlg.exec()

    def create_project(self, folder: Path, name: str, model: str) -> None:
        job = Job("new project", [C.project_init(folder, name, model)])

        def done(ok: bool) -> None:
            if ok:
                self.open_project(folder)
            else:
                from ggs.ui.errors import show_error

                show_error(self, "\n".join(job.output))

        job.done = done
        self.queue.submit(job)

    def closeEvent(self, event) -> None:
        if self.queue.busy and QMessageBox.question(
                self, "GGS", "Something is still being made. Stop it and quit?") != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        self.queue.cancel()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    project = Path(argv[argv.index("--project") + 1]) if "--project" in argv else None
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("GGS")
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    from ggs.ui.icons import icon

    app.setWindowIcon(icon("assets"))
    win = MainWindow(project)
    win.show()
    if "--no-setup" not in argv:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(300, win.check_models)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
