"""Job queue for the desktop app: runs ``ggs`` command lines one at a time with QProcess.

One job at a time is a feature, not a limit: generation and training each want the whole GPU (hard rule 2).
A job is a list of steps (command lines) that run in order; a failing step stops the job.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from ggs.paths import repo_root
from ggs.ui.errors import stage_text


@dataclass
class Job:
    title: str
    steps: list[list[str]]
    done: Callable[[bool], None] | None = None  # called with success when the job ends
    on_line: Callable[[str], None] | None = None  # every output line (progress parsing)
    exit_ok: tuple[int, ...] = (0,)  # exit codes that count as success (e.g. (0, 2) when QA failures are kept)
    output: list[str] = field(default_factory=list)  # everything the job printed (for "details")
    exit_code: int | None = None
    _step: int = field(default=0, repr=False)


class JobQueue(QObject):
    line = Signal(str)  # every output line, prefixed by nothing
    started = Signal(str)  # job title
    stage = Signal(str)  # friendly name of the step that started ("Drawing...")
    finished = Signal(str, bool)  # job title, success
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: deque[Job] = deque()
        self._job: Job | None = None
        self._proc: QProcess | None = None
        self._buf = ""

    @property
    def busy(self) -> bool:
        return self._job is not None

    def pending(self) -> int:
        return len(self._queue)

    def submit(self, job: Job) -> None:
        self._queue.append(job)
        if not self.busy:
            self._next()

    def cancel(self) -> None:
        """Stop the running job and drop the queue."""
        self._queue.clear()
        if self._proc is not None:
            self._proc.kill()

    # ------------------------------------------------------------------ internals

    def _next(self) -> None:
        if not self._queue:
            self._job = None
            self.busy_changed.emit(False)
            return
        self._job = self._queue.popleft()
        self._job._step = 0
        self.busy_changed.emit(True)
        self.started.emit(self._job.title)
        self._run_step()

    def _run_step(self) -> None:
        job = self._job
        cmd = job.steps[job._step]
        self.line.emit("$ " + " ".join(_quote(c) for c in cmd[1:]))
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("GGS_PROGRESS", "1")
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(repo_root()))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._read)
        proc.finished.connect(self._step_done)
        self._proc = proc
        proc.start(cmd[0], cmd[1:])

    def _read(self) -> None:
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._buf += data.replace("\r", "\n")
        *lines, self._buf = self._buf.split("\n")
        for ln in lines:
            if ln.strip() and "it/s]" not in ln and "Loading weights" not in ln:
                if self._job:
                    self._job.output.append(ln)
                st = stage_text(ln)
                if st:
                    self.stage.emit(st)
                    continue
                self.line.emit(ln)
                if self._job and self._job.on_line:
                    self._job.on_line(ln)

    def _step_done(self, code: int, _status) -> None:
        if self._buf.strip():
            self.line.emit(self._buf)
            self._buf = ""
        job = self._job
        job.exit_code = code
        ok = code in job.exit_ok
        job._step += 1
        if ok and job._step < len(job.steps):
            self._run_step()
            return
        self._proc = None
        self.line.emit(f"-- {job.title}: {'done' if ok else f'failed (exit {code})'}")
        self.finished.emit(job.title, ok)
        if job.done:
            job.done(ok)
        self._next()


def _quote(s: str) -> str:
    return f'"{s}"' if (" " in s or not s) else s

