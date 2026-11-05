"""Progress for the desktop app: ``@progress N`` lines (0-100) on stdout while a job runs (``GGS_PROGRESS=1``).

The stages of a job (reading the prompt, drawing, cutting out...) take very different times, and the same stage
takes a different time on every machine. How long each stage took is kept in ``~/.ggs/cache/stage_times.json``
per job kind, so the percentage moves at the speed it will really go on this computer. A ticker thread prints
the estimate twice a second; inside a stage it creeps toward 95% of the stage's share, never backwards.

    progress.start("character:flux2-klein-4b", ["encode", "denoise", "matte"])
    ... Stages.stage(name) calls enter(name) / leave(name) ...
    progress.finish()   # also runs at exit

Without ``GGS_PROGRESS`` everything here is a no-op (the CLI prints nothing extra).
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path

DEFAULT_SECONDS = {  # first run on a machine (measured on the 8 GB laptop, lite profile)
    "start": 8.0, "encode": 25.0, "denoise": 55.0, "decode": 6.0, "matte": 12.0, "latents": 40.0,
    "caption": 90.0, "train": 3600.0, "motion_text": 35.0, "motion": 12.0, "control": 4.0,
    "cut": 90.0,
}
_lock = threading.Lock()
_active: _Ticker | None = None


def _times_file() -> Path:
    from ggs.paths import shared_dir

    return shared_dir() / "cache" / "stage_times.json"


def _load_times() -> dict:
    try:
        return json.loads(_times_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class _Ticker:
    def __init__(self, kind: str, plan: list[str]):
        self.kind = kind
        self.times = _load_times()
        self.plan = ["start", *plan]
        self.done: list[tuple[str, float]] = []  # finished stages and how long they took
        self.current = "start"
        self.since = time.perf_counter()
        self.shown = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def expected(self, stage: str) -> float:
        return float(self.times.get(f"{self.kind}:{stage}") or self.times.get(f"*:{stage}")
                     or DEFAULT_SECONDS.get(stage, 10.0))

    def enter(self, stage: str) -> None:
        with _lock:
            self._close()
            if stage in self.plan:  # stages planned before this one did not run (e.g. a cached prompt)
                self.plan = self.plan[self.plan.index(stage):]
            else:
                self.plan.insert(0, stage)
            self.current, self.since = stage, time.perf_counter()

    def leave(self, stage: str) -> None:
        with _lock:
            if self.current == stage:
                self._close()
                self.current, self.since = None, time.perf_counter()

    def _close(self) -> None:
        if self.current is None:
            return
        self.done.append((self.current, time.perf_counter() - self.since))
        if self.plan and self.plan[0] == self.current:
            self.plan.pop(0)

    def percent(self) -> int:
        with _lock:
            spent = sum(s for _, s in self.done)
            ahead = sum(self.expected(s) for s in self.plan if s != self.current)
            if self.current is not None:
                el = time.perf_counter() - self.since
                exp = self.expected(self.current)
                spent += min(el, 0.95 * exp)
                ahead += max(exp - el, 0.05 * exp)
            total = spent + ahead
        pct = int(99 * spent / total) if total > 0 else 0
        self.shown = max(self.shown, min(pct, 99))
        return self.shown

    def _run(self) -> None:
        while not self.stop.wait(0.5):
            print(f"@progress {self.percent()}", flush=True)

    def finish(self) -> None:
        self.stop.set()
        with _lock:
            self._close()
            self.current = None
        times = _load_times()
        for stage, sec in self.done:
            for key in (f"{self.kind}:{stage}", f"*:{stage}"):
                old = times.get(key)
                times[key] = round(sec if old is None else 0.5 * old + 0.5 * sec, 2)
        try:
            f = _times_file()
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(times, indent=1, sort_keys=True), encoding="utf-8")
        except OSError:
            pass
        print("@progress 100", flush=True)


def start(kind: str, plan: list[str]) -> None:
    """Begin reporting for a job of ``kind`` whose stages are expected in ``plan`` order."""
    global _active
    if not os.environ.get("GGS_PROGRESS") or _active is not None:
        return
    _active = _Ticker(kind, plan)
    atexit.register(finish)


def enter(stage: str) -> None:
    if _active is not None:
        _active.enter(stage)


def leave(stage: str) -> None:
    if _active is not None:
        _active.leave(stage)


def finish() -> None:
    global _active
    if _active is not None:
        t, _active = _active, None
        t.finish()
