"""VRAM discipline: run each heavy stage inside ``Stages.stage(name)``.

On exit the stage's objects must already be released by the caller (``del``); the manager then runs
``gc.collect()`` + ``torch.cuda.empty_cache()``, records peak VRAM and wall time, and checks that memory went
back near the baseline, so the next model never shares the GPU with the previous one (hard rule 2).
Works without torch/CUDA: then it only records wall time.
"""

from __future__ import annotations

import gc
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

BUDGET_GB = {"standard": 14.0, "lite": 7.0}
LEAK_TOLERANCE_GB = 0.25


class VramBudgetError(RuntimeError):
    pass


@dataclass
class StageReport:
    name: str
    seconds: float
    peak_gb: float | None
    left_gb: float | None  # allocated after unload


@dataclass
class Stages:
    profile: str = "lite"
    enforce: bool = True
    reports: list[StageReport] = field(default_factory=list)

    @staticmethod
    def _cuda():
        try:
            import torch
        except ImportError:
            return None
        return torch if torch.cuda.is_available() else None

    @contextmanager
    def stage(self, name: str):
        torch = self._cuda()
        if torch is not None:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated() / 2**30
        if os.environ.get("GGS_PROGRESS"):  # the desktop app turns these into a progress bar
            print(f"@stage {name}", flush=True)
        from ggs.pipeline import progress

        progress.enter(name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            progress.leave(name)
            gc.collect()
            peak = left = None
            if torch is not None:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                peak = torch.cuda.max_memory_allocated() / 2**30
                left = torch.cuda.memory_allocated() / 2**30
            self.reports.append(StageReport(name, time.perf_counter() - t0, peak, left))
        if torch is not None and self.enforce:
            budget = BUDGET_GB.get(self.profile)
            if budget is not None and peak > budget:
                raise VramBudgetError(f"stage {name!r} peaked at {peak:.2f} GB, over the {self.profile} "
                                      f"budget of {budget:.0f} GB")
            if left - base > LEAK_TOLERANCE_GB:
                raise VramBudgetError(f"stage {name!r} left {left - base:.2f} GB on the GPU after unloading")

    def summary(self) -> list[dict]:
        return [asdict(r) for r in self.reports]
