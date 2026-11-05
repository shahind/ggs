"""The app's icon set (made with GGS: tools/make_icons.py). ``icon(name)`` gives a QIcon with the 64 px and
128 px (HiDPI) images; unknown names give an empty icon."""

from __future__ import annotations

from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent


@cache
def icon(name: str):
    from PySide6.QtGui import QIcon

    ic = QIcon()
    for suffix in ("", "@2x"):
        f = HERE / f"{name}{suffix}.png"
        if f.exists():
            ic.addFile(str(f))
    return ic


def pixmap(name: str, size: int = 20):
    return icon(name).pixmap(size, size)


def names() -> list[str]:
    return sorted(p.stem for p in HERE.glob("*.png") if "@" not in p.stem)
