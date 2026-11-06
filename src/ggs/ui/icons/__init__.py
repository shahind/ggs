"""The app's icon set: Lucide line icons (ISC, tools/fetch_icons.py) tinted at run time, plus the app logo.

``icon(name)`` gives a QIcon in the text colour with a muted disabled state; ``icon(name, "dark")`` is for buttons
on the accent colour. ``set_icon(button, name)`` picks the tone from the button's style. Unknown names give an
empty icon.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
TONES = {"light": "#e8e9ee", "dark": "#16171c", "muted": "#9a9db0", "accent": "#FFEA00"}
SIZES = (16, 20, 24, 32, 48, 64, 128)


def _render(name: str, colour: str, size: int):
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QPainter, QPixmap
    from PySide6.QtSvg import QSvgRenderer

    svg = (HERE / f"{name}.svg").read_text(encoding="utf-8").replace("currentColor", colour)
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(p)
    p.end()
    return pm


@cache
def icon(name: str, tone: str = "light"):
    from PySide6.QtGui import QIcon

    ic = QIcon()
    if not (HERE / f"{name}.svg").exists():
        return ic
    colour = TONES.get(tone, tone)
    for s in SIZES:
        ic.addPixmap(_render(name, colour, s), QIcon.Mode.Normal)
        ic.addPixmap(_render(name, colour if name == "app" else TONES["muted"], s), QIcon.Mode.Disabled)
    return ic


def set_icon(button, name: str, size: int = 16) -> None:
    """Icon on a button: dark on accent ("primary") buttons, light elsewhere, all at the same size."""
    from PySide6.QtCore import QSize

    button.setIcon(icon(name, "dark" if button.objectName() in ("primary", "notice") else "light"))
    button.setIconSize(QSize(size, size))


def pixmap(name: str, size: int = 20, tone: str = "light"):
    return icon(name, tone).pixmap(size, size)


def names() -> list[str]:
    return sorted(p.stem for p in HERE.glob("*.svg"))


@cache
def padded(name: str, box: int, size: int = 18, tone: str = "light"):
    """Icon centred in a larger square, for lists whose rows also hold bigger thumbnails."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon, QPainter, QPixmap

    out = QIcon()
    for scale in (1, 2):
        pm = QPixmap(box * scale, box * scale)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        off = (box - size) * scale // 2
        p.drawPixmap(off, off, _render(name, TONES.get(tone, tone), size * scale))
        p.end()
        out.addPixmap(pm)
    return out
