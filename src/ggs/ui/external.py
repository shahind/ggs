"""Open-source image editors installed on this computer, for edits beyond the built-in tools.

A file opened in one of them is watched: when it is saved there, GGS reloads it.
"""

from __future__ import annotations

import os
import shutil
from glob import glob
from pathlib import Path

# (name, glob patterns under Program Files / LocalAppData, executable names on PATH)
EDITORS = [
    ("Krita", [r"{pf}\Krita (x64)\bin\krita.exe", r"{pf}\Krita*\bin\krita.exe"], ["krita"]),
    ("GIMP", [r"{pf}\GIMP*\bin\gimp-*.exe", r"{local}\Programs\GIMP*\bin\gimp-*.exe"], ["gimp"]),
    ("Pixelorama", [r"{pf}\Pixelorama\Pixelorama.exe", r"{local}\Programs\Pixelorama\Pixelorama.exe",
                    r"{pf86}\Steam\steamapps\common\Pixelorama\Pixelorama.exe"], ["pixelorama"]),
    ("LibreSprite", [r"{pf}\LibreSprite\libresprite.exe", r"{local}\Programs\LibreSprite\libresprite.exe"],
     ["libresprite"]),
    ("Paint.NET", [r"{pf}\paint.net\paintdotnet.exe"], ["paintdotnet"]),
]


def find_editors() -> list[tuple[str, Path]]:
    """(name, executable) of the editors found, in the order above."""
    roots = {"pf": os.environ.get("ProgramFiles", r"C:\Program Files"),
             "pf86": os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
             "local": os.environ.get("LOCALAPPDATA", "")}
    out = []
    for name, patterns, exes in EDITORS:
        hit = None
        for pat in patterns:
            matches = sorted(glob(pat.format(**roots)))
            if matches:
                hit = Path(matches[-1])
                break
        if hit is None:
            for exe in exes:
                found = shutil.which(exe)
                if found:
                    hit = Path(found)
                    break
        if hit is not None:
            out.append((name, hit))
    return out


def open_with(exe: Path | None, path: Path) -> None:
    """Open ``path`` in ``exe`` (None: the system's default app for PNG files)."""
    import subprocess

    if exe is None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        return
    subprocess.Popen([str(exe), str(path)], close_fds=True)
