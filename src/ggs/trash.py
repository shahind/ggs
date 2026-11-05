"""Deleting from the app: files and folders go to the Recycle Bin, so a wrong click can be undone there.

Windows uses the shell (SHFileOperationW with "allow undo"); elsewhere ``gio trash`` when it is there. When no
recycle bin can take it (another OS without one, a network drive) the path is deleted for good only if
``permanent_ok``; otherwise ``TrashError`` is raised.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class TrashError(OSError):
    pass


def _windows(path: Path) -> bool:
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    fo_delete = 3
    flags = 0x0040 | 0x0010 | 0x0400 | 0x0004  # ALLOWUNDO | NOCONFIRMATION | NOERRORUI | SILENT
    op = SHFILEOPSTRUCTW(None, fo_delete, str(path.resolve()) + "\0", None, flags, False, None, None)
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    return rc == 0 and not op.fAnyOperationsAborted and not path.exists()


def move_to_trash(path: str | Path, permanent_ok: bool = False) -> None:
    """Send a file or folder to the Recycle Bin. A path that does not exist is ignored."""
    path = Path(path)
    if not path.exists():
        return
    if sys.platform == "win32":
        if _windows(path):
            return
    elif shutil.which("gio"):
        if subprocess.run(["gio", "trash", str(path)], check=False, capture_output=True).returncode == 0:
            return
    if not permanent_ok:
        raise TrashError(f"could not move {path} to the Recycle Bin")
    shutil.rmtree(path) if path.is_dir() else path.unlink()
