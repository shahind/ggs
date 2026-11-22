"""Stage the small ("web") installers: the GGS source and data plus uv, and nothing large.

Python 3.13, the libraries (torch, Qt...), the motion library and the AI models are downloaded during setup by
``tools/setup_env.py``, so the installers are a few tens of MB instead of 5 GB.

    python tools/stage_app.py --windows      # dist/stage/GGS (+ uv.exe, GGS.ico, launchers); then
                                             #   ISCC tools/installer_web.iss  ->  dist/GGS-Setup-<ver>-web.exe
    python tools/stage_app.py --mac          # on macOS: dist/GGS-<ver>-macos-arm64.dmg (GGS.app inside)

Needs PySide6 and Pillow to draw the icon. The full offline installer (portable Python + CUDA torch, ~5 GB) is
still ``tools/build_app.py`` + ``tools/installer.iss``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UV_VERSION = "0.13.9"
UV_ASSETS = {  # asset name, sha256 (github.com/astral-sh/uv/releases)
    "windows": ("uv-x86_64-pc-windows-msvc.zip", "edd61f7f5728a56cc62455386661860bdfc7bab65872f39b0c25bdcc1d8e1beb"),
    "mac": ("uv-aarch64-apple-darwin.tar.gz", "5aeca67f2283f75235de40d0bd9c0a5bbc5fce23b6d48b3bf9fa1b2db08382ca"),
}
DATA = ["recipes", "rigs", "clips"]
SKIP = ("__pycache__", ".pyc", ".pyo")
UV_ENV = ("UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR", "UV_PYTHON_PREFERENCE")


def version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE)
    if not m:
        raise SystemExit("pyproject.toml has no version")
    return m.group(1)


def copy(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    for p in sorted(src.rglob("*")):
        if p.is_dir() or any(s in p.as_posix() for s in SKIP):
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)


def stage_source(app: Path) -> None:
    """Everything setup_env.py needs to install GGS: pyproject, src, data, constraints, the setup script."""
    if app.exists():
        shutil.rmtree(app)
    app.mkdir(parents=True)
    for f in ("pyproject.toml", "README.md", "requirements/constraints.txt", "tools/setup_env.py",
              "motions/library.yaml", "motions/CREDITS.md"):
        copy(ROOT / f, app / f)
    copy(ROOT / "src" / "ggs", app / "src" / "ggs")
    for d in DATA:
        copy(ROOT / d, app / d)
    for f in ("registry.yaml", "pipelines.yaml", "downloads.yaml"):
        copy(ROOT / "models" / f, app / "models" / f)
    lic = app / "licenses"
    copy(ROOT / "docs" / "LICENSE_REVIEW.md", lic / "LICENSE_REVIEW.md")
    copy(ROOT / "motions" / "CREDITS.md", lic / "Motion-CREDITS.md")
    (app / "VERSION").write_text(version(), encoding="utf-8")


def fetch_uv(platform_key: str, dest_dir: Path) -> Path:
    asset, sha = UV_ASSETS[platform_key]
    url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{asset}"
    print(f"uv {UV_VERSION}: {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise SystemExit(f"uv download checksum mismatch: {got}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            out = dest_dir / "uv.exe"
            out.write_bytes(z.read("uv.exe"))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            member = next(m for m in t.getmembers() if m.name.endswith("/uv"))
            out = dest_dir / "uv"
            out.write_bytes(t.extractfile(member).read())
            out.chmod(0o755)
    (dest_dir / "uv-LICENSE.txt").write_text(
        "uv (https://github.com/astral-sh/uv) is licensed under Apache-2.0 or MIT, at your option.\n", "utf-8")
    return out


def render_logo(px: int):
    """The app logo (src/ggs/ui/icons/app.svg) as a Pillow image."""
    from PIL import Image
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    _app = QGuiApplication.instance() or QGuiApplication([])
    svg = QSvgRenderer(QByteArray((ROOT / "src" / "ggs" / "ui" / "icons" / "app.svg").read_bytes()))
    img = QImage(px, px, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    svg.render(p)
    p.end()
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")


# ------------------------------------------------------------------ Windows


SETUP_CMD = r"""@echo off
rem Installs or repairs GGS's Python, libraries and motion library in this folder (see tools\setup_env.py).
rem   "Repair GGS.cmd" --models required   also download the AI models this PC needs
setlocal
set "UV_PYTHON_INSTALL_DIR=%~dp0python"
set "UV_CACHE_DIR=%~dp0.uv-cache"
set "UV_PYTHON_PREFERENCE=only-managed"
"%~dp0uv.exe" run --no-project --python 3.13 "%~dp0tools\setup_env.py" --app --root "%~dp0." --pause %*
set "RESULT=%ERRORLEVEL%"
rem the download cache (several GB) is not needed once the environment is built; uv run held it until now
"%~dp0uv.exe" cache clean >nul 2>&1
exit /b %RESULT%
"""
START_CMD = r"""@echo off
if not exist "%~dp0env\Scripts\pythonw.exe" call "%~dp0Repair GGS.cmd" || exit /b 1
start "" "%~dp0env\Scripts\pythonw.exe" -m ggs.ui.app %*
"""
GGS_CMD = '@echo off\r\n"%~dp0env\\Scripts\\python.exe" -m ggs %*\r\n'


def stage_windows(out: Path) -> Path:
    app = out / "GGS"
    stage_source(app)
    fetch_uv("windows", app)
    for name, text in (("Repair GGS.cmd", SETUP_CMD), ("Start GGS.cmd", START_CMD)):
        (app / name).write_text(text.replace("\r\n", "\n").replace("\n", "\r\n"), encoding="utf-8")
    (app / "ggs.cmd").write_text(GGS_CMD, encoding="utf-8")
    render_logo(256).save(app / "GGS.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128),
                                                 (256, 256)])
    print(f"staged {app}; next: ISCC tools\\installer_web.iss")
    return app


# ------------------------------------------------------------------ macOS


LAUNCHER = """#!/bin/bash
# GGS launcher: first start (or a new version) runs setup in Terminal; afterwards it starts the app.
RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
ROOT="$HOME/Library/Application Support/GGS"
VER="$(cat "$RES/app/VERSION")"
if [ "$(cat "$ROOT/.setup_ok" 2>/dev/null)" != "$VER" ] || [ ! -x "$ROOT/env/bin/python" ]; then
  open -a Terminal "$RES/setup.command"
  exit 0
fi
export GGS_ROOT="$ROOT"
exec "$ROOT/env/bin/python" -m ggs.ui.app
"""
SETUP_COMMAND = """#!/bin/bash
# Installs GGS's Python, libraries, motion library and the AI models a Mac can run into
# ~/Library/Application Support/GGS (tools/setup_env.py). Run again to repair.
RES="$(cd "$(dirname "$0")" && pwd)"
APP="$(cd "$RES/../.." && pwd)"
ROOT="$HOME/Library/Application Support/GGS"
echo "Setting up GGS in $ROOT"
echo "(Python, the libraries, the motion library and the AI models: a few GB, 5-20 minutes)"
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true
mkdir -p "$ROOT"
rsync -a --delete --exclude /env --exclude /python --exclude /.uv-cache --exclude /.setup_ok \\
  --exclude /motions/catalogs --exclude /motions/clips --exclude /motions/embeds "$RES/app/" "$ROOT/"
export UV_PYTHON_INSTALL_DIR="$ROOT/python" UV_CACHE_DIR="$ROOT/.uv-cache" UV_PYTHON_PREFERENCE=only-managed
"$RES/uv" run --no-project --python 3.13 "$ROOT/tools/setup_env.py" --app --root "$ROOT" --models required
OK=$?
"$RES/uv" cache clean >/dev/null 2>&1  # the download cache is not needed once the environment is built
if [ "$OK" = 0 ]; then
  echo
  echo "GGS is ready. Starting it..."
  open "$APP"
else
  echo
  echo "Setup did not finish. Open GGS again to retry (finished steps are skipped)."
fi
"""
PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>GGS</string>
  <key>CFBundleDisplayName</key><string>GGS - Game Graphic Studio</string>
  <key>CFBundleIdentifier</key><string>io.github.shahind.ggs</string>
  <key>CFBundleVersion</key><string>{ver}</string>
  <key>CFBundleShortVersionString</key><string>{ver}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>GGS</string>
  <key>CFBundleIconFile</key><string>GGS</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"""


def make_icns(res: Path) -> None:
    iconset = res / "GGS.iconset"
    iconset.mkdir()
    for px in (16, 32, 128, 256, 512):
        render_logo(px).save(iconset / f"icon_{px}x{px}.png")
        render_logo(px * 2).save(iconset / f"icon_{px}x{px}@2x.png")
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(res / "GGS.icns")], check=True)
    shutil.rmtree(iconset)


def stage_mac(out: Path) -> Path:
    if sys.platform != "darwin":
        raise SystemExit("the macOS app is built on a Mac (GitHub Actions: .github/workflows/installers.yml)")
    ver = version()
    bundle = out / "dmg" / "GGS.app"
    if bundle.parent.exists():
        shutil.rmtree(bundle.parent)
    contents = bundle / "Contents"
    res = contents / "Resources"
    stage_source(res / "app")
    fetch_uv("mac", res)
    (contents / "MacOS").mkdir(parents=True)
    for path, text in ((contents / "MacOS" / "GGS", LAUNCHER), (res / "setup.command", SETUP_COMMAND)):
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
    (contents / "Info.plist").write_text(PLIST.format(ver=ver), encoding="utf-8")
    make_icns(res)
    # the usual drag-to-install window: GGS.app next to a link to /Applications (macOS only, inside the image)
    (bundle.parent / "Applications").symlink_to("/Applications")
    dmg = ROOT / "dist" / f"GGS-{ver}-macos-arm64.dmg"
    dmg.unlink(missing_ok=True)
    subprocess.run(["hdiutil", "create", "-volname", f"GGS {ver}", "-srcfolder", str(bundle.parent), "-ov",
                    "-format", "UDZO", str(dmg)], check=True)
    print(f"built {dmg}")
    return dmg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--windows", action="store_true")
    g.add_argument("--mac", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "dist" / "stage"))
    a = ap.parse_args()
    out = Path(a.out)
    if a.windows:
        stage_windows(out)
    else:
        stage_mac(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
