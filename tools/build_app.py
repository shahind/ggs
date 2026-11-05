"""Build the standalone app folder ``dist/GGS`` (then ``tools/installer.iss`` packs it into a setup .exe).

Run with the project's venv python (the one that has the CUDA torch and the ``gpu``/``ui`` extras installed):

    .venv\\Scripts\\python tools\\build_app.py

Layout (no main AI models; the app downloads them on first run, see models/downloads.yaml):

    dist/GGS/
      python/            a private copy of the base Python + the venv's site-packages + the ggs package
      recipes/ rigs/ clips/ models/*.yaml   data (ggs.paths.repo_root() finds them next to python/)
      models/<dir>/...   the bundled LoRAs (downloads.yaml `bundled`), searched after ~/.ggs/models
      ggs.cmd            command line:  ggs.cmd generate --class sprite ...
      Start GGS.cmd      the desktop app (the installer makes shortcuts to python/pythonw.exe -m ggs.ui.app)
      GGS.ico, licenses/

Why not PyInstaller: torch, diffusers/transformers lazy imports and BiRefNet's remote code (loaded from the
Hugging Face module cache at run time) all want a real Python on disk; a portable Python avoids freezing bugs.
Small model files listed under ``bundled`` in models/downloads.yaml (the LoRAs) go into ``models/``.
A ``python313._pth`` file isolates the bundled Python from any Python, PYTHONPATH or user site on the machine.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASE_SKIP = ["Doc", "include", "libs", "Scripts", "tcl", "Lib/site-packages", "Lib/test", "Lib/idlelib",
             "Lib/tkinter", "Lib/turtledemo", "Lib/ensurepip", "DLLs/_tkinter.pyd", "DLLs/tcl*.dll",
             "DLLs/tk*.dll", "NEWS.txt"]
SITE_SKIP = ["__editable__*", "ggs-*.dist-info", "pytest*", "_pytest", "pluggy*", "iniconfig*", "ruff*", "pip",
             "pip-*.dist-info", "torch/include", "torch/lib/*.lib", "torch/test", "pkg_resources/tests", "setuptools/tests", "torchgen", "*/__pycache__",
             "__pycache__"]
# Qt: the app only uses QtCore/QtGui/QtWidgets; drop WebEngine, QML/Quick, 3D, multimedia and the tools (~600 MB)
QT_KEEP_MODULES = {"Core", "Gui", "Widgets", "Svg", "Network", "OpenGL", "OpenGLWidgets", "SvgWidgets"}
QT_KEEP_PLUGINS = {"platforms", "imageformats", "styles", "iconengines", "generic", "platforminputcontexts", "tls"}
QT_SKIP_DIRS = {"resources", "qml", "metatypes", "include", "typesystems", "glue", "doc", "scripts", "support"}
QT_SKIP_FILES = ["av*.dll", "sw*.dll", "opengl32sw.dll", "*.exe"]
DATA = ["recipes", "rigs", "clips", "motions"]
MODEL_YAML = ["registry.yaml", "pipelines.yaml", "downloads.yaml"]


def _skip(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def copy_tree(src: Path, dst: Path, skip: list[str]) -> int:
    """Copy src to dst, skipping paths (relative, '/'-separated) that match a pattern. Returns bytes copied."""
    total = 0
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src).as_posix()
        if _skip(rel, skip) or any(_skip("/".join(rel.split("/")[: i + 1]), skip)
                                   for i in range(rel.count("/"))):
            continue
        out = dst / rel
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            total += p.stat().st_size
    return total


def trim_qt(qt: Path) -> int:
    """Remove the Qt modules GGS does not use. Returns bytes removed."""
    removed = 0

    def rm(p: Path) -> None:
        nonlocal removed
        if p.is_dir():
            removed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p)
        elif p.exists():
            removed += p.stat().st_size
            p.unlink()

    for p in list(qt.iterdir()):
        n = p.name
        if p.is_dir():
            if n in QT_SKIP_DIRS:
                rm(p)
            elif n == "plugins":
                for sub in p.iterdir():
                    if sub.name not in QT_KEEP_PLUGINS:
                        rm(sub)
            elif n == "translations":
                for f in p.iterdir():
                    if not f.name.startswith("qtbase_"):
                        rm(f)
            continue
        if n.startswith("Qt6") and n.endswith(".dll") and n[3:-4] not in QT_KEEP_MODULES or n.startswith("Qt") and n.endswith((".pyd", ".pyi")) and n[2:].split(".")[0] not in QT_KEEP_MODULES or any(fnmatch.fnmatch(n, pat) for pat in QT_SKIP_FILES):
            rm(p)
    return removed


def bundle_models(dest: Path) -> None:
    """Copy the small model files listed under ``bundled`` in models/downloads.yaml (LoRAs) into the app."""
    from ggs.pipeline.download import bundled_files
    from ggs.pipeline.models import model_dirs

    dirs = [d for d in model_dirs() if d.resolve() != (ROOT / "models").resolve()]
    for rel in bundled_files(ROOT / "models" / "downloads.yaml"):
        src = next((d / rel for d in dirs if (d / rel).is_file()), None)
        if src is None:
            raise SystemExit(f"bundled model file {rel} not found in {[str(d) for d in dirs]}")
        out = dest / rel
        if not out.exists() or out.stat().st_size != src.stat().st_size:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
        print(f"bundled {rel} ({src.stat().st_size / 2**20:.0f} MB)")


def make_icon(out: Path) -> None:
    from PIL import Image

    src = ROOT / "src" / "ggs" / "ui" / "icons" / "assets@2x.png"
    img = Image.open(src).convert("RGBA")
    img.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128)])


def build(dist: Path, skip_site: bool = False) -> Path:
    base = Path(sys.base_prefix)
    site = Path(sysconfig.get_paths()["purelib"])
    if Path(sys.prefix) == base:
        raise SystemExit("run this with the project's venv python (.venv\\Scripts\\python tools\\build_app.py)")
    app = dist / "GGS"
    py = app / "python"
    if app.exists() and not skip_site:
        shutil.rmtree(app)
    app.mkdir(parents=True, exist_ok=True)

    print(f"base python {base} -> {py}")
    copy_tree(base, py, BASE_SKIP)
    psite = py / "Lib" / "site-packages"
    if not skip_site:
        print(f"site-packages {site} (this takes a few minutes: torch is ~4 GB)")
        n = copy_tree(site, psite, SITE_SKIP)
        print(f"  {n / 2**30:.2f} GB copied")
        print(f"  Qt trimmed: {trim_qt(psite / 'PySide6') / 2**20:.0f} MB removed")
    else:
        # quick build: still bring in packages added to the venv since the last full build (new dependencies)
        for p in sorted(site.iterdir()):
            if not _skip(p.name, SITE_SKIP) and not (psite / p.name).exists():
                print(f"  new package {p.name}")
                if p.is_dir():
                    copy_tree(p, psite / p.name, SITE_SKIP)
                else:
                    shutil.copy2(p, psite / p.name)
    if (psite / "ggs").exists():
        shutil.rmtree(psite / "ggs")
    copy_tree(ROOT / "src" / "ggs", psite / "ggs", ["__pycache__", "*/__pycache__"])
    ver = f"{sys.version_info.major}{sys.version_info.minor}"
    (py / f"python{ver}._pth").write_text("Lib\nDLLs\nLib\\site-packages\nimport site\n", encoding="utf-8")
    subprocess.run([str(py / "python.exe"), "-m", "compileall", "-q", str(psite / "ggs")], check=True)

    missing = [p for p in ("motions/catalogs/cmu.zip", "motions/catalogs/100style.zip", "motions/clips/walk.json")
               if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit(f"motion library not built ({missing}): run tools/make_motion_catalogs.py (cmu, 100style) "
                         "and tools/make_motion_library.py first")
    for d in DATA:
        if (app / d).exists():
            shutil.rmtree(app / d)
        copy_tree(ROOT / d, app / d, ["__pycache__"])
    (app / "models").mkdir(exist_ok=True)
    for f in MODEL_YAML:
        shutil.copy2(ROOT / "models" / f, app / "models" / f)
    bundle_models(app / "models")
    lic = app / "licenses"
    lic.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "docs" / "LICENSE_REVIEW.md", lic / "LICENSE_REVIEW.md")
    shutil.copy2(base / "LICENSE.txt", lic / "Python-LICENSE.txt")
    shutil.copy2(ROOT / "motions" / "CREDITS.md", lic / "Motion-CREDITS.md")
    (app / "ggs.cmd").write_text('@echo off\r\n"%~dp0python\\python.exe" -m ggs %*\r\n', encoding="utf-8")
    (app / "Start GGS.cmd").write_text('@echo off\r\nstart "" "%~dp0python\\pythonw.exe" -m ggs.ui.app %*\r\n',
                                 encoding="utf-8")
    make_icon(app / "GGS.ico")
    from ggs import __version__

    (app / "VERSION").write_text(__version__, encoding="utf-8")
    return app


def check(app: Path) -> None:
    """The bundled Python must import everything from its own folder, never from the venv or the repo."""
    py = app / "python" / "python.exe"
    code = ("import sys, ggs, torch, diffusers, transformers, PySide6.QtWidgets, PIL, spellchecker, cv2, ggs.paths as p;"
            "print(sys.prefix); print(ggs.__file__); print(p.repo_root());"
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
    out = subprocess.run([str(py), "-c", code], check=False, capture_output=True, text=True, cwd=str(app.parent))
    print(out.stdout + out.stderr)
    if out.returncode or str(ROOT / "src") in out.stdout or str(app) not in out.stdout:
        raise SystemExit("bundle check failed")
    for args in (["rig", "validate", str(app / "rigs" / "humanoid_side_v1.yaml")], ["models", "status"]):
        r = subprocess.run([str(py), "-m", "ggs", *args], check=False, capture_output=True, text=True, cwd=str(app.parent))
        print(" ".join(args[:2]), "->", r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:])
    # the check wrote byte-code caches (~35 MB) into the bundle; only ggs's own (compiled on purpose) stay
    keep = app / "python" / "Lib" / "site-packages" / "ggs"
    for d in [d for d in (app / "python").rglob("__pycache__") if keep not in d.parents]:
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dist", type=Path, default=ROOT / "dist")
    ap.add_argument("--code-only", action="store_true", help="refresh only the ggs package and data (fast)")
    a = ap.parse_args()
    app = build(a.dist, skip_site=a.code_only)
    check(app)
    size = sum(f.stat().st_size for f in app.rglob("*") if f.is_file())
    print(f"built {app}  ({size / 2**30:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
