"""Set up GGS on this computer: Python, the libraries, the motion library and (optionally) the AI models.

It runs through uv, which brings its own Python, so nothing needs to be installed first. ``setup.cmd`` (Windows) and
``setup.sh`` (macOS / Linux) fetch uv and run this; the Windows installer and the macOS app run it too:

    uv run --no-project --python 3.13 tools/setup_env.py                      # source checkout: .venv + dev tools
    uv run --no-project --python 3.13 tools/setup_env.py --app --root <dir> --models required   # installed app

Steps; each one is skipped when it is already done, so running it again repairs or updates an install:
  1. a virtual environment on Python 3.13 (``<root>/env`` for --app, else ``<repo>/.venv``)
  2. torch: the CUDA 12.8 build when an NVIDIA GPU is found (Windows / Linux), else the standard build
     (CPU, or Apple's GPU on a Mac)
  3. GGS and its libraries at the tested versions (``requirements/constraints.txt``)
  4. the motion library (``ggs motion fetch``: a release download checked by SHA-256)
  5. the AI models this computer needs (``ggs models fetch required``) with ``--models required``

Standard library only: this runs before any package is installed.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = "3.13"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def say(text: str) -> None:
    print(f"\n=== {text}", flush=True)


def run(cmd: list[str], env: dict | None = None) -> None:
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    r = subprocess.run(cmd, check=False, env=env)
    if r.returncode != 0:
        raise SystemExit(f"\nSetup stopped: the step above failed (exit {r.returncode}). Run setup again to retry; "
                         "finished steps are skipped.")


def find_uv(given: str | None) -> str:
    for c in (given, os.environ.get("UV"), shutil.which("uv")):
        if c and Path(c).exists():
            return c
    raise SystemExit("uv was not found. Run setup.cmd (Windows) or setup.sh (macOS / Linux), which fetch it.")


def nvidia_gpu() -> bool:
    if sys.platform == "darwin":
        return False
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "-L"], check=False, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "GPU" in out


def pins(constraints: Path) -> dict[str, str]:
    out = {}
    for line in constraints.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def env_python(env: Path) -> Path:
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def version(root: Path) -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE)
    return m.group(1) if m else "0"


def main() -> int:
    ap = argparse.ArgumentParser(description="Set up GGS (Python, libraries, motion library, models).")
    ap.add_argument("--app", action="store_true", help="an installed app: <root>/env, no dev tools, not editable")
    ap.add_argument("--root", default=None, help="folder with pyproject.toml, src/ and the data folders "
                                                 "(default: this checkout)")
    ap.add_argument("--models", choices=["none", "required"], default="none",
                    help="also download the AI models this computer needs (about 11 GB with an NVIDIA GPU)")
    ap.add_argument("--cpu", action="store_true", help="install the CPU build of torch even with an NVIDIA GPU")
    ap.add_argument("--uv", default=None, help="path to uv (default: the one running this script)")
    ap.add_argument("--pause", action="store_true", help="on an error, wait for Enter before closing (a window "
                                                       "the installer opened)")
    a = ap.parse_args()
    try:
        return setup(a)
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(exc.code if isinstance(exc.code, str) else f"Setup failed ({exc.code}).", file=sys.stderr)
            if a.pause:
                input("\nPress Enter to close this window.")
            return 1
        raise


def setup(a: argparse.Namespace) -> int:
    t0 = time.time()
    root = Path(a.root).resolve() if a.root else REPO
    uv = find_uv(a.uv)
    env_dir = root / "env" if a.app else root / ".venv"
    py = env_python(env_dir)
    constraints = root / "requirements" / "constraints.txt"
    pinned = pins(constraints)
    gpu = nvidia_gpu() and not a.cpu
    print(f"GGS {version(root)} setup on {platform.system()} {platform.machine()}; folder {root}")
    print(f"NVIDIA GPU: {'yes' if gpu else 'no'}")

    say(f"1/5 Python {PYTHON} environment ({env_dir})")
    if py.exists():
        print("  already there")
    else:
        run([uv, "venv", str(env_dir), "--python", PYTHON, "--seed"])
    pip = [uv, "pip", "install", "--python", str(py)]

    say("2/5 PyTorch " + ("(CUDA 12.8 build, about 2.5 GB)" if gpu else "(standard build)"))
    torch = [f"torch=={pinned['torch']}", f"torchvision=={pinned['torchvision']}"]
    if gpu:
        run([*pip, *torch, "--index-url", TORCH_CUDA_INDEX])
    else:
        run([*pip, *torch])

    say("3/5 GGS and its libraries")
    extras = "gpu,image,ui" + ("" if a.app else ",dev")
    target = f"{root}[{extras}]"
    run([*pip, "-c", str(constraints), *(["-e", target] if not a.app else [target]),
         *(["--reinstall-package", "ggs"] if a.app else [])])

    env = {**os.environ, "GGS_ROOT": str(root), "PYTHONUTF8": "1"}
    say("4/5 Motion library")
    run([str(py), "-m", "ggs", "motion", "fetch"], env)

    say("5/5 AI models")
    if a.models == "required":
        run([str(py), "-m", "ggs", "models", "fetch", "required"], env)
    else:
        print("  skipped: the app offers them on first start (or run: ggs models fetch required)")
    run([str(py), "-m", "ggs", "doctor"], env)

    (root / ".setup_ok").write_text(version(root), encoding="utf-8")
    mins = (time.time() - t0) / 60
    say(f"Done in {mins:.1f} min")
    start = (f'"{env_dir / "Scripts" / "pythonw.exe"}" -m ggs.ui.app' if os.name == "nt"
             else f'"{py}" -m ggs.ui.app')
    print(f"Start the app with:\n  {start}")
    cache = os.environ.get("UV_CACHE_DIR")
    if a.app and cache and Path(cache).resolve().is_relative_to(root):  # installed app: drop the download cache
        subprocess.run([uv, "cache", "clean"], check=False, capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
