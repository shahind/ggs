"""Install NVIDIA Kimodo's pure-Python package into this environment (``pip install`` of the repo wants CMake
for an optional C++ foot-contact extension that GGS does not use).

    .venv/Scripts/python tools/install_kimodo.py   (Windows: the same with backslashes)

Downloads the source from GitHub (Apache-2.0), copies the ``kimodo`` package to site-packages without the demo
assets and the G1 robot skeleton meshes (~56 MB), then installs the ``motion`` extra's dependencies.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

REPO = "git+https://github.com/nv-tlabs/kimodo.git"
SKIP = ("assets/demo", "assets/skeletons/g1skel34")


def main() -> int:
    site = Path(sysconfig.get_paths()["purelib"])
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--no-binary", ":all:", "-d", tmp,
                        REPO], check=True)
        archive = next(Path(tmp).glob("kimodo-*.zip"))
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        src = next(p for p in Path(tmp).rglob("kimodo") if (p / "__init__.py").is_file() and p.parent.name != "src")
        dest = site / "kimodo"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
        for rel in SKIP:
            shutil.rmtree(dest / rel, ignore_errors=True)
        shutil.copy2(src.parent / "LICENSE", dest / "LICENSE")
    subprocess.run([sys.executable, "-m", "pip", "install", "hydra-core>=1.3", "omegaconf>=2.3", "scipy>=1.10",
                    "einops>=0.7", "bvhio"], check=True)
    print(f"installed kimodo into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
