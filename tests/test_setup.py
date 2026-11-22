"""Setup and installers without the network: pinned versions, the packs a computer needs, the motion library
download, and the staged app folder."""

import hashlib
import importlib.util
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ggs.pipeline import download as dl

ROOT = Path(__file__).resolve().parents[1]


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_constraints_pin_torch_without_a_cuda_tag():
    pins = _tool("setup_env").pins(ROOT / "requirements" / "constraints.txt")
    assert pins["torch"] and "+" not in pins["torch"] and pins["torchvision"]
    assert "ggs" not in pins


def test_required_packs_follow_the_hardware():
    packs = dl.load_packs()
    assert dl.required_packs(packs, cuda=True, cloud=False) == ["drawing", "cutout"]
    assert dl.required_packs(packs, cuda=False, cloud=False) == ["cutout"]  # a Mac: drawing is CUDA-only
    assert dl.required_packs(packs, cuda=True, cloud=True) == ["cutout"]  # a Google key stands in for drawing
    assert all(packs[n].needs_cuda for n in ("drawing", "sdxl", "motion", "frames_o2a"))
    assert not packs["cutout"].needs_cuda
    sdxl = {p.dir for p in packs["sdxl"].parts}
    assert "SDXL-Lightning" in sdxl  # no longer only bundled by the full installer


def test_motion_library_download_checks_and_unpacks(tmp_path, monkeypatch):
    src = tmp_path / "lib.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("motions/catalogs/cmu.zip", b"x")
        z.writestr("motions/clips/walk.json", b"{}")
    data = src.read_bytes()
    lib = dl.Library(url="https://example.invalid/ggs-motions-v1.zip", size=len(data),
                     sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(dl, "shared_dir", lambda: tmp_path / "home")
    monkeypatch.setattr(dl, "fetch_file", lambda url, dest, size, on_bytes=None: (
        dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(data), on_bytes and on_bytes(len(data))))
    root = tmp_path / "app"
    assert not dl.library_ready(root)
    assert dl.fetch_library(root, progress=lambda m: None, lib=lib)
    assert dl.library_ready(root) and (root / "motions" / "clips" / "walk.json").exists()
    assert not dl.fetch_library(root, progress=lambda m: None, lib=lib)  # already there

    bad = dl.Library(url=lib.url, size=lib.size, sha256="0" * 64)
    with pytest.raises(OSError, match="damaged"):
        dl.fetch_library(tmp_path / "other", progress=lambda m: None, lib=bad)


def test_shipped_library_entry_is_valid():
    lib = dl.load_library()
    assert lib is not None and lib.url.startswith("https://github.com/") and lib.size > 1_000_000


def test_staged_app_has_what_setup_needs(tmp_path):
    stage = _tool("stage_app")
    app = tmp_path / "GGS"
    stage.stage_source(app)
    for rel in ("pyproject.toml", "VERSION", "requirements/constraints.txt", "tools/setup_env.py",
                "src/ggs/cli.py", "recipes/sprite.yaml", "rigs/humanoid_side_v1.yaml", "models/downloads.yaml",
                "motions/library.yaml", "licenses/LICENSE_REVIEW.md"):
        assert (app / rel).is_file(), rel
    assert not list(app.rglob("__pycache__"))
    assert (app / "VERSION").read_text(encoding="utf-8") == stage.version()
    total = sum(p.stat().st_size for p in app.rglob("*") if p.is_file())
    assert total < 15 * 2**20  # small: no models, no motion library, no Python


def test_setup_ui_hides_cuda_packs_without_a_gpu():
    pytest.importorskip("PySide6")
    from ggs.ui.setup import needed

    assert not needed("drawing", SimpleNamespace(required=True, needs_cuda=True), cloud=False, cuda=False)
