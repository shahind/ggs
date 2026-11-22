"""Model packs: which are ready, and a resumable downloader (``ggs models status | fetch``).

The downloader lists a repo's files on the Hugging Face hub, keeps the ones a pack includes, and fetches each
with HTTP range requests into ``<file>.part``: a dropped connection or a stall resumes where it stopped (the hub
library's own resume proved unreliable on flaky networks). Progress goes to stdout as ``@progress <0-100>``
for the app's progress bar.
"""

from __future__ import annotations

import fnmatch
import time
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ggs.paths import repo_root, shared_dir

HUB = "https://huggingface.co"


class Part(BaseModel):
    repo: str
    dir: str
    include: list[str] = Field(default_factory=lambda: ["*"])
    revision: str = "main"


class Pack(BaseModel):
    label: str
    why: str = ""
    required: bool = False
    size_gb: float = 0.0
    ready_when: dict[str, list[str]] = Field(default_factory=dict)
    parts: list[Part]
    needs_cuda: bool = False  # runs only on an NVIDIA GPU: not offered on a Mac or a PC without one


class Library(BaseModel):
    """The motion library zip (a release download; tools/pack_motions.py)."""

    url: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _manifest(path: Path | None = None) -> tuple[Path, dict]:
    path = path or repo_root() / "models" / "downloads.yaml"
    try:
        return path, yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_packs(path: Path | None = None) -> dict[str, Pack]:
    path, data = _manifest(path)
    try:
        return {k: Pack.model_validate(v) for k, v in data.get("packs", {}).items()}
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_library(path: Path | None = None) -> Library | None:
    path, data = _manifest(path)
    if not data.get("library"):
        return None
    try:
        return Library.model_validate(data["library"])
    except Exception as exc:
        raise ValueError(f"{path}: library: {exc}") from exc


def required_packs(packs: dict[str, Pack], cuda: bool, cloud: bool) -> list[str]:
    """The packs this computer needs: required ones it can run. A Google API key stands in for the local
    drawing model (``CLOUD_REPLACES``)."""
    return [n for n, p in packs.items()
            if p.required and (cuda or not p.needs_cuda) and not (cloud and n in CLOUD_REPLACES)]


CLOUD_REPLACES = ("drawing",)


def library_ready(root: Path | None = None) -> bool:
    m = (root or repo_root()) / "motions"
    return any((m / "catalogs").glob("*.zip")) and any((m / "clips").glob("*.json"))


def fetch_library(root: Path | None = None, progress=print, lib: Library | None = None) -> bool:
    """Download and unpack the motion library into ``<root>/motions`` unless it is there. True if fetched."""
    import hashlib
    import zipfile

    root = root or repo_root()
    if library_ready(root):
        return False
    lib = lib or load_library()
    if lib is None:
        raise OSError("models/downloads.yaml has no motion library download")
    zpath = shared_dir() / "cache" / Path(lib.url).name
    state = {"done": zpath.stat().st_size if zpath.exists() else 0, "last": -1}

    def tick(n: int) -> None:
        state["done"] += n
        pct = int(100 * state["done"] / lib.size)
        if pct != state["last"]:
            state["last"] = pct
            progress(f"@progress {min(pct, 100)}")

    progress(f"fetching the motion library ({lib.size / 2**20:.0f} MB)")
    fetch_file(lib.url, zpath, lib.size, on_bytes=tick)
    h = hashlib.sha256()
    with open(zpath, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != lib.sha256:
        zpath.unlink()
        raise OSError(f"the motion library download is damaged (checksum {h.hexdigest()[:12]}...); run it again")
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.startswith("motions/") or ".." in Path(name).parts:
                raise OSError(f"unexpected file in the motion library zip: {name}")
        z.extractall(root)
    zpath.unlink()
    progress("@progress 100")
    return True


def bundled_files(path: Path | None = None) -> list[str]:
    """Model files (relative to a models folder) that the installer ships."""
    path, data = _manifest(path)
    files = data.get("bundled") or []
    if not all(isinstance(f, str) for f in files):
        raise ValueError(f"{path}: bundled must be a list of relative file paths")
    return files


def pack_ready(pack: Pack) -> bool:
    from ggs.pipeline.models import (
        MissingModelError,
        load_pipelines,
        resolve_captioner,
        resolve_frames,
        resolve_matting,
        resolve_motion,
        resolve_motion_text,
        resolve_pipeline,
        resolve_pose,
    )

    spec = load_pipelines()
    try:
        for name in pack.ready_when.get("pipelines", []):
            resolve_pipeline(name, "lite", spec_file=spec)
        for name in pack.ready_when.get("matting", []):
            resolve_matting(name, spec_file=spec)
        for name in pack.ready_when.get("frames", []):
            resolve_frames(name, spec_file=spec)
        for name in pack.ready_when.get("pose", []):
            resolve_pose(name, spec_file=spec)
        for name in pack.ready_when.get("captioning", []):
            resolve_captioner(name, spec_file=spec)
        for name in pack.ready_when.get("motion", []):
            resolve_motion(name, spec_file=spec)
        for name in pack.ready_when.get("motion_text", []):
            resolve_motion_text(name, spec_file=spec)
    except (MissingModelError, ValueError):
        return False
    return True


def list_files(part: Part) -> list[tuple[str, int]]:
    """(path in repo, size) of the files the part includes."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(part.repo, revision=part.revision, files_metadata=True)
    out = []
    for s in info.siblings:
        if any(fnmatch.fnmatch(s.rfilename, pat) for pat in part.include) and not s.rfilename.startswith("."):
            out.append((s.rfilename, int(s.size or 0)))
    return out


def fetch_file(url: str, dest: Path, size: int, on_bytes=None, retries: int = 60, chunk: int = 1 << 20,
               timeout: float = 30.0) -> None:
    """Download ``url`` to ``dest`` with resume; ``on_bytes(n)`` is called with each new chunk's length."""
    import requests

    if dest.exists() and (size == 0 or dest.stat().st_size == size):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if size and have >= size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(timeout, timeout)) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
                        if on_bytes:
                            on_bytes(len(block))
            if not size or part.stat().st_size >= size:
                break
        except (requests.RequestException, OSError) as exc:
            if attempt == retries - 1:
                raise OSError(f"download of {url} kept failing: {exc}") from exc
            time.sleep(min(30, 2 + attempt))
    part.replace(dest)


def fetch_pack(name: str, pack: Pack, models_dir: Path | None = None, progress=print) -> list[Path]:
    """Download every file of a pack (skipping ones already complete)."""
    root = models_dir or shared_dir() / "models"
    plan = []
    for part in pack.parts:
        for f, size in list_files(part):
            plan.append((part, f, size))
    total = sum(s for _, _, s in plan) or 1
    done = sum(min((root / p.dir / f).stat().st_size, s) for p, f, s in plan if (root / p.dir / f).exists())
    state = {"done": done, "last": -1}

    def tick(n: int) -> None:
        state["done"] += n
        pct = int(100 * state["done"] / total)
        if pct != state["last"]:
            state["last"] = pct
            progress(f"@progress {pct}")

    out = []
    for p, f, size in plan:
        dest = root / p.dir / f
        progress(f"fetching {p.repo}/{f} ({size / 2**20:.0f} MB)")
        url = f"{HUB}/{p.repo}/resolve/{p.revision}/{f}"
        fetch_file(url, dest, size, on_bytes=tick)
        out.append(dest)
    progress("@progress 100")
    return out
