"""The starter animation library (``motions/library.yaml``) and turning a 3D take into a 2D clip.

Library clips live in ``motions/clips/<name>.json`` (+ ``<name>.gif`` preview); a project's own animations live in
``<project>/clips``. Generated takes keep a ``<clip>.motion.json`` sidecar with the prompt, seconds and seed,
so the same take can be made again.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from ggs.paths import repo_root
from ggs.rig.clip import Clip

CLIP_FPS = 24.0


class LibraryEntry(BaseModel):
    name: str
    label: str
    category: str
    prompt: str
    seconds: float
    seed: int = 0
    loop: bool = False
    in_place: bool = False
    loop_seconds: tuple[float, float] = (0.6, 1.6)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not v.replace("_", "").isalnum() or v.lower() != v:
            raise ValueError(f"motion name {v!r}: use lower-case letters, digits and _")
        return v


class Library(BaseModel):
    format: str
    version: int
    motions: list[LibraryEntry] = Field(default_factory=list)

    @field_validator("format")
    @classmethod
    def _format(cls, v: str) -> str:
        if v != "ggs.motions":
            raise ValueError(f"format must be 'ggs.motions', got {v!r}")
        return v

    def get(self, name: str) -> LibraryEntry:
        for m in self.motions:
            if m.name == name:
                return m
        raise KeyError(f"no motion {name!r} in the library; known: {[m.name for m in self.motions]}")


def library_dir() -> Path:
    return repo_root() / "motions"


def load_library(path: Path | None = None) -> Library:
    path = path or library_dir() / "library.yaml"
    try:
        lib = Library.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc
    names = [m.name for m in lib.motions]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"{path}: duplicate motion names {dup}")
    return lib


def library_clip(name: str) -> Path:
    return library_dir() / "clips" / f"{name}.json"


def sidecar(clip_path: Path) -> Path:
    return Path(clip_path).with_suffix(".motion.json")


def to_clip(bvh: Path, rig, *, loop: bool = False, in_place: bool = False,
            loop_seconds: tuple[float, float] = (0.6, 1.6), fps: float = CLIP_FPS) -> Clip:
    """Retarget a Kimodo BVH (rig profile ``soma``) and optionally cut a seamless loop out of it."""
    from ggs.rig.bvh import load_bvh
    from ggs.rig.cliptools import cut_loop, find_loop
    from ggs.rig.retarget import retarget

    clip = retarget(load_bvh(bvh), rig, "soma", fps=fps, in_place=in_place, drop_tpose_frame=False)
    if loop:
        lo, hi = loop_seconds
        hi = min(hi, clip.duration * 0.9)
        start, end, _ = find_loop(clip, min(lo, hi), hi)
        warnings = list(clip.warnings)
        clip = cut_loop(clip, start, end)
        clip.warnings = warnings + list(clip.warnings)
    return clip


def write_sidecar(clip_path: Path, info: dict) -> Path:
    p = sidecar(clip_path)
    p.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return p


def add_to(names: list[str], dest: Path, lib: Library | None = None) -> list[Path]:
    """Copy library motions (clip, preview, sidecar) into a folder, e.g. a project's clips folder."""
    lib = lib or load_library()
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for n in names:
        lib.get(n)
        src = library_clip(n)
        if not src.is_file():
            raise FileNotFoundError(f"library motion {n!r} has no clip yet ({src}); run tools/make_motion_library.py")
        for f in (src, src.with_suffix(".gif"), sidecar(src)):
            if f.is_file():
                shutil.copy2(f, dest / f.name)
        out.append(dest / src.name)
    return out
