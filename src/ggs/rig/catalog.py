"""Motion catalogs: big ready-made sets of 2D clips converted from free motion capture (CMU, 100STYLE).

Each catalog is one zip, ``motions/catalogs/<source>.zip``:
    index.json          {"format": "ggs.catalog", "version": 1, "source", "title", "credit", "license",
                         "motions": [{"id", "name", "category", "tags", "seconds", "frames", "loop", ...}]}
    clips/<id>.json     a compact ggs.clip for humanoid_side_v1: absolute angles at 0.1 degree, no
                        local angles (rebuilt from the bone parents on load), swaps as "0"/"1" strings
Zipped JSON is ~4x smaller than loose files and needs no unpacking: the app reads clips straight from the zip,
draws previews on demand (cached in ~/.ggs/cache/previews), and copies a clip into a project when it is used.
Made by ``tools/make_motion_catalogs.py``; credits in ``motions/CREDITS.md``.
"""

from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from ggs.paths import repo_root, shared_dir
from ggs.rig.clip import Clip


class CatalogMotion(BaseModel):
    id: str
    name: str
    category: str
    tags: list[str] = Field(default_factory=list)
    seconds: float
    frames: int
    loop: bool = False
    in_place: bool = False
    note: str = ""


class CatalogIndex(BaseModel):
    format: str = "ggs.catalog"
    version: int = 1
    source: str
    title: str
    credit: str
    license: str
    url: str = ""
    rig: str = "humanoid_side_v1"
    motions: list[CatalogMotion]


def catalogs_dir() -> Path:
    return repo_root() / "motions" / "catalogs"


def catalog_files() -> list[Path]:
    d = catalogs_dir()
    return sorted(d.glob("*.zip")) if d.is_dir() else []


@lru_cache(maxsize=8)
def _index(path: str, mtime: float) -> CatalogIndex:
    with zipfile.ZipFile(path) as z:
        try:
            return CatalogIndex.model_validate_json(z.read("index.json"))
        except Exception as exc:
            raise ValueError(f"{path}: index.json: {exc}") from exc


def load_index(path: Path) -> CatalogIndex:
    return _index(str(path), path.stat().st_mtime)


def all_motions() -> list[tuple[CatalogIndex, CatalogMotion]]:
    """(catalog, motion) for every motion of every installed catalog."""
    out = []
    for f in catalog_files():
        idx = load_index(f)
        out.extend((idx, m) for m in idx.motions)
    return out


def _path_of(source: str) -> Path:
    p = catalogs_dir() / f"{source}.zip"
    if not p.is_file():
        raise FileNotFoundError(f"no motion catalog {source!r} (looked for {p})")
    return p


def find(ref: str) -> tuple[CatalogIndex, CatalogMotion]:
    """``ref`` = ``<source>/<id>``, e.g. ``cmu/02_01``."""
    source, _, mid = ref.partition("/")
    idx = load_index(_path_of(source))
    for m in idx.motions:
        if m.id == mid:
            return idx, m
    raise KeyError(f"no motion {mid!r} in catalog {source!r}")


def compact(clip: Clip) -> dict:
    """The clip as stored in a catalog (about half the size of a full clip once zipped)."""
    d = clip.model_dump()
    d.pop("local_angles")
    d["angles"] = {b: [round(v, 1) for v in t] for b, t in d["angles"].items()}
    d["root"] = {k: [round(v, 4) for v in t] for k, t in d["root"].items()}
    d["bone_lengths"] = {b: round(v, 4) for b, v in d["bone_lengths"].items()}
    d["swaps"] = {g: "".join("1" if s else "0" for s in t) for g, t in d["swaps"].items()}
    return d


@lru_cache(maxsize=4)
def _parents(rig: str) -> dict[str, str | None]:
    from ggs.rig import load_rig_spec

    return {b.name: b.parent for b in load_rig_spec(repo_root() / "rigs" / f"{rig}.yaml").bones}


def expand(d: dict) -> Clip:
    """A full clip from its catalog form."""
    import numpy as np

    d = dict(d)
    parents = _parents(d["rig"])
    ang = {b: np.asarray(v, dtype=float) for b, v in d["angles"].items()}

    def wrap(x):
        return -((-x + 180.0) % 360.0 - 180.0)

    d["local_angles"] = {b: [round(float(x), 1) for x in (wrap(v) if parents[b] is None else wrap(v - ang[parents[b]]))]
                         for b, v in ang.items()}
    d["swaps"] = {g: [c == "1" for c in s] for g, s in d["swaps"].items()}
    return Clip.model_validate(d)


def read_clip(ref: str) -> Clip:
    source, _, mid = ref.partition("/")
    find(ref)
    with zipfile.ZipFile(_path_of(source)) as z:
        return expand(json.loads(z.read(f"clips/{mid}.json")))


def search(words: str = "", category: str | None = None, source: str | None = None
           ) -> list[tuple[CatalogIndex, CatalogMotion]]:
    """Motions whose name/tags/category contain every word (case-insensitive)."""
    terms = words.lower().split()
    out = []
    for idx, m in all_motions():
        if source and idx.source != source or category and m.category != category:
            continue
        hay = " ".join([m.name, m.category, *m.tags, m.id]).lower()
        if all(t in hay for t in terms):
            out.append((idx, m))
    return out


def safe_name(text: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in text.lower())
    return "_".join(p for p in keep.split("_") if p)[:48] or "motion"


def preview(ref: str, cell: int = 120) -> Path:
    """Stick-figure GIF of a catalog motion (made once, then cached)."""
    from ggs.rig import load_rig_spec
    from ggs.rig.preview import render_preview

    source, _, mid = ref.partition("/")
    out = shared_dir() / "cache" / "previews" / source / f"{mid}_{cell}.gif"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        clip = read_clip(ref)
        render_preview(load_rig_spec(repo_root() / "rigs" / f"{clip.rig}.yaml"), clip, out, cell=cell,
                       max_frames=240)
    return out


def add_to(refs: list[str], dest: Path) -> list[Path]:
    """Copy catalog motions into a folder (a project's clips folder) as ``<name>.json`` + ``.motion.json`` (the
    sidecar keeps where it came from, for the export's CREDITS.txt; file names do not)."""
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for ref in refs:
        idx, m = find(ref)
        clip = read_clip(ref)
        base, n = safe_name(m.name), 1
        path = dest / f"{base}.json"
        while path.exists():  # another motion with the same name: number it (the same motion is replaced)
            try:
                same = json.loads(path.with_suffix(".motion.json").read_text(encoding="utf-8")).get("id") == m.id
            except (OSError, ValueError):
                same = False
            if same:
                break
            n += 1
            path = dest / f"{base}_{n}.json"
        clip.save(path)
        path.with_suffix(".motion.json").write_text(json.dumps(
            {"catalog": idx.source, "id": m.id, "name": m.name, "credit": idx.credit, "license": idx.license},
            indent=2), encoding="utf-8")
        out.append(path)
    return out
