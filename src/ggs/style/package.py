"""Portable style package (``.ggsstyle``, a zip) and project folders.

Package contents:
    style.yaml      metadata: name, trigger word, base model, training settings, dataset size
    lora.safetensors LoRA weights (diffusers format)
    preview.png     samples generated right after training
    captions.txt    the training captions, for reference

A project is any folder with ``.ggs/project.yaml``. Installed styles live in ``.ggs/styles/<name>/`` (unzipped),
so a project carries its own copy. ``~/.ggs/styles/`` holds styles shared by every project.
"""

from __future__ import annotations

import datetime as dt
import re
import zipfile
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from ggs import __version__
from ggs.paths import shared_dir

PACKAGE_SUFFIX = ".ggsstyle"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")


class StyleMeta(BaseModel):
    format: str = "ggs.style"
    version: int = 1
    name: str
    trigger: str
    base_model: str  # pipeline/registry id the LoRA was trained on; it only loads on this base
    trained_on: str = ""  # transformer file the LoRA was trained against
    rank: int
    alpha: float
    steps: int
    resolution: int
    learning_rate: float
    images: int
    recommended_weight: float = 1.0
    rights_confirmed: bool = True
    created: str = Field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    generator: str = f"ggs {__version__}"

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _NAME.match(v):
            raise ValueError("style name: lowercase letters, digits, '-' or '_' (max 48 chars)")
        return v


class StyleError(ValueError):
    pass


def write_package(out: str | Path, meta: StyleMeta, lora: Path, preview=None, captions: list[str] | None = None) -> Path:
    out = Path(out)
    if out.suffix != PACKAGE_SUFFIX:
        out = out.with_suffix(PACKAGE_SUFFIX)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("style.yaml", yaml.safe_dump(meta.model_dump(), sort_keys=False))
        z.write(lora, "lora.safetensors")
        if preview is not None:
            import io

            buf = io.BytesIO()
            preview.save(buf, format="PNG")
            z.writestr("preview.png", buf.getvalue())
        if captions:
            z.writestr("captions.txt", "\n".join(captions) + "\n")
    return out


def read_meta(package: str | Path) -> StyleMeta:
    package = Path(package)
    try:
        with zipfile.ZipFile(package) as z:
            data = yaml.safe_load(z.read("style.yaml"))
            if "lora.safetensors" not in z.namelist():
                raise StyleError(f"{package}: no lora.safetensors inside")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise StyleError(f"{package}: not a GGS style package") from exc
    return StyleMeta.model_validate(data)


def install(package: str | Path, styles_dir: Path, overwrite: bool = False) -> Path:
    """Unzip a package into ``styles_dir/<name>/``. Returns that folder."""
    meta = read_meta(package)
    dest = styles_dir / meta.name
    if dest.exists() and not overwrite:
        raise StyleError(f"style {meta.name!r} is already installed in {styles_dir} (use --overwrite)")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as z:
        for member in ("style.yaml", "lora.safetensors", "preview.png", "captions.txt"):
            if member in z.namelist():
                (dest / member).write_bytes(z.read(member))
    return dest


# ---------------------------------------------------------------- projects


class ProjectMeta(BaseModel):
    format: str = "ggs.project"
    version: int = 1
    name: str
    created: str = Field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    base_model: str | None = None  # pipeline id used for generation and style training; None = recipe default


def project_dir(path: str | Path) -> Path:
    return Path(path) / ".ggs"


def load_project(path: str | Path) -> ProjectMeta:
    meta_path = project_dir(require_project(path)) / "project.yaml"
    try:
        return ProjectMeta.model_validate(yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {})
    except Exception as exc:
        raise StyleError(f"{meta_path}: {exc}") from exc


def init_project(path: str | Path, name: str | None = None, base_model: str | None = None) -> Path:
    root = Path(path)
    meta_path = project_dir(root) / "project.yaml"
    if meta_path.exists():
        raise StyleError(f"{root} is already a GGS project")
    (project_dir(root) / "styles").mkdir(parents=True, exist_ok=True)
    meta = ProjectMeta(name=name or root.resolve().name, base_model=base_model)
    meta_path.write_text(yaml.safe_dump(meta.model_dump(), sort_keys=False), encoding="utf-8")
    return meta_path


def require_project(path: str | Path) -> Path:
    root = Path(path)
    if not (project_dir(root) / "project.yaml").exists():
        raise StyleError(f"{root} is not a GGS project; run `ggs project init {root}`")
    return root


def styles_dirs(project: str | Path | None) -> list[Path]:
    dirs = []
    if project is not None:
        dirs.append(project_dir(require_project(project)) / "styles")
    dirs.append(shared_dir() / "styles")
    return dirs


def list_styles(project: str | Path | None) -> list[tuple[StyleMeta, Path]]:
    out = []
    for d in styles_dirs(project):
        for meta_file in sorted(d.glob("*/style.yaml")):
            out.append((StyleMeta.model_validate(yaml.safe_load(meta_file.read_text(encoding="utf-8"))),
                        meta_file.parent))
    return out


def resolve_style(ref: str, project: str | Path | None) -> tuple[StyleMeta, Path]:
    """``ref`` = an installed style name (project first, then ~/.ggs/styles) or a path to a .ggsstyle file
    (unpacked into ~/.ggs/cache/styles). Returns (meta, lora path)."""
    p = Path(ref)
    if p.suffix == PACKAGE_SUFFIX:
        if not p.exists():
            raise StyleError(f"{p}: no such style package")
        meta = read_meta(p)
        dest = install(p, shared_dir() / "cache" / "styles", overwrite=True)
        return meta, dest / "lora.safetensors"
    for d in styles_dirs(project):
        folder = d / ref
        if (folder / "style.yaml").exists():
            meta = StyleMeta.model_validate(yaml.safe_load((folder / "style.yaml").read_text(encoding="utf-8")))
            return meta, folder / "lora.safetensors"
    where = " or ".join(str(d) for d in styles_dirs(project))
    raise StyleError(f"style {ref!r} not found in {where}; install it with `ggs style install`")
