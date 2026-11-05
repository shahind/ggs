"""Style training datasets: import user images (rights confirmation required), normalise them, keep captions.

Layout of a dataset folder:
    dataset.yaml        manifest (source, rights confirmation, trigger word, images)
    images/NNN.png      normalised training images (RGB, square, ``size`` px)
    images/NNN.txt      one caption per image (written by ``ggs style caption`` or by hand)
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RIGHTS_WARNING = (
    "Only train on images you own or have permission to use. You are responsible for the rights to your "
    "training images; GGS does not check them (see CLAUDE.md, copyright stance)."
)
MIN_IMAGES, RECOMMENDED = 5, (20, 40)


class RightsNotConfirmed(PermissionError):
    pass


class DatasetImage(BaseModel):
    file: str  # relative to images/
    source: str
    sha256: str


class DatasetManifest(BaseModel):
    format: str = "ggs.style_dataset"
    version: int = 1
    source: str
    rights_confirmed: bool
    rights_confirmed_at: str
    size: int
    trigger: str | None = None
    images: list[DatasetImage] = Field(default_factory=list)


def _normalise(path: Path, size: int):
    """RGB, transparency flattened onto white, padded to square with the border colour, resized."""
    import numpy as np
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    im = im.convert("RGB")
    a = np.asarray(im)
    ring = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in np.median(ring, axis=0))
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def import_dataset(src: str | Path, out: str | Path, *, rights_confirmed: bool, size: int = 1024) -> DatasetManifest:
    if not rights_confirmed:
        raise RightsNotConfirmed(RIGHTS_WARNING + " Confirm with --i-have-the-rights to continue.")
    src, out = Path(src), Path(out)
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if len(files) < MIN_IMAGES:
        raise ValueError(f"{src}: found {len(files)} images; need at least {MIN_IMAGES} (20-40 recommended)")
    if size % 16 or not 256 <= size <= 2048:
        raise ValueError("size must be a multiple of 16 between 256 and 2048")
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest(source=str(src.resolve()), rights_confirmed=True,
                               rights_confirmed_at=dt.datetime.now().isoformat(timespec="seconds"), size=size)
    seen: set[str] = set()
    for p in files:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen:
            continue  # exact duplicate
        seen.add(digest)
        name = f"{len(manifest.images) + 1:03d}.png"
        _normalise(p, size).save(img_dir / name)
        cap = p.with_suffix(".txt")
        if cap.exists():  # keep a caption that came with the image
            (img_dir / name).with_suffix(".txt").write_text(cap.read_text(encoding="utf-8").strip() + "\n",
                                                           encoding="utf-8")
        manifest.images.append(DatasetImage(file=name, source=str(p.relative_to(src)), sha256=digest))
    save_manifest(out, manifest)
    return manifest


def save_manifest(folder: str | Path, manifest: DatasetManifest) -> Path:
    path = Path(folder) / "dataset.yaml"
    path.write_text(yaml.safe_dump(manifest.model_dump(), sort_keys=False), encoding="utf-8")
    return path


def load_dataset(folder: str | Path) -> tuple[DatasetManifest, list[tuple[Path, str | None]]]:
    """Manifest + (image path, caption or None) per image."""
    folder = Path(folder)
    path = folder / "dataset.yaml"
    try:
        manifest = DatasetManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise ValueError(f"{folder}: not a style dataset (no dataset.yaml); use `ggs style dataset import`") from exc
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not manifest.rights_confirmed:
        raise RightsNotConfirmed(f"{path}: rights were not confirmed")
    items = []
    for im in manifest.images:
        p = folder / "images" / im.file
        cap = p.with_suffix(".txt")
        items.append((p, cap.read_text(encoding="utf-8").strip() if cap.exists() else None))
    return manifest, items
