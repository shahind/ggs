"""Rig-sheet adapter (M6): a klein edit LoRA that turns a master image into the rig's part sheet.

Folder layout (``~/.ggs/adapters/<name>/`` or any folder): adapter.yaml, lora.safetensors, preview.png.
The adapter draws the sheet in the factory layout: the rig's default pivots, the programmer-art bone lengths
and ``SHEET_PX_PER_HEIGHT`` pixels per standing height, which are also the guide's proportions, so the
character it makes needs no measuring.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field

from ggs.rig.spec import RigSpec

TRIGGER = "GGSRIGSHEET"
ADAPTER_PROMPT = f"{TRIGGER}, split the character in the image into its body-part sheet, one part per cell"


class AdapterMeta(BaseModel):
    format: Literal["ggs.adapter"] = "ggs.adapter"
    version: Literal[1] = 1
    kind: Literal["rig_sheet"] = "rig_sheet"
    name: str
    rig: str
    base_model: str
    prompt: str = ADAPTER_PROMPT
    ref_size: tuple[int, int]
    target_size: tuple[int, int]
    sheet_px_per_height: float
    pairs: int
    steps: int
    rank: int
    learning_rate: float
    seconds: float = 0.0
    holdout: list[str] = Field(default_factory=list)  # pair ids kept out of training (for `ggs adapter eval`)
    created: str = Field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))


def write_adapter(out_dir: str | Path, meta: AdapterMeta, lora: Path, preview=None) -> Path:
    import shutil

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(lora, out / "lora.safetensors")
    if preview is not None:
        preview.save(out / "preview.png")
    (out / "adapter.yaml").write_text(yaml.safe_dump(meta.model_dump(), sort_keys=False), encoding="utf-8")
    return out


def load_adapter(folder: str | Path) -> tuple[AdapterMeta, Path]:
    folder = Path(folder)
    path = folder / "adapter.yaml"
    try:
        meta = AdapterMeta.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise ValueError(f"{folder} is not an adapter folder (no adapter.yaml)") from exc
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not (folder / "lora.safetensors").exists():
        raise ValueError(f"{folder}: lora.safetensors is missing")
    return meta, folder / "lora.safetensors"


def write_sheet_character(sheet_rgba: np.ndarray, rig: RigSpec, out_dir: str | Path, name: str) -> Path:
    """A generated sheet in the factory layout -> character folder (parts.png + character.yaml)."""
    from PIL import Image

    from ggs.rig.character import CharacterSpec
    from ggs.rig.factory import SHEET_PX_PER_HEIGHT, sheet_guide

    ps = rig.part_sheet
    size = (ps.cols * ps.cell_px, ps.rows * ps.cell_px)
    img = Image.fromarray(sheet_rgba, "RGBA")
    if img.size != size:
        img = img.convert("RGBa").resize(size, Image.LANCZOS).convert("RGBA")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img.save(out / "parts.png")
    base = yaml.safe_load(sheet_guide(rig).character_yaml)
    spec = CharacterSpec(name=name, rig=rig.id, sheet="parts.png", px_per_height=SHEET_PX_PER_HEIGHT,
                         bone_lengths=base["bone_lengths"])
    path = out / "character.yaml"
    path.write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False), encoding="utf-8")
    return path


def sheets_from_masters(masters: list, adapter: str | Path, *, profile: str = "lite", enforce_vram: bool = True,
                        size: tuple[int, int] | None = None, seed: int = 0, lora_scale: float = 1.0) -> list:
    """Masters (RGBA arrays) -> sheet RGBA arrays via the adapter (one model load for all)."""
    from PIL import Image

    from ggs.generate import build_backends, real_rgba
    from ggs.pipeline.vram import Stages
    from ggs.recipes import load_recipe
    from ggs.rig.make import on_white

    meta, lora = load_adapter(adapter)
    recipe = load_recipe("sprite")
    gen, matter = build_backends(recipe, meta.base_model, profile)
    w, h = size or tuple(meta.target_size)
    rw, rh = meta.ref_size
    refs = [on_white(Image.fromarray(m, "RGBA")).resize((rw, rh), Image.LANCZOS) for m in masters]
    stages = Stages(profile=profile, enforce=enforce_vram)
    images = gen.render([(meta.prompt, seed, [r]) for r in refs], w, h, stages, lora=lora, lora_scale=lora_scale)
    alphas = matter.alphas(images, stages)
    return [real_rgba(im, a, recipe.post.alpha_low, recipe.post.alpha_high)[0] for im, a in zip(images, alphas)]


def adapter_character(master_json: str | Path, sheet_rgba: np.ndarray, rig: RigSpec, out_dir: str | Path,
                      name: str):
    """Write the adapter's sheet as a character next to the master and validate it like a cut (CutOutcome)."""
    from PIL import Image

    from ggs.rig.character import load_character
    from ggs.rig.cutter import cut_parts
    from ggs.rig.factory import SHEET_PX_PER_HEIGHT
    from ggs.rig.make import CutOutcome
    from ggs.rig.master import load_master
    from ggs.rig.validate import debug_sheet, validate_character

    spec, master, labels = load_master(master_json)
    path = write_sheet_character(sheet_rgba, rig, out_dir, name)
    cut = cut_parts(master, labels, spec, rig)  # labels and joint widths for the validator's checks
    report = validate_character(load_character(path), rig, master, spec, cut, labels)
    out = Path(out_dir)
    (out / "validation.json").write_text(json.dumps({**report.to_dict(), "master": str(Path(master_json).resolve()),
                                                     "source": "adapter"}, indent=1), encoding="utf-8")
    debug_sheet(Image.fromarray(master, "RGBA"), report).save(out / "validation.png")
    return CutOutcome(path, report, 1.0, SHEET_PX_PER_HEIGHT)


# ---------------------------------------------------------------- evaluation against factory ground truth

def pose_fidelity(candidate, truth, rig: RigSpec, height_px: float = 400.0, size: int = 800) -> dict:
    """Render both characters in every test pose: silhouette IoU and colour error against the ground truth."""
    from ggs.rig.master import MASTER_POSE, TEST_POSES, pose_clip

    ious, errs = [], []
    for overrides in TEST_POSES.values():
        angles = {**MASTER_POSE, **overrides}
        ims = []
        for ch in (candidate, truth):
            clip = pose_clip(rig, angles, ch.spec.bone_lengths)
            ims.append(np.asarray(ch.puppet(rig).render_frame(clip, 0, height_px, (size, size),
                                                              (size / 2, size * 0.55))))
        a, b = (im[..., 3] >= 128 for im in ims)
        union = (a | b).sum()
        ious.append(float((a & b).sum() / union) if union else 0.0)
        both = a & b
        errs.append(float(np.abs(ims[0][..., :3].astype(np.float32) - ims[1][..., :3])[both].mean() / 255)
                    if both.any() else 1.0)
    return {"pose_iou": round(float(np.mean(ious)), 4), "pose_colour_error": round(float(np.mean(errs)), 4)}
