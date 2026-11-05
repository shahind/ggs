"""Character pipeline glue: pose guide -> generated master (klein edit + matting) -> cut -> validate.

Every step writes its files, so each can be re-run on its own (`ggs character master | cut`).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ggs.rig.spec import RigSpec

MASTER_PROMPT = (
    "{style}Redraw the figure in the image as {description}{body}. Keep exactly the same pose, body proportions, size "
    "and position. Full body, side view facing right, flat 2D game character art with clean outlines, plain flat "
    "pure white background, no shadow, no text"
)


VIEW_WORDS = {"side": "side view facing right", "front": "front view facing the viewer, symmetrical stance"}
CHARACTER_PROMPT = (
    "{style}Redraw the figure in the image as {description}{body}. Keep exactly the same pose, body proportions, "
    "size and position. Full body, {view}, flat 2D game character art with clean outlines, plain flat pure white "
    "background, no shadow, no text"
)
CHARACTER_EDIT = ("{instruction}. Keep everything else exactly the same: the same character, pose, proportions, "
                  "position and drawing style, plain flat pure white background.")


def draw_character(guide, description: str, *, view: str, body=None, seeds: list[int], model: str,
                   profile: str = "lite", lora=None, lora_scale: float = 1.0, style: str = "",
                   edit_of=None, instruction: str = "", enforce_vram: bool = True) -> list[tuple]:
    """Draw a character over a pose guide (or edit a previous drawing ``edit_of`` with ``instruction``).
    Returns [(rgba array on the guide canvas, raw RGB image, seed)] per seed."""
    from ggs.generate import build_backends, real_rgba
    from ggs.pipeline.vram import Stages
    from ggs.recipes import load_recipe

    recipe = load_recipe("sprite")
    gen, matter = build_backends(recipe, model, profile)
    if not getattr(gen, "supports_reference", False):
        raise ValueError(f"{model!r} cannot draw over a pose guide; characters need flux2-klein-4b")
    size = guide.image.size[0]
    if edit_of is not None:
        prompt = CHARACTER_EDIT.format(instruction=instruction.strip().rstrip("."))
        ref = edit_of
    else:
        words = body.describe() if body is not None else ""
        prompt = CHARACTER_PROMPT.format(style=style, description=description.strip(),
                                         body=f", with {words}" if words else "", view=VIEW_WORDS[view])
        ref = on_white(guide.image)
    prompt = " ".join(prompt.split())
    stages = Stages(profile=profile, enforce=enforce_vram)
    images = gen.generate(prompt, seeds, size, size, stages, lora=lora, lora_scale=lora_scale, reference=[ref])
    alphas = matter.alphas(images, stages)
    out = []
    for seed, img, alpha in zip(seeds, images, alphas):
        rgba, _, _ = real_rgba(img, alpha, recipe.post.alpha_low, recipe.post.alpha_high)
        out.append((rgba, img, seed))
    return out


def guide_puppet(rig: RigSpec, body=None):
    """The programmer-art puppet (drawn on the fly, in ``body``'s proportions) and its bone lengths."""
    from ggs.rig.character import load_character, make_programmer_art

    with tempfile.TemporaryDirectory() as tmp:
        # drawn small enough that any body shape fits the cells; the guide render scales it up
        ch = load_character(make_programmer_art(rig, tmp, px_per_height=460, body=body))
        return ch.puppet(rig), dict(ch.spec.bone_lengths)


def make_guide(rig: RigSpec, size: int = 1024, body=None, pose: str = "walk"):
    from ggs.rig.master import MASTER_POSES, render_guide

    if pose not in MASTER_POSES:
        raise ValueError(f"unknown master pose {pose!r}; known: {sorted(MASTER_POSES)}")
    puppet, lengths = guide_puppet(rig, body)
    g = render_guide(rig, puppet, lengths, size=size, angles=MASTER_POSES[pose])
    if body is not None:
        g.spec = g.spec.model_copy(update={"body": body.model_dump()})
    return g


def on_white(img):
    from PIL import Image

    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img.convert("RGBA"))
    return bg.convert("RGB")


def make_masters(rig: RigSpec, description: str, out_dir: str | Path, *, seeds: list[int], model: str,
                 profile: str = "lite", style_ref: str | None = None, style_weight: float | None = None,
                 project: Path | None = None, enforce_vram: bool = True, size: int = 1024,
                 body=None, pose: str = "walk") -> list[Path]:
    """Generate one master per seed from the pose guide (in ``body``'s proportions). Returns the master.json
    paths (out/seed_NNN/)."""
    from PIL import Image

    from ggs.generate import build_backends, real_rgba
    from ggs.pipeline.vram import Stages
    from ggs.recipes import load_recipe
    from ggs.rig.master import save_master

    recipe = load_recipe("sprite")  # matting model and edge settings
    guide = make_guide(rig, size, body, pose)
    style, lora = "", None
    if style_ref:
        from ggs.style.package import StyleError, resolve_style

        meta, lora = resolve_style(style_ref, project)
        if model != "stub" and meta.base_model != model:
            raise StyleError(f"style {meta.name!r} was trained on {meta.base_model!r}; it cannot load on {model!r}")
        style = f"{meta.trigger}, "
        style_weight = meta.recommended_weight if style_weight is None else style_weight
    gen, matter = build_backends(recipe, model, profile)
    if not getattr(gen, "supports_reference", False):
        raise ValueError(f"{model!r} cannot draw over a pose guide; character masters need flux2-klein-4b")
    words = body.describe() if body is not None else ""
    prompt = " ".join(MASTER_PROMPT.format(style=style, description=description.strip(),
                                           body=f", with {words}" if words else "").split())
    stages = Stages(profile=profile, enforce=enforce_vram)
    images = gen.generate(prompt, seeds, size, size, stages, lora=lora,
                          lora_scale=1.0 if style_weight is None else style_weight, reference=[on_white(guide.image)])
    alphas = matter.alphas(images, stages)
    out = Path(out_dir)
    paths = []
    for seed, img, alpha in zip(seeds, images, alphas):
        rgba, _, _ = real_rgba(img, alpha, recipe.post.alpha_low, recipe.post.alpha_high)
        spec = guide.spec.model_copy(update={"prompt": prompt, "seed": seed, "model": model})
        folder = out / f"seed_{seed:03d}"
        paths.append(save_master(folder, spec, Image.fromarray(rgba, "RGBA"), guide))
        img.save(folder / "master_raw.png")
    (out / "stages.json").write_text(json.dumps(stages.summary(), indent=1), encoding="utf-8")
    return paths


def guide_only_master(rig: RigSpec, out_dir: str | Path, size: int = 1024, body=None, pose: str = "walk") -> Path:
    """The guide itself as the master (no GPU): a reference run of the cutter and validator."""
    from ggs.rig.master import save_master

    guide = make_guide(rig, size, body, pose)
    return save_master(out_dir, guide.spec.model_copy(update={"model": "guide"}), guide.image, guide)


@dataclass
class CutOutcome:
    character: Path | None  # None when the master could not be cut
    report: object  # ggs.rig.validate.Report
    scale: float
    px_per_height: float


def cut_master(master_json: str | Path, rig: RigSpec, out_dir: str | Path, name: str,
               stature: float = 1.0) -> CutOutcome:
    """master.json -> part sheet + character.yaml + validation.json + validation.png in ``out_dir``."""
    from PIL import Image

    from ggs.rig.character import load_character
    from ggs.rig.cutter import CutError, cut_parts, normalize, write_character
    from ggs.rig.master import load_master
    from ggs.rig.validate import debug_sheet, validate_character

    spec, master, labels = load_master(master_json)
    if spec.rig != rig.id:
        raise ValueError(f"{master_json} was made for rig {spec.rig!r}, not {rig.id!r}")
    out = Path(out_dir)
    try:
        cut = cut_parts(master, labels, spec, rig)
        sheet = normalize(cut, spec, rig)
    except CutError as exc:  # the master is too far off to cut at all: a rejection, not a crash
        from ggs.rig.validate import Check, Report

        report = Report(False, 0.0, [Check("cut", False, 0.0, 0.0, str(exc))])
        out.mkdir(parents=True, exist_ok=True)
        (out / "validation.json").write_text(json.dumps({**report.to_dict(), "master": str(Path(master_json)
                                                         .resolve())}, indent=1), encoding="utf-8")
        return CutOutcome(None, report, 0.0, 0.0)
    path = write_character(sheet, rig, out_dir, name, stature=stature)
    report = validate_character(load_character(path), rig, master, spec, cut, labels)
    info = {**report.to_dict(), "master": str(Path(master_json).resolve()), "scale": round(sheet.scale, 4),
            "px_per_height": round(sheet.px_per_height, 2), "parts": sheet.placements}
    (out / "validation.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    debug_sheet(Image.fromarray(master, "RGBA"), report).save(out / "validation.png")
    return CutOutcome(path, report, sheet.scale, sheet.px_per_height)
