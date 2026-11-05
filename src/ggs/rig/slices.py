"""A character's sliced version: its accepted design drawn again in a template's pose and cut into parts.

    <item>/design/          the accepted design (``ggs.items.accept``)
    <item>/sliced/          character.yaml + parts.png (+ validation.png, refine.json) and slices.json
    <item>/sliced/work/     every try (master, cut, report)

Templates (``rigs/slice_templates.yaml``) pick the rig, so the part-sheet layout, and a pose that keeps arms and
legs apart. The drawing model sees a stick figure in that pose plus the design as the character to draw (klein);
models that cannot look at the design (SDXL) draw the design's description with its seed in the pose instead.
The cut is the existing cutter (``ggs.rig.make.cut_master``): a try that fails validation is not used.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

SLICE_PROMPT = (
    "{style}Image 1 is a stick figure, image 2 is a game character: {who}. Draw the character from image 2 - the "
    "same face, hair, clothes, colours and proportions - in exactly the pose of the stick figure in image 1, with "
    "the same size and place in the picture, {view}. Arms and legs apart from the body, every hand and foot "
    "visible. The stick figure is only a pose reference: do not draw its lines or dots. Flat 2D game character "
    "art with clean outlines, plain flat pure white background, no shadow, no text"
)


class SliceTemplate(BaseModel):
    id: str
    label: str
    about: str = ""
    rig: str
    pose: str
    refine: bool = False


class SliceTemplates(BaseModel):
    format: Literal["ggs.slice_templates"]
    version: Literal[1]
    templates: list[SliceTemplate] = Field(default_factory=list)


def load_templates(path: Path | None = None) -> list[SliceTemplate]:
    from ggs.paths import rigs_dir

    path = path or rigs_dir() / "slice_templates.yaml"
    try:
        return SliceTemplates.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))).templates
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc


def get_template(tid: str) -> SliceTemplate:
    for t in load_templates():
        if t.id == tid:
            return t
    raise ValueError(f"unknown template {tid!r}; known: {[t.id for t in load_templates()]}")


def template_view(t: SliceTemplate) -> str:
    return "front" if "front" in t.rig else "side"


def template_preview(t: SliceTemplate, size: int = 320) -> Path:
    """A picture of the template's part sheet (placeholder parts), cached in ~/.ggs/cache/templates."""
    import tempfile

    from PIL import Image

    from ggs.paths import shared_dir
    from ggs.rig.character import make_programmer_art
    from ggs.rig.poses import rig_for

    out = shared_dir() / "cache" / "templates" / f"{t.id}_{size}.png"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    rig = rig_for(template_view(t))
    with tempfile.TemporaryDirectory() as tmp:
        make_programmer_art(rig, tmp, px_per_height=460)
        sheet = Image.open(Path(tmp) / "parts.png").convert("RGBA")
    if not t.refine:  # only the detailed template draws the expression cells
        cell = rig.part_sheet.cell_px
        rows = max(p.cell[1] for p in rig.part_sheet.parts) + 1
        sheet = sheet.crop((0, 0, sheet.width, rows * cell))
    bg = Image.new("RGBA", sheet.size, (255, 255, 255, 255))
    bg.alpha_composite(sheet)
    bg.thumbnail((size, size), Image.LANCZOS)
    bg.convert("RGB").save(out)
    return out


ROUGH_IOU = 0.75  # no try validates: the best one at least this close to the drawing is kept as a rough cut


@dataclass
class SliceResult:
    character: Path | None  # sliced/character.yaml, None when no try could be used
    tries: list[dict]  # per seed: passed, score, folder, failures
    rough: bool = False  # kept although it failed validation (pieces may show gaps)


def sliced_dir(item_dir: Path) -> Path:
    return Path(item_dir) / "sliced"


def sliced_character(item_dir: Path) -> Path | None:
    p = sliced_dir(item_dir) / "character.yaml"
    return p if p.exists() else None


def sliced_info(item_dir: Path) -> dict:
    p = sliced_dir(item_dir) / "slices.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def warp_skin(item_dir: Path) -> Path | None:
    """The item's warp rig (sliced/warp/skin.json), made with the sliced version."""
    p = sliced_dir(item_dir) / "warp" / "skin.json"
    return p if p.exists() else None


def fit_masters(masters: list[Path], rig, model: str, stages) -> bool:
    """Move each master's skeleton onto the joints found in its drawing (ViTPose). Skipped for the stub model,
    for rigs other than the side view, and when the joint finder is not downloaded (the guide's joints are
    used then). Returns whether the joints were fitted."""
    from PIL import Image

    from ggs.pipeline.models import MissingModelError
    from ggs.rig.joints import detect_keypoints, fit_master
    from ggs.rig.master import load_master

    if model == "stub" or rig.view != "side" or not masters:
        return False
    try:
        kps = detect_keypoints([Image.fromarray(load_master(m)[1], "RGBA") for m in masters], stages=stages)
    except (MissingModelError, OSError) as exc:
        print(f"joint finder not used ({exc}); cutting along the guide")
        return False
    for m, kp in zip(masters, kps):
        fit_master(m, kp, rig)
    return True


def make_warp(master_json: Path, rig, out_dir: Path) -> Path:
    """Build and save the warp rig of a (fitted) master."""
    from PIL import Image

    from ggs.rig.joints import bone_widths
    from ggs.rig.master import load_master
    from ggs.rig.skin import build_skin

    spec, master, labels = load_master(master_json)
    guide_labels = labels
    guide_copy = Path(master_json).parent / "guide_labels_guide.png"
    if guide_copy.exists():  # widths come from the guide's own shapes
        import numpy as np

        guide_labels = np.asarray(Image.open(guide_copy).convert("L")).astype(np.int16) - 1
    widths = spec.fit.get("widths") or bone_widths(guide_labels, rig, spec.joints)
    image = Image.fromarray(master, "RGBA")
    skin = build_skin(rig, image, spec.joints, widths, spec.height_px)
    k = skin.height_px / spec.height_px
    if k < 1.0:
        image = image.resize((round(image.width * k), round(image.height * k)), Image.LANCZOS)
    return skin.save(out_dir, image)


def make_slices(project: Path, ref: str, template: str, *, seeds: list[int] | None = None, profile: str = "lite",
                enforce_vram: bool = True) -> SliceResult:
    """Draw the item's accepted design in the template's pose, cut it, keep the first try that validates."""
    from PIL import Image

    from ggs.generate import build_backends, real_rgba
    from ggs.items import CharacterSettings, ItemError, design_dir, load
    from ggs.pipeline import progress
    from ggs.pipeline.vram import Stages
    from ggs.recipes import load_recipe
    from ggs.rig.body import PRESETS, BodyShape
    from ggs.rig.make import CHARACTER_TEXT, cut_master, guide_puppet, on_white, pose_reference
    from ggs.rig.master import render_guide, save_master
    from ggs.rig.poses import pose_angles, rig_for

    project = Path(project)
    rec, d = load(project, ref)
    info_file = design_dir(d) / "design.json"
    if rec.accepted is None or not info_file.exists():
        raise ItemError(f"{rec.name!r} has no design yet: press 'Use this design' first")
    t = get_template(template)
    info = json.loads(info_file.read_text(encoding="utf-8"))
    drawing = info["drawing"]
    cs = CharacterSettings(**(drawing.get("character") or {}))
    body = BodyShape(**cs.body) if cs.body else PRESETS.get(cs.body_preset, PRESETS["standard"])
    model = drawing.get("model") or "flux2-klein-4b"
    view = template_view(t)
    rig = rig_for(view)
    puppet, lengths = guide_puppet(rig, body)
    guide = render_guide(rig, puppet, lengths, size=1024, angles=pose_angles(view, t.pose))
    who = "; ".join(info.get("text") or [rec.description])
    style, lora, weight = "", None, 1.0
    if drawing.get("style"):
        from ggs.style.package import resolve_style

        meta, lora = resolve_style(drawing["style"], project)
        style, weight = f"{meta.trigger}, ", drawing.get("style_weight") or meta.recommended_weight
    seeds = seeds or [drawing["seed"] + i for i in range(3)]
    view_words = "side view facing right" if view == "side" else "front view facing the viewer, symmetrical T-pose"
    progress.start(f"slices:{model}", ["encode", "denoise", "matte", "joints", "cut"] +
                   (["denoise", "matte"] if t.refine else []))
    recipe = load_recipe("sprite")
    gen, matter = build_backends(recipe, model, profile)
    stages = Stages(profile=profile, enforce=enforce_vram)
    if getattr(gen, "supports_reference", False):
        design = on_white(Image.open(design_dir(d) / "design.png"))
        design.thumbnail((768, 768), Image.LANCZOS)
        prompt = " ".join(SLICE_PROMPT.format(style=style, who=who, view=view_words).split())
        pose = pose_reference(rig, guide, "stub" if model == "stub" else "reference")
        images = gen.render([(prompt, sd, [pose, design]) for sd in seeds], 1024, 1024,
                            stages, lora=lora, lora_scale=weight)
    elif getattr(gen, "supports_pose", False):
        prompt = " ".join(CHARACTER_TEXT.format(style=style, description=who, body="", view=view_words).split())
        images = gen.generate(prompt, seeds, 1024, 1024, stages, lora=lora, lora_scale=weight,
                              pose=pose_reference(rig, guide, "openpose"))
    else:
        raise ValueError(f"{model!r} cannot draw a character in a chosen pose")
    alphas = matter.alphas(images, stages)
    can_refine = getattr(gen, "supports_reference", False)
    del gen
    out = sliced_dir(d)
    work = out / "work"
    if work.exists():
        shutil.rmtree(work)
    tries, best = [], None
    spec0 = guide.spec.model_copy(update={"body": body.model_dump(), "prompt": prompt, "model": model})
    masters = []
    for sd, img, alpha in zip(seeds, images, alphas):
        rgba, _, _ = real_rgba(img, alpha, recipe.post.alpha_low, recipe.post.alpha_high)
        folder = work / f"seed_{sd:06d}"
        masters.append((sd, folder, save_master(folder, spec0.model_copy(update={"seed": sd}),
                                                Image.fromarray(rgba, "RGBA"), guide)))
        img.save(folder / "master_raw.png")
    fitted = fit_masters([m for _, _, m in masters], rig, model, stages)
    with stages.stage("cut"):  # CPU: cut every try into pieces and check it (the longest part after drawing)
        for sd, folder, master in masters:
            outcome = cut_master(master, rig, folder / "character", rec.name)
            fails = [f"{c.name}: {c.detail}" for c in outcome.report.checks if not c.passed]
            tries.append({"seed": sd, "passed": outcome.report.passed, "score": round(outcome.report.score, 4),
                          "folder": str(folder), "failures": fails[:6], "cut": outcome.character is not None})
            if outcome.report.passed and best is None:
                best = folder / "character"
    rough = False
    if best is None:
        usable = [t for t in tries if t["cut"] and t["score"] >= ROUGH_IOU]
        if usable:
            best, rough = Path(max(usable, key=lambda t: t["score"])["folder"]) / "character", True
    if best is not None and t.refine and can_refine:
        from ggs.rig.refine import refine_characters

        refine_characters([best], rig, model=model, profile=profile, enforce_vram=enforce_vram)
    for f in out.glob("*"):  # replace the previous sliced version (the work folder stays)
        if f.name != "work":
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    if best is not None:
        for f in best.iterdir():
            if f.is_file():
                shutil.copy2(f, out / f.name)
    # the warp rig (bend the whole drawing): from the try that was cut best, else the first one
    warp = None
    if rig.view == "side" and tries:
        pick = best.parent if best is not None else Path(max(tries, key=lambda t: t["score"])["folder"])
        warp = make_warp(pick / "master.json", rig, out / "warp")
    (out / "slices.json").write_text(json.dumps({
        "format": "ggs.slices", "version": 1, "template": t.id, "rig": rig.id, "design": rec.accepted,
        "passed": best is not None and not rough, "rough": rough, "tries": tries, "joints_fitted": fitted,
        "warp": "warp/skin.json" if warp else None}, indent=1), encoding="utf-8")
    return SliceResult(out / "character.yaml" if best is not None else None, tries, rough)
