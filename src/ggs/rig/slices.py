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


LAYERS = "see-through-layers"  # pose-section model that splits a character into complete layers


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
    images = [Image.fromarray(load_master(m)[1], "RGBA") for m in masters]
    try:
        kps = detect_keypoints(images, stages=stages)
    except (MissingModelError, OSError) as exc:
        print(f"joint finder not used ({exc}); cutting along the guide")
        return False
    try:
        from ggs.rig.parsing import parse_bodies

        parses = parse_bodies(images, stages=stages)
    except (MissingModelError, OSError) as exc:
        print(f"body parser not used ({exc})")
        parses = [None] * len(masters)
    for m, kp, seg in zip(masters, kps, parses, strict=True):
        fit_master(m, kp, rig, seg)
    return True


def sliced_stale(item_dir: Path) -> bool:
    """True when the sliced version was cut from an earlier design than the accepted one."""
    from ggs.items import ItemRecord

    p = sliced_dir(item_dir) / "slices.json"
    if not p.exists() or not (Path(item_dir) / "item.json").exists():
        return False
    made_from = json.loads(p.read_text(encoding="utf-8")).get("design")
    rec = ItemRecord.model_validate_json((Path(item_dir) / "item.json").read_text(encoding="utf-8"))
    return made_from is not None and rec.accepted is not None and made_from != rec.accepted


def rigdraw_dir(item_dir: Path) -> Path:
    """The animation-ready drawing: the design drawn by GGS in its layout's capture pose (``master.json`` +
    ``master.png`` + labels), shared by the sliced version and "Bend the drawing"."""
    return Path(item_dir) / "rigdraw"


def resolve_layout(layout, item_dir: Path, base, cs) -> tuple:
    """(Layout, measures) for 'auto' (suggested from the design itself), a preset name or a Layout."""
    from ggs.rig.layout import PRESETS, Layout, suggest

    if isinstance(layout, Layout):
        return layout, {}
    if isinstance(layout, dict):
        return Layout(**layout), {}
    if layout in PRESETS:
        return PRESETS[layout].model_copy(deep=True), {}
    if layout != "auto":
        raise ValueError(f"unknown layout {layout!r}; use auto or one of {sorted(PRESETS)}")
    import numpy as np
    from PIL import Image

    from ggs.items import design_dir
    from ggs.rig.skin import drawing_labels
    from ggs.rig.warp import body_widths, design_skeleton

    data, joints, height_px = design_skeleton(item_dir)
    widths = body_widths(base, cs, joints, height_px)
    arr = np.asarray(Image.open(design_dir(item_dir) / data["image"]).convert("RGBA"))
    labels = drawing_labels(arr, base, joints, widths, height_px)
    return suggest(labels, base, joints, widths, height_px, cs.body_preset)


def layout_master(master_json: Path, base, layout) -> object:
    """Apply ``layout`` to a fitted try: its joints, widths, labels and drawing order become the layout's. The
    fitted plain-rig version is kept (``master_v1.json``, ``labels_v1.png``) so another layout can be applied
    without drawing again. Returns the layout rig."""
    import numpy as np
    from PIL import Image

    from ggs.rig.layout import apply_layout
    from ggs.rig.master import bone_angle

    path = Path(master_json)
    v1, l1 = path.parent / "master_v1.json", path.parent / "labels_v1.png"
    if not v1.exists():
        shutil.copy2(path, v1)
        shutil.copy2(path.parent / "guide_labels.png", l1)
    data = json.loads(v1.read_text(encoding="utf-8"))
    labels = np.asarray(Image.open(l1).convert("L")).astype(np.int16) - 1
    joints = {b: (tuple(s), tuple(e)) for b, (s, e) in data["joints"].items()}
    widths = (data.get("fit") or {}).get("widths")
    if not widths:
        from ggs.rig.joints import bone_widths

        g = path.parent / "guide_labels_guide.png"
        glabels = np.asarray(Image.open(g if g.exists() else l1).convert("L")).astype(np.int16) - 1
        widths = bone_widths(glabels, base, joints)
    rig, j, w, lab, lay = apply_layout(labels, base, layout, joints, widths, float(data["height_px"]))
    Image.fromarray((lab + 1).astype(np.uint8), "L").save(path.parent / data.get("labels", "guide_labels.png"))
    fit = dict(data.get("fit") or {})
    fit.update({"layout": lay.model_dump(), "widths": w, "draw_order": None})
    data.update({"joints": {b: [list(s), list(e)] for b, (s, e) in j.items()},
                 "angles": {b: bone_angle(*j[b]) for b in j}, "fit": fit})
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return rig


def make_slices(project: Path, ref: str, template: str = "side_simple", *, layout="auto",
                seeds: list[int] | None = None, profile: str = "lite", enforce_vram: bool = True) -> SliceResult:
    """Draw the item's accepted design in its layout's capture pose (side view) or the template's pose (front),
    find the joints, cut every try with the layout's rig, keep the best. The best try is also kept as the
    animation-ready drawing (``rigdraw/``)."""
    from PIL import Image

    from ggs.generate import build_backends, real_rgba
    from ggs.items import CharacterSettings, ItemError, design_dir, load
    from ggs.pipeline import progress
    from ggs.pipeline.vram import Stages
    from ggs.recipes import load_recipe
    from ggs.rig.body import PRESETS, BodyShape
    from ggs.rig.layout import capture_pose
    from ggs.rig.make import CHARACTER_TEXT, guide_puppet, on_white, pose_reference
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
    side = view == "side"
    lay, measures = resolve_layout(layout, d, rig, cs) if side else (None, {})
    puppet, lengths = guide_puppet(rig, body)
    angles = capture_pose(lay) if side else pose_angles(view, t.pose)
    guide = render_guide(rig, puppet, lengths, size=1024, angles=angles)
    who = "; ".join(info.get("text") or [rec.description])
    style, lora, weight = "", None, 1.0
    if drawing.get("style"):
        from ggs.style.package import resolve_style

        meta, lora = resolve_style(drawing["style"], project)
        style, weight = f"{meta.trigger}, ", drawing.get("style_weight") or meta.recommended_weight
    seeds = seeds or [drawing["seed"] + i for i in range(3)]
    view_words = "side view facing right" if side else "front view facing the viewer, symmetrical T-pose"
    progress.start(f"slices:{model}", ["encode", "denoise", "matte", "joints", "cut", "layers", "cut", "denoise"] +
                   (["matte"] if t.refine else []))
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
    spec0 = guide.spec.model_copy(update={"body": body.model_dump(), "prompt": prompt, "model": model})
    masters = []
    for sd, img, alpha in zip(seeds, images, alphas, strict=True):
        rgba, _, _ = real_rgba(img, alpha, recipe.post.alpha_low, recipe.post.alpha_high)
        folder = work / f"seed_{sd:06d}"
        masters.append((sd, folder, save_master(folder, spec0.model_copy(update={"seed": sd}),
                                                Image.fromarray(rgba, "RGBA"), guide)))
        img.save(folder / "master_raw.png")
    fitted = fit_masters([m for _, _, m in masters], rig, model, stages)
    if side and layout == "auto" and fitted:  # the drawing tells more than the design: suggest again from it
        lay, measures = suggest_from_try(masters[0][2], rig, cs.body_preset)
    return _cut_tries(project, rec, d, out, [(sd, folder, m) for sd, folder, m in masters], rig, lay, measures,
                      t, fitted, stages, refine=can_refine, model=model, profile=profile,
                      enforce_vram=enforce_vram)


def suggest_from_try(master_json: Path, base, body_preset: str):
    """The suggested layout from a fitted try (its plain-rig labels)."""
    import numpy as np
    from PIL import Image

    from ggs.rig.layout import suggest

    path = Path(master_json)
    v1 = path.parent / "master_v1.json"
    data = json.loads((v1 if v1.exists() else path).read_text(encoding="utf-8"))
    lab = path.parent / ("labels_v1.png" if v1.exists() else "guide_labels.png")
    labels = np.asarray(Image.open(lab).convert("L")).astype(np.int16) - 1
    joints = {b: (tuple(s), tuple(e)) for b, (s, e) in data["joints"].items()}
    return suggest(labels, base, joints, data["fit"]["widths"], data["height_px"], body_preset)


def relabel_v1(master_json: Path, base) -> None:
    """Recompute a fitted try's plain-rig labels (``labels_v1.png``) from its stored joints and parsing."""
    import numpy as np
    from PIL import Image

    from ggs.rig.parsing import load_parse
    from ggs.rig.skin import drawing_labels

    path = Path(master_json)
    v1 = path.parent / "master_v1.json"
    if not v1.exists():
        return
    data = json.loads(v1.read_text(encoding="utf-8"))
    widths = (data.get("fit") or {}).get("widths")
    if not widths:
        return
    rgba = np.asarray(Image.open(path.parent / data.get("image", "master.png")).convert("RGBA"))
    joints = {b: (tuple(s), tuple(e)) for b, (s, e) in data["joints"].items()}
    labels = drawing_labels(rgba, base, joints, widths, float(data["height_px"]), load_parse(path.parent / "parse.png"))
    Image.fromarray((labels + 1).astype(np.uint8), "L").save(path.parent / "labels_v1.png")


def recut_slices(project: Path, ref: str, layout) -> SliceResult:
    """Cut the tries already drawn again with another layout (no drawing: a few seconds per try)."""
    from ggs.items import CharacterSettings, ItemError, load
    from ggs.pipeline.vram import Stages
    from ggs.rig.poses import rig_for

    rec, d = load(Path(project), ref)
    out = sliced_dir(d)
    tries = sorted((out / "work").glob("seed_*/master.json"))
    if not tries:
        raise ItemError("no tries drawn yet: make the sliced version first")
    info = sliced_info(d)
    rig = rig_for("side")
    from ggs.items import design_dir

    drawing = json.loads((design_dir(d) / "design.json").read_text(encoding="utf-8"))["drawing"]
    cs = CharacterSettings(**(drawing.get("character") or {}))
    stages = Stages(enforce=False)
    unfitted = [m for m in tries if not (m.parent / "parse.png").exists()
                or not (json.loads(m.read_text(encoding="utf-8")).get("fit") or (m.parent / "master_v1.json").exists())]
    for m in unfitted:  # fit again from the plain-skeleton version of the try (not the layout's)
        v1 = m.parent / "master_v1.json"
        if v1.exists():
            shutil.move(str(v1), str(m))
            shutil.move(str(m.parent / "labels_v1.png"), str(m.parent / "guide_labels.png"))
    fitted = fit_masters(unfitted, rig, drawing.get("model") or "flux2-klein-4b", stages) if unfitted else True
    for m in tries:  # labels from the stored joints and body parsing, with the current rules
        relabel_v1(m, rig)
    if layout == "auto" and fitted:
        lay, measures = suggest_from_try(tries[0], rig, cs.body_preset)
    else:
        lay, measures = resolve_layout(layout, d, rig, cs)
    masters = [(int(m.parent.name.split("_")[1]), m.parent, m) for m in tries]
    return _cut_tries(Path(project), rec, d, out, masters, rig, lay, measures,
                      get_template(info.get("template", "side_simple")), bool(fitted), stages, refine=False)


def _layer_try(best: Path, rig, name: str, stages) -> bool:
    """See-through's complete layers for the chosen try (saved in ``<try>/layers``), and the try cut again from
    them. Kept only when it validates at least as well as the plain cut; otherwise the plain cut stays."""
    import numpy as np  # noqa: F401 - loaded with the layers
    from PIL import Image

    from ggs.pipeline.layers import SeeThroughLayers, save_layers
    from ggs.pipeline.models import MissingModelError, resolve_pose
    from ggs.rig.make import cut_master
    from ggs.rig.master import load_master

    try:
        files = {"model": resolve_pose(LAYERS)}
    except (MissingModelError, ValueError):
        return False
    folder = best.parent
    master_json = folder / "master.json"
    before = json.loads((best / "validation.json").read_text(encoding="utf-8"))
    lay_dir = folder / "layers"
    if not any(lay_dir.glob("*.png")):  # made once per try; re-applying a layout reuses them
        _spec, master, _labels = load_master(master_json)
        layers = SeeThroughLayers(files).decompose(Image.fromarray(master, "RGBA"), stages)
        save_layers(layers, lay_dir)
    trial = folder / "character_layers"
    if trial.exists():
        shutil.rmtree(trial)
    with stages.stage("cut"):
        outcome = cut_master(master_json, rig, trial, name)
    good = outcome.character is not None and (outcome.report.passed or not before.get("passed")) and \
        outcome.report.score >= float(before.get("score", 0)) - 0.02
    if good:
        shutil.rmtree(best)
        trial.rename(best)
    else:
        shutil.rmtree(trial, ignore_errors=True)
        shutil.rmtree(lay_dir)  # a later re-cut must not pick up layers that did worse
    print(f"layers: {'used' if good else 'not used'} (score {outcome.report.score:.3f} vs "
          f"{float(before.get('score', 0)):.3f})")
    return good


def _cut_tries(project, rec, d, out, masters, rig, lay, measures, t, fitted, stages, *, refine: bool,
               model: str = "", profile: str = "lite", enforce_vram: bool = True, layers: bool = True) -> SliceResult:
    from ggs.rig.make import cut_master

    tries, best = [], None
    cut_rig = rig
    rigs = {}
    with stages.stage("cut"):  # CPU: cut every try into pieces and check it (the longest part after drawing)
        for sd, folder, master in masters:
            if lay is not None and fitted:
                cut_rig = layout_master(master, rig, lay)
            rigs[folder] = cut_rig
            if (folder / "character").exists():
                shutil.rmtree(folder / "character")
            outcome = cut_master(master, cut_rig, folder / "character", rec.name)
            fails = [f"{c.name}: {c.detail}" for c in outcome.report.checks if not c.passed]
            tries.append({"seed": sd, "passed": outcome.report.passed, "score": round(outcome.report.score, 4),
                          "folder": str(folder), "failures": fails[:6], "cut": outcome.character is not None})
            if outcome.report.passed and best is None:
                best = folder / "character"
    rough = False
    if best is None:
        usable = [x for x in tries if x["cut"] and x["score"] >= ROUGH_IOU]
        if usable:
            best, rough = Path(max(usable, key=lambda x: x["score"])["folder"]) / "character", True
    layered = False
    if best is not None and fitted and layers:  # complete layers for the chosen try, then cut it again from them
        cut_rig = rigs.get(best.parent, cut_rig)
        layered = _layer_try(best, cut_rig, rec.name, stages)
        if layered:
            row = next(x for x in tries if Path(x["folder"]) == best.parent)
            report = json.loads((best / "validation.json").read_text(encoding="utf-8"))
            row.update(passed=bool(report.get("passed")), score=round(float(report.get("score", 0)), 4),
                       layered=True)
            rough = rough and not row["passed"]
    if best is not None and refine and (t.refine or not layered):
        # paint the hidden areas (not needed when the layers are whole); expressions for the detailed template
        from ggs.rig.refine import RefineSettings, refine_characters

        refine_characters([best], rigs.get(best.parent, cut_rig), model=model, profile=profile,
                          enforce_vram=enforce_vram, settings=RefineSettings(expressions=t.refine, fills=not layered))
    for f in out.glob("*"):  # replace the previous sliced version (the work folder stays)
        if f.name != "work":
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    if best is not None:
        for f in best.iterdir():
            if f.is_file():
                shutil.copy2(f, out / f.name)
    # the animation-ready drawing: the try that was cut best (else the first), for "Bend the drawing"
    pick = best.parent if best is not None else (Path(max(tries, key=lambda x: x["score"])["folder"])
                                                 if tries else None)
    rd = rigdraw_dir(d)
    if pick is not None and rig.view == "side":
        if rd.exists():
            shutil.rmtree(rd)
        rd.mkdir(parents=True)
        for f in pick.iterdir():
            if f.is_file():
                shutil.copy2(f, rd / f.name)
    (out / "slices.json").write_text(json.dumps({
        "format": "ggs.slices", "version": 1, "template": t.id, "rig": rig.id, "design": rec.accepted,
        "passed": best is not None and not rough, "rough": rough, "tries": tries, "joints_fitted": fitted,
        "layout": lay.model_dump() if lay is not None else None, "measures": measures}, indent=1),
        encoding="utf-8")
    return SliceResult(out / "character.yaml" if best is not None else None, tries, rough)


def pieces_preview(item_dir: Path) -> Path | None:
    """The sliced version's pieces only (the part sheet also holds copies of the head for expressions), in a
    small grid: ``sliced/pieces.png``, remade when the part sheet changes."""
    from PIL import Image


    d = sliced_dir(item_dir)
    sheet, info = d / "parts.png", d / "character.yaml"
    out = d / "pieces.png"
    if not sheet.exists() or not info.exists():
        return None
    made = max(sheet.stat().st_mtime, (d / "slices.json").stat().st_mtime if (d / "slices.json").exists() else 0)
    if out.exists() and out.stat().st_mtime > made:  # the sheet is copied with its old time: compare with the run
        return out

    from ggs.rig.character import load_character

    rig = load_character(info).rig()
    ps = rig.part_sheet
    img = Image.open(sheet).convert("RGBA")
    c = img.width // ps.cols
    cols = 3
    rows = -(-len(ps.parts) // cols)
    grid = Image.new("RGBA", (cols * c, rows * c), (0, 0, 0, 0))
    for k, p in enumerate(ps.parts):
        cell = img.crop((p.cell[0] * c, p.cell[1] * c, (p.cell[0] + 1) * c, (p.cell[1] + 1) * c))
        grid.alpha_composite(cell, ((k % cols) * c, (k // cols) * c))
    grid.save(out)
    return out


def sync_head_variants(item_dir: Path) -> None:
    """After the head was touched up: the expression slots (blink, mouth open, hurt) are copies of the head
    until expressions are drawn, so copy the new head into them."""
    from PIL import Image


    d = sliced_dir(item_dir)
    sheet, info = d / "parts.png", d / "character.yaml"
    if not sheet.exists() or not info.exists():
        return

    from ggs.rig.character import load_character

    rig = load_character(info).rig()
    ps = rig.part_sheet
    head = next((p for p in ps.parts if p.bone == "head"), None)
    if head is None or not ps.head_variants:
        return
    img = Image.open(sheet).convert("RGBA")
    c = img.width // ps.cols
    cell = img.crop((head.cell[0] * c, head.cell[1] * c, (head.cell[0] + 1) * c, (head.cell[1] + 1) * c))
    for v in ps.head_variants:
        img.paste(cell, (v.cell[0] * c, v.cell[1] * c))
    img.save(sheet)


def save_pieces(item_dir: Path, pieces_png: Path | None = None) -> None:
    """Write an edited pieces grid (``sliced/pieces.png``) back into the part sheet, cell by cell, and copy the
    head into the expression slots."""
    from PIL import Image

    from ggs.rig.character import load_character

    d = sliced_dir(item_dir)
    sheet_path, grid_path = d / "parts.png", Path(pieces_png or d / "pieces.png")
    rig = load_character(d / "character.yaml").rig()
    ps = rig.part_sheet
    sheet = Image.open(sheet_path).convert("RGBA")
    grid = Image.open(grid_path).convert("RGBA")
    c = sheet.width // ps.cols
    cols = 3
    for k, p in enumerate(ps.parts):
        cell = grid.crop(((k % cols) * c, (k // cols) * c, (k % cols + 1) * c, (k // cols + 1) * c))
        box = (p.cell[0] * c, p.cell[1] * c)
        sheet.paste(Image.new("RGBA", (c, c), (0, 0, 0, 0)), box)
        sheet.paste(cell, box)
    sheet.save(sheet_path)
    sync_head_variants(item_dir)
