"""Part refiner (M6): zero-shot klein edits on the cutter's part sheet, checked and with a safe fallback.

Two kinds of edit, all in one model load:
  fill        parts with a large filled-in area (the torso under the arm, the pelvis under the thigh) are
              repainted. Only the filled-in pixels (``parts_filled.png`` from the cutter) take the repaint;
              everything the master shows stays pixel-exact, and the part's shape (alpha) never changes.
  expression  each head variant (blink, mouth_open, hurt) is drawn by editing the neutral head. The result is
              matted and kept only if it has the same shape, place and colours as the head; otherwise the cell
              keeps the copy of the neutral head.
``refine.json`` in the character folder records what was done; ``parts_cut.png`` keeps the cutter's sheet.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ggs.rig.spec import RigSpec

KEEP = ("Keep everything else exactly the same: same character, same drawing style, same size, position and angle, "
        "plain white background.")
FILL_PROMPT = ("This is the {part} piece of a cut-out 2D game character. Repaint the blurry smudged areas so they "
               "match the rest of the piece, with clean shapes and crisp outlines. " + KEEP)
EXPR_PROMPT = "Edit the head in the image so {what}. " + KEEP
EXPRESSIONS = {
    "blink": "the eyes are closed (blinking)",
    "mouth_open": "the mouth is wide open as if shouting",
    "hurt": "it looks hurt: eyes squeezed shut, pained grimace",
}


@dataclass
class RefineSettings:
    scale: int = 3  # cells are edited at cell_px * scale (768 px for 256 px cells)
    min_fill: float = 0.08  # repaint a part when this share of its pixels was filled in
    expressions: bool = True
    fills: bool = True
    seed: int = 0
    min_head_iou: float = 0.85  # an expression must keep the head's shape and place
    max_head_colour: float = 0.15  # ... and its colours (mean |RGB difference| / 255)
    max_white: float = 0.30  # a repaint that turns this share of the filled area white is refused


@dataclass
class _Job:
    char_dir: Path
    kind: str  # "fill" | "expression"
    name: str  # part name or head variant name
    cell: tuple[int, int]
    image: object  # PIL RGB input for klein
    prompt: str
    mask: np.ndarray | None = None  # fill: soft filled-in mask of the cell (0..1)


@dataclass
class RefineReport:
    char_dir: Path
    edits: list[dict] = field(default_factory=list)

    @property
    def applied(self) -> int:
        return sum(e["applied"] for e in self.edits)


def _cell(arr: np.ndarray, cell: tuple[int, int], px: int) -> np.ndarray:
    c, r = cell
    return arr[r * px:(r + 1) * px, c * px:(c + 1) * px]


def _on_white(rgba: np.ndarray, size: int):
    from PIL import Image

    from ggs.rig.make import on_white

    return on_white(Image.fromarray(rgba, "RGBA")).resize((size, size), Image.LANCZOS)


def source_sheet(char_dir: Path) -> Path:
    """The cutter's sheet: refining again starts from it, not from an earlier refinement."""
    return char_dir / "parts_cut.png" if (char_dir / "parts_cut.png").exists() else char_dir / "parts.png"


def plan(char_dir: Path, rig: RigSpec, s: RefineSettings) -> list[_Job]:
    from PIL import Image

    sheet = np.asarray(Image.open(source_sheet(char_dir)).convert("RGBA"))
    filled_path = char_dir / "parts_filled.png"
    filled = np.asarray(Image.open(filled_path).convert("L")) if filled_path.exists() else None
    px = rig.part_sheet.cell_px
    size = px * s.scale
    jobs = []
    if s.fills and filled is not None:
        for p in rig.part_sheet.parts:
            a = _cell(sheet, p.cell, px)[..., 3] > 0
            m = _cell(filled, p.cell, px).astype(np.float32) / 255.0
            frac = float((m[a] > 0.5).sum() / max(a.sum(), 1))
            if frac >= s.min_fill:
                jobs.append(_Job(char_dir, "fill", p.name, p.cell, _on_white(_cell(sheet, p.cell, px), size),
                                 FILL_PROMPT.format(part=p.name.replace("_", " ")), m))
    head = next((p for p in rig.part_sheet.parts if p.name == "head"), None)
    if s.expressions and head is not None:
        img = _on_white(_cell(sheet, head.cell, px), size)
        for v in rig.part_sheet.head_variants:
            what = EXPRESSIONS.get(v.name, f"it shows the expression '{v.name.replace('_', ' ')}'")
            jobs.append(_Job(char_dir, "expression", v.name, v.cell, img, EXPR_PROMPT.format(what=what)))
    return jobs


def refine_characters(char_dirs: list[Path], rig: RigSpec, *, model: str = "flux2-klein-4b", profile: str = "lite",
                      settings: RefineSettings | None = None, enforce_vram: bool = True) -> list[RefineReport]:
    """Refine every character folder in place (one generator load, one matting load for all of them)."""
    from PIL import Image

    from ggs.generate import build_backends, real_rgba
    from ggs.pipeline.vram import Stages
    from ggs.recipes import load_recipe

    s = settings or RefineSettings()
    jobs = [j for d in char_dirs for j in plan(Path(d), rig, s)]
    reports = {Path(d): RefineReport(Path(d)) for d in char_dirs}
    if not jobs:
        return list(reports.values())
    recipe = load_recipe("sprite")
    gen, matter = build_backends(recipe, model, profile)
    if not getattr(gen, "supports_reference", False):
        raise ValueError(f"{model!r} cannot edit images; the part refiner needs flux2-klein-4b")
    px = rig.part_sheet.cell_px
    size = px * s.scale
    stages = Stages(profile=profile, enforce=enforce_vram)
    images = gen.render([(j.prompt, s.seed, [j.image]) for j in jobs], size, size, stages)
    expr = [i for i, j in enumerate(jobs) if j.kind == "expression"]
    alphas = dict(zip(expr, matter.alphas([images[i] for i in expr], stages))) if expr else {}

    sheets = {d: np.asarray(Image.open(source_sheet(d)).convert("RGBA")).copy() for d in reports}
    head = next((p for p in rig.part_sheet.parts if p.name == "head"), None)
    for i, (j, img) in enumerate(zip(jobs, images)):
        sheet = sheets[j.char_dir]
        c, r = j.cell
        ys, xs = slice(r * px, (r + 1) * px), slice(c * px, (c + 1) * px)
        if j.kind == "fill":
            rep = np.asarray(img.convert("RGB").resize((px, px), Image.LANCZOS)).astype(np.float32)
            cur = sheet[ys, xs]
            inside = (j.mask > 0.5) & (cur[..., 3] > 0)
            white = float((rep[inside].min(axis=1) > 235).mean()) if inside.any() else 1.0
            ok = white <= s.max_white
            if ok:
                w = (j.mask * (cur[..., 3] > 0))[..., None]
                cur[..., :3] = np.clip(cur[..., :3] * (1 - w) + rep * w, 0, 255).astype(np.uint8)
            reports[j.char_dir].edits.append({"kind": "fill", "part": j.name, "applied": ok,
                                              "white_share": round(white, 3)})
        else:
            rgba, _, _ = real_rgba(img, alphas[i], recipe.post.alpha_low, recipe.post.alpha_high)
            small = np.asarray(Image.fromarray(rgba, "RGBA").convert("RGBa").resize((px, px), Image.LANCZOS)
                               .convert("RGBA"))
            ref = _cell(sheet, head.cell, px)
            a, b = small[..., 3] >= 128, ref[..., 3] >= 128
            iou = float((a & b).sum() / max((a | b).sum(), 1))
            both = a & b
            colour = float(np.abs(small[..., :3].astype(np.float32) - ref[..., :3])[both].mean() / 255) if both.any() \
                else 1.0
            ok = iou >= s.min_head_iou and colour <= s.max_head_colour
            if ok:
                sheet[ys, xs] = small
            reports[j.char_dir].edits.append({"kind": "expression", "variant": j.name, "applied": ok,
                                              "iou": round(iou, 3), "colour_error": round(colour, 3)})
    for d, rep in reports.items():
        if rep.applied:
            if not (d / "parts_cut.png").exists():
                shutil.copyfile(d / "parts.png", d / "parts_cut.png")
            Image.fromarray(sheets[d], "RGBA").save(d / "parts.png")
        (d / "refine.json").write_text(json.dumps({"model": model, "edits": rep.edits,
                                                   "stages": stages.summary()}, indent=1), encoding="utf-8")
    return list(reports.values())
