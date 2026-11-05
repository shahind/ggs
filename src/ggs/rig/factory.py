"""Pair factory for the rig-sheet adapter (M6): (master image, part sheet) pairs that match exactly.

1. klein redraws the programmer-art *part sheet* as a new character (the sheet is the reference image).
2. Per-cell QA keeps only sheets whose parts sit where the rig expects them (one piece per cell, same place,
   length and joint as the guide part, nothing clipped).
3. The puppet renderer assembles the kept sheet in the master pose: that render is the pair's master, with the
   guide joints and labels of `ggs character master`, so the pair is consistent by construction.
Pairs live in a folder per pair: parts.png + character.yaml (the target), master.png + master.json (the input),
meta.json (prompt, seed, QA).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ggs.rig.spec import RigSpec

SHEET_PX_PER_HEIGHT = 540.0  # the guide sheet (and so every factory sheet) is drawn at this scale
FACTORY_PROMPT = (
    "Redraw every body part in this character part sheet as {who}. Keep the exact layout: each part stays in its "
    "own place with the same size, angle and joint position, separate pieces with empty white space between "
    "them. The face looks to the right. Flat 2D game art with clean outlines, plain white background"
)

# Deterministic character list: archetype x look. Order matters (seeded runs pick from the front).
_ARCHETYPES = [
    "a knight in {c} plate armour", "a wizard in a {c} robe", "a pirate with a {c} coat", "a ninja in a {c} outfit",
    "a robot with {c} metal panels", "an elf ranger in {c} leather", "a zombie in torn {c} clothes",
    "a farmer in {c} overalls", "an astronaut in a {c} space suit", "a cowboy with a {c} vest",
    "an orc warrior with {c} armour", "a skeleton with {c} bones", "a samurai in {c} armour",
    "a cyberpunk hacker in a {c} jacket", "a viking with a {c} tunic", "a chef in a {c} uniform",
    "a soldier in {c} camouflage", "a monk in a {c} robe", "a goblin in a {c} rag tunic", "a detective in a {c} coat",
    "a superhero in a {c} suit", "a lumberjack in a {c} plaid shirt", "a mechanic in {c} work clothes",
    "a medieval peasant in a {c} tunic", "a space marine in {c} armour", "a vampire in a {c} suit",
    "a rogue in a {c} hood", "a paladin in {c} and gold armour", "a sailor in a {c} uniform",
    "an archer in a {c} tunic", "a dwarf miner in {c} clothes", "a punk with a {c} jacket",
    "a firefighter in a {c} suit", "a gladiator with {c} armour", "a scientist in a {c} lab coat",
    "a ghost hunter in a {c} jumpsuit", "a desert nomad in {c} cloth", "a cyborg with {c} plates",
    "a king in a {c} royal outfit", "a hiker in a {c} jacket",
]
_COLOURS = ["red", "blue", "green", "black", "white", "purple", "orange", "brown", "grey", "yellow"]


def character_prompts(n: int, start: int = 0) -> list[str]:
    out = []
    for k in range(start, start + n):
        a = _ARCHETYPES[k % len(_ARCHETYPES)]
        c = _COLOURS[(k * 7 + k // len(_ARCHETYPES)) % len(_COLOURS)]
        out.append(a.format(c=c))
    return out


@dataclass
class SheetGuide:
    image: object  # PIL RGB on white, the reference
    alpha: np.ndarray  # guide sheet alpha, bool
    character_yaml: str  # programmer-art character.yaml text (lengths, pivots, px_per_height)


def sheet_guide(rig: RigSpec, body=None) -> SheetGuide:
    """The programmer-art part sheet (in ``body``'s proportions), drawn as large as fits every cell."""
    from PIL import Image

    from ggs.rig.character import make_programmer_art
    from ggs.rig.make import on_white

    cell = rig.part_sheet.cell_px
    for pph in (SHEET_PX_PER_HEIGHT, 500, 460, 420, 380, 340, 300, 260):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                path = make_programmer_art(rig, tmp, px_per_height=pph, body=body)
            except ValueError:  # a part does not fit its cell at this size
                continue
            sheet = Image.open(Path(tmp) / "parts.png").convert("RGBA")
            sheet.load()
            text = path.read_text(encoding="utf-8")
        a = np.asarray(sheet.getchannel("A")) > 0
        edges = a[::cell].any() or a[cell - 1::cell].any() or a[:, ::cell].any() or a[:, cell - 1::cell].any()
        if not edges:
            return SheetGuide(on_white(sheet), np.asarray(sheet.getchannel("A")) >= 128, text)
    raise ValueError("the body shape does not fit the rig's part cells")


# ---------------------------------------------------------------- per-cell QA

@dataclass
class CellCheck:
    name: str
    passed: bool
    detail: str
    scores: dict = field(default_factory=dict)


def _cells(rig: RigSpec):
    """(label, cell, part) for every part and head variant (variants are checked against the head)."""
    ps = rig.part_sheet
    head = next((p for p in ps.parts if p.name == "head"), None)
    for p in ps.parts:
        yield p.name, p.cell, p
    for v in ps.head_variants:
        if head is not None:
            yield f"head:{v.name}", v.cell, head


def check_sheet(alpha: np.ndarray, guide: SheetGuide, rig: RigSpec, dilate_frac: float = 0.06,
                min_precision: float = 0.6, min_recall: float = 0.6) -> list[CellCheck]:
    """Does each cell hold one part where the guide part is? ``alpha`` = generated sheet alpha (bool)."""
    from ggs.rig.cutter import _grow, component_sizes

    cell = rig.part_sheet.cell_px
    d = max(2, round(dilate_frac * cell))
    head_cell = next(p.cell for p in rig.part_sheet.parts if p.name == "head")
    out = []
    for label, (col, row), part in _cells(rig):
        a = alpha[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell]
        gc = head_cell if label.startswith("head:") else part.cell
        g = guide.alpha[gc[1] * cell:(gc[1] + 1) * cell, gc[0] * cell:(gc[0] + 1) * cell]
        if a.sum() < 30:
            out.append(CellCheck(label, False, "empty"))
            continue
        sizes = component_sizes(a)
        one = sizes[0] / sum(sizes)
        border = bool(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())
        precision = float((a & _grow(g, d)).sum() / a.sum())  # the part lies where the guide part is
        recall = float((g & _grow(a, d)).sum() / g.sum())  # and covers it (same length and angle)
        px, py = round(part.pivot[0] * cell), round(part.pivot[1] * cell)
        ys, xs = np.nonzero(a)
        joint = float(np.sqrt(((xs - px) ** 2 + (ys - py) ** 2).min()) / cell)
        ok = one >= 0.9 and not border and precision >= min_precision and recall >= min_recall and joint <= 0.08
        why = [w for w, bad in (("in pieces", one < 0.9), ("touches the cell border", border),
                                (f"outside the guide part ({precision:.2f})", precision < min_precision),
                                (f"does not cover the guide part ({recall:.2f})", recall < min_recall),
                                (f"joint {joint:.2f} cells away", joint > 0.08)) if bad]
        out.append(CellCheck(label, ok, "; ".join(why) or "ok",
                             {"pieces": round(one, 3), "precision": round(precision, 3), "recall": round(recall, 3),
                              "joint": round(joint, 3)}))
    return out


# ---------------------------------------------------------------- pairs

def assemble_pair(sheet_rgba: np.ndarray, guide: SheetGuide, rig: RigSpec, out_dir: Path, meta: dict,
                  size: int = 1024) -> Path:
    """Write a pair folder: the sheet as a character, and its master-pose render as master.json/master.png."""
    from PIL import Image

    from ggs.rig.character import load_character
    from ggs.rig.master import render_guide, save_master

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet_rgba, "RGBA").save(out_dir / "parts.png")
    (out_dir / "character.yaml").write_text(guide.character_yaml, encoding="utf-8")
    ch = load_character(out_dir / "character.yaml")
    g = render_guide(rig, ch.puppet(rig), ch.spec.bone_lengths, size=size)
    spec = g.spec.model_copy(update={"prompt": meta.get("prompt", ""), "seed": meta.get("seed"),
                                     "model": "pair-factory"})
    path = save_master(out_dir, spec, g.image, g)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return path


def make_pairs(rig: RigSpec, out_dir: str | Path, prompts: list[str], seeds: list[int], *, model: str = "flux2-klein-4b",
               profile: str = "lite", enforce_vram: bool = True, keep_rejected: bool = True,
               batch: int = 24) -> dict:
    """Generate sheets for every (prompt, seed), QA them, and write pair folders for the ones that pass."""

    from ggs.generate import build_backends, real_rgba
    from ggs.pipeline.vram import Stages
    from ggs.recipes import load_recipe

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    guide = sheet_guide(rig)
    guide.image.save(out / "sheet_guide.png")
    recipe = load_recipe("sprite")
    gen, matter = build_backends(recipe, model, profile)
    if not getattr(gen, "supports_reference", False):
        raise ValueError(f"{model!r} cannot follow a reference sheet; the pair factory needs flux2-klein-4b")
    w, h = guide.image.size
    jobs = [(p, s) for p in prompts for s in seeds]
    summary = {"kept": 0, "rejected": 0, "pairs": []}
    for b0 in range(0, len(jobs), batch):
        chunk = jobs[b0:b0 + batch]
        stages = Stages(profile=profile, enforce=enforce_vram)
        texts = [" ".join(FACTORY_PROMPT.format(who=p).split()) for p, _ in chunk]
        images = gen.render([(t, s, [guide.image]) for t, (_, s) in zip(texts, chunk)], w, h, stages)
        alphas = matter.alphas(images, stages)
        for (who, seed), text, img, alpha in zip(chunk, texts, images, alphas):
            rgba, _, _ = real_rgba(img, alpha, recipe.post.alpha_low, recipe.post.alpha_high)
            checks = check_sheet(rgba[..., 3] >= 128, guide, rig)
            ok = all(c.passed for c in checks)
            pid = f"{len(summary['pairs']):04d}"
            meta = {"id": pid, "who": who, "prompt": text, "seed": seed, "passed": ok,
                    "cells": {c.name: {"passed": c.passed, "detail": c.detail, **c.scores} for c in checks}}
            if ok:
                folder = out / "pairs" / pid
                assemble_pair(rgba, guide, rig, folder, meta)
                img.save(folder / "sheet_raw.png")
                summary["kept"] += 1
            else:
                summary["rejected"] += 1
                if keep_rejected:
                    folder = out / "rejected" / pid
                    folder.mkdir(parents=True, exist_ok=True)
                    img.save(folder / "sheet_raw.png")
                    (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            summary["pairs"].append({"id": pid, "who": who, "seed": seed, "passed": ok,
                                     "failed": [c.name for c in checks if not c.passed]})
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"batch {b0 // batch + 1}: {summary['kept']} kept, {summary['rejected']} rejected", flush=True)
    return summary


def load_pairs(folder: str | Path) -> list[Path]:
    """Pair folders (with master.json + parts.png) under ``folder`` or ``folder/pairs``."""
    root = Path(folder)
    base = root / "pairs" if (root / "pairs").is_dir() else root
    return sorted(p for p in base.iterdir() if (p / "master.json").exists() and (p / "parts.png").exists())
