"""Frame-by-frame sprites: the generator draws the character in every pose of a clip (the second animation route).

The rig route cuts a master into parts and animates them; this route renders the clip's poses as a *pose sheet*
(the guide puppet in the character's body shape, one pose per cell) and lets klein draw the character from the
master in every cell, all frames in one image so they stay consistent. Cloth, hair and capes move the way the
model draws them, which the cut-out rig cannot do; the price is that the result is a fixed animation (GIF, APNG,
sprite sheet), not editable parts.

Every frame is checked: it must follow its pose, keep the master's colours and keep the character's height.
Several seeds are tried and the best sheet is exported.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ggs.rig.spec import RigSpec

SPRITES_PROMPT = (
    "{style}Image 1 is a sheet of {n} {motion} poses of a mannequin, image 2 is a game character: {who}. Redraw "
    "every mannequin in image 1 as the character from image 2: the same character in every cell, with the same "
    "face, hair, clothes, colours and proportions as image 2, each in exactly the pose, size and place of its "
    "mannequin, side view facing right, clothes and hair moving naturally with the motion. Flat 2D game art with "
    "clean outlines, plain white background, no ground, no shadow"
)


@dataclass
class SpriteSettings:
    frames: int = 8
    cols: int = 4
    rows: int = 2
    width: int = 1024
    height: int = 768
    fill: float = 0.80  # puppet standing height / cell height
    min_precision: float = 0.70  # share of the frame inside its pose (grown by `grow`)
    min_recall: float = 0.75  # share of the pose covered by the frame (grown by `grow`)
    grow: float = 0.06  # of the cell height: clothes are wider than the mannequin
    max_colour: float = 0.35  # palette distance to the master (histogram, 0 same .. 1 disjoint)
    max_height_dev: float = 0.12  # frame height vs the median frame height


@dataclass
class FrameCheck:
    index: int
    passed: bool
    scores: dict
    why: list[str] = field(default_factory=list)


def describe_master(prompt: str) -> str:
    """The character description inside a master prompt ("Redraw the figure in the image as X. Keep ...")."""
    m = re.search(r"Redraw the figure in the image as (.*?)\. Keep exactly", prompt)
    return m.group(1) if m else "the character"


def pose_sheet(rig: RigSpec, puppet, clip, s: SpriteSettings):
    """(sheet RGB on white, per-cell guide alpha masks, frame indices, cell anchor (x, ground y))."""
    from PIL import Image

    from ggs.rig.make import on_white

    n = s.frames
    if n > s.cols * s.rows:
        raise ValueError(f"{n} frames do not fit a {s.cols}x{s.rows} sheet")
    idx = [round(i * clip.frame_count / n) % clip.frame_count for i in range(n)]
    cw, ch = s.width // s.cols, s.height // s.rows
    frames, anchor = puppet.render_clip(clip, int(ch * s.fill), padding=2)
    cell_anchor = (cw // 2, int(ch * 0.95))
    sheet = Image.new("RGBA", (s.width, s.height), (0, 0, 0, 0))
    masks = []
    for k, i in enumerate(idx):
        f = frames[i]
        c, r = k % s.cols, k // s.cols
        cell = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        cell.alpha_composite(f, (cell_anchor[0] - anchor[0], cell_anchor[1] - anchor[1]))
        sheet.alpha_composite(cell, (c * cw, r * ch))
        masks.append(np.asarray(cell.getchannel("A")) >= 128)
    return on_white(sheet), masks, idx, cell_anchor


def _hist(rgba: np.ndarray, bins: int = 8) -> np.ndarray:
    on = rgba[..., 3] >= 128
    q = (rgba[..., :3][on] // (256 // bins)).astype(np.int32)
    h = np.bincount(q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2], minlength=bins ** 3).astype(np.float64)
    return h / max(h.sum(), 1.0)


def palette_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - histogram intersection of the opaque pixels' colours (0 = same palette)."""
    return float(1.0 - np.minimum(_hist(a), _hist(b)).sum())


def check_frames(cells: list[np.ndarray], masks: list[np.ndarray], master: np.ndarray,
                 s: SpriteSettings) -> list[FrameCheck]:
    from ggs.rig.cutter import _grow

    ch = cells[0].shape[0]
    g = max(1, round(s.grow * ch))
    heights = []
    for c in cells:
        ys = np.flatnonzero((c[..., 3] >= 128).any(axis=1))
        heights.append(int(ys[-1] - ys[0] + 1) if ys.size else 0)
    median = float(np.median([h for h in heights if h] or [1]))
    out = []
    for k, (c, m) in enumerate(zip(cells, masks)):
        a = c[..., 3] >= 128
        if a.sum() < 50:
            out.append(FrameCheck(k, False, {}, ["empty"]))
            continue
        precision = float((a & _grow(m, g)).sum() / a.sum())
        recall = float((m & _grow(a, g)).sum() / max(m.sum(), 1))
        colour = palette_distance(c, master)
        dev = abs(heights[k] - median) / median
        why = [w for w, bad in ((f"off its pose ({precision:.2f})", precision < s.min_precision),
                                (f"misses part of its pose ({recall:.2f})", recall < s.min_recall),
                                (f"colours differ from the master ({colour:.2f})", colour > s.max_colour),
                                (f"height {dev:.0%} off the other frames", dev > s.max_height_dev)) if bad]
        out.append(FrameCheck(k, not why, {"precision": round(precision, 3), "recall": round(recall, 3),
                                           "colour": round(colour, 3), "height_dev": round(dev, 3)}, why))
    return out


def crop_frames(cells: list[np.ndarray], cell_anchor: tuple[int, int], padding: int = 2):
    """Crop every cell to one shared box: (PIL frames, anchor = feet point in frame pixels)."""
    from PIL import Image

    union = np.zeros(cells[0].shape[:2], bool)
    for c in cells:
        union |= c[..., 3] > 0
    ys, xs = np.nonzero(union)
    h, w = union.shape
    y0, y1 = max(0, ys.min() - padding), min(h, ys.max() + 1 + padding)
    x0, x1 = max(0, xs.min() - padding), min(w, xs.max() + 1 + padding)
    frames = [Image.fromarray(c[y0:y1, x0:x1], "RGBA") for c in cells]
    return frames, (int(cell_anchor[0] - x0), int(min(cell_anchor[1], y1) - y0))


@dataclass
class SpriteResult:
    seed: int
    folder: Path
    checks: list[FrameCheck]
    frames: list  # PIL RGBA, cropped to one box
    anchor: tuple[int, int]

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.checks)

    @property
    def score(self) -> float:
        """For picking the best seed: frames passed first, then the mean precision/recall and colours."""
        sc = [c.scores for c in self.checks if c.scores]
        if not sc:
            return float(self.passed)
        quality = np.mean([(x["precision"] + x["recall"]) / 2 - x["colour"] for x in sc])
        return self.passed + float(quality)


def make_sprites(master_json: str | Path, clip_path: str | Path, rig: RigSpec, out_dir: str | Path, *,
                 seeds: list[int], model: str = "flux2-klein-4b", profile: str = "lite",
                 settings: SpriteSettings | None = None, style_ref: str | None = None,
                 style_weight: float | None = None, project: Path | None = None, motion: str = "walking",
                 enforce_vram: bool = True) -> list[SpriteResult]:
    """Draw the clip frame by frame for every seed; writes one folder per seed. Results best first."""
    from PIL import Image

    from ggs.generate import build_backends, real_rgba
    from ggs.pipeline.vram import Stages
    from ggs.recipes import load_recipe
    from ggs.rig.body import BodyShape
    from ggs.rig.clip import load_clip
    from ggs.rig.make import guide_puppet, on_white
    from ggs.rig.master import MasterSpec

    s = settings or SpriteSettings()
    master_json = Path(master_json)
    spec = MasterSpec.model_validate_json(master_json.read_text(encoding="utf-8"))
    master = np.asarray(Image.open(master_json.parent / spec.image).convert("RGBA"))
    clip = load_clip(clip_path)
    if clip.rig != rig.id:
        raise ValueError(f"clip is for rig {clip.rig!r}, not {rig.id!r}")
    puppet, _ = guide_puppet(rig, BodyShape(**spec.body) if spec.body else None)
    sheet, masks, idx, cell_anchor = pose_sheet(rig, puppet, clip, s)
    style, lora = "", None
    if style_ref:
        from ggs.style.package import StyleError, resolve_style

        meta, lora = resolve_style(style_ref, project)
        if model != "stub" and meta.base_model != model:
            raise StyleError(f"style {meta.name!r} was trained on {meta.base_model!r}; it cannot load on {model!r}")
        style = f"{meta.trigger}, "
        style_weight = meta.recommended_weight if style_weight is None else style_weight
    recipe = load_recipe("sprite")
    gen, matter = build_backends(recipe, model, profile)
    if not getattr(gen, "supports_reference", False):
        raise ValueError(f"{model!r} cannot follow reference images; frame-by-frame sprites need flux2-klein-4b")
    prompt = " ".join(SPRITES_PROMPT.format(style=style, n=s.frames, motion=motion,
                                            who=describe_master(spec.prompt)).split())
    ref_master = on_white(Image.fromarray(master, "RGBA")).resize((768, 768), Image.LANCZOS)
    stages = Stages(profile=profile, enforce=enforce_vram)
    images = gen.render([(prompt, seed, [sheet, ref_master]) for seed in seeds], s.width, s.height, stages,
                        lora=lora, lora_scale=1.0 if style_weight is None else style_weight)
    alphas = matter.alphas(images, stages)
    out = Path(out_dir)
    cw, ch = s.width // s.cols, s.height // s.rows
    results = []
    for seed, img, alpha in zip(seeds, images, alphas):
        rgba, _, _ = real_rgba(img, alpha, recipe.post.alpha_low, recipe.post.alpha_high)
        cells = [rgba[(k // s.cols) * ch:(k // s.cols + 1) * ch, (k % s.cols) * cw:(k % s.cols + 1) * cw]
                 for k in range(s.frames)]
        checks = check_frames(cells, masks, master, s)
        frames, anchor = crop_frames(cells, cell_anchor)
        folder = out / f"seed_{seed:03d}"
        (folder / "frames").mkdir(parents=True, exist_ok=True)
        img.save(folder / "sheet_raw.png")
        sheet.save(folder / "poses.png")
        for k, f in enumerate(frames):
            f.save(folder / "frames" / f"frame_{k:02d}.png")
        (folder / "report.json").write_text(json.dumps({
            "seed": seed, "prompt": prompt, "clip": str(clip_path), "clip_frames": idx,
            "frames": [{"index": c.index, "passed": c.passed, **c.scores, "why": c.why} for c in checks]},
            indent=1), encoding="utf-8")
        results.append(SpriteResult(seed, folder, checks, frames, anchor))
    (out / "stages.json").write_text(json.dumps(stages.summary(), indent=1), encoding="utf-8")
    return sorted(results, key=lambda r: -r.score)
