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
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ggs.rig.spec import RigSpec

SPRITES_PROMPT = (
    "{style}Image 1 is a sheet of {n} stick figures showing the poses of a {motion} animation, image 2 is a game "
    "character: {who}. Draw the character from image 2 once in every cell of image 1: the same character with the "
    "same face, hair, clothes, colours and proportions as image 2, each in exactly the pose, size and place of its "
    "stick figure, side view facing right, clothes and hair moving naturally with the motion. The stick figures "
    "are only pose references: do not draw their lines or dots. Flat 2D game art with clean outlines, plain white "
    "background, no ground, no shadow"
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
    m = (re.search(r"Redraw the figure in the image as (.*?)\. Keep exactly", prompt)
         or re.search(r"Draw a new full-body character: (.*?)\. Give the character", prompt))
    return m.group(1) if m else "the character"


def settings_for(frames: int, cell: int = 256) -> SpriteSettings:
    """Sheet layout for ``frames`` poses: up to 4 per row."""
    if not 2 <= frames <= 16:
        raise ValueError("draw between 2 and 16 frames")
    cols = min(frames, 4)
    rows = -(-frames // cols)
    return SpriteSettings(frames=frames, cols=cols, rows=rows, width=cols * cell, height=rows * round(cell * 1.5))


def frame_indices(clip, n: int) -> list[int]:
    return [round(i * clip.frame_count / n) % clip.frame_count for i in range(n)]


def pose_sheet(rig: RigSpec, puppet, clip, s: SpriteSettings, lengths: dict | None = None):
    """(sheet RGB on white, per-cell guide alpha masks, frame indices, cell anchor (x, ground y)).
    With ``lengths`` (the puppet's bone lengths) the sheet shows stick figures, which is what the model is given
    (it copies a mannequin's shapes); the masks always come from the puppet."""
    from PIL import Image, ImageDraw

    from ggs.rig.make import on_white
    from ggs.rig.pose import bone_defs, pose_bones
    from ggs.rig.stick import draw_stick

    n = s.frames
    if n > s.cols * s.rows:
        raise ValueError(f"{n} frames do not fit a {s.cols}x{s.rows} sheet")
    idx = frame_indices(clip, n)
    cw, ch = s.width // s.cols, s.height // s.rows
    height_px = int(ch * s.fill)
    frames, anchor = puppet.render_clip(clip, height_px, padding=2)
    root = puppet.last_root
    cell_anchor = (cw // 2, int(ch * 0.95))
    sheet = Image.new("RGBA", (s.width, s.height), (0, 0, 0, 0))
    sticks = Image.new("RGB", (s.width, s.height), (255, 255, 255))
    draw = ImageDraw.Draw(sticks)
    defs = bone_defs(rig)
    masks = []
    for k, i in enumerate(idx):
        f = frames[i]
        c, r = k % s.cols, k // s.cols
        cell = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        dx, dy = cell_anchor[0] - anchor[0], cell_anchor[1] - anchor[1]
        cell.alpha_composite(f, (dx, dy))
        sheet.alpha_composite(cell, (c * cw, r * ch))
        masks.append(np.asarray(cell.getchannel("A")) >= 128)
        if lengths is not None:
            joints = pose_bones(defs, clip, i, lengths, in_place=True)
            draw_stick(draw, rig, joints, scale=height_px,
                       offset=(c * cw + dx + root[0], r * ch + dy + root[1]), width=max(2.0, ch / 90))
    return (sticks if lengths is not None else on_white(sheet)), masks, idx, cell_anchor


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
    puppet, lengths = guide_puppet(rig, BodyShape(**spec.body) if spec.body else None)
    sheet, masks, idx, cell_anchor = pose_sheet(rig, puppet, clip, s, lengths)
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


def frames_dir(item_dir: Path) -> Path:
    return Path(item_dir) / "frames"


def frames_method() -> str:
    """'animatediff' when its models are downloaded, else 'klein' (the drawing model)."""
    from ggs.pipeline.models import MissingModelError, resolve_frames

    try:
        resolve_frames(ANIMATEDIFF)
        return "animatediff"
    except (MissingModelError, ValueError):
        return "klein"


ANIMATEDIFF = "animatediff-sd15"
AD_SIZE = (384, 512)


def _design_lengths(d: Path) -> dict[str, float]:
    from ggs.items import design_dir

    sk = json.loads((design_dir(d) / "design_skeleton.json").read_text(encoding="utf-8"))
    hp = float(sk["px_per_height"])
    return {b["name"]: max(math.hypot(b["end"][0] - b["start"][0], b["end"][1] - b["start"][1]) / hp, 1e-3)
            for b in sk["bones"]}


def openpose_frames(rig, clip, idx: list[int], lengths: dict[str, float], size=AD_SIZE, fill: float = 0.78):
    """One OpenPose image per frame index: the character's own proportions, in place, centred."""
    from ggs.rig.pose import bone_defs, pose_bones
    from ggs.rig.stick import openpose_image

    w, h = size
    scale = fill * h
    out = []
    for f in idx:
        p = pose_bones(bone_defs(rig), clip, f, lengths, 1.0, True)
        ys = [y for s_, e in p.values() for y in (s_[1], e[1])]
        oy = h * 0.5 - (min(ys) + max(ys)) / 2 * scale
        joints = {b: ((w / 2 + s_[0] * scale, oy + s_[1] * scale), (w / 2 + e[0] * scale, oy + e[1] * scale))
                  for b, (s_, e) in p.items()}
        out.append(openpose_image(joints, size, "side"))
    return out


def _animatediff_frames(d: Path, info: dict, clip, n: int, seed: int, profile: str, enforce_vram: bool):
    """(RGBA frames cropped to one box, anchor, pose strip) drawn by AnimateDiff."""
    import numpy as np
    from PIL import Image

    from ggs.generate import real_rgba
    from ggs.items import design_dir
    from ggs.pipeline.animatediff import AnimateDiffFrames
    from ggs.pipeline.matting import BiRefNetMatter
    from ggs.pipeline.models import load_pipelines, resolve_frames, resolve_matting
    from ggs.pipeline.vram import Stages
    from ggs.rig.poses import rig_for

    rig = rig_for("side")
    idx = frame_indices(clip, n)
    poses = openpose_frames(rig, clip, idx, _design_lengths(d))
    design = Image.open(design_dir(d) / "design.png").convert("RGBA")
    ref = Image.new("RGB", design.size, "white")
    ref.paste(design, mask=design.getchannel("A"))
    who = "; ".join(info.get("text") or ["a character"])
    prompt = (f"{who}, full body, side view, facing right, flat cartoon game character, clean outlines, plain white "
              "background")
    spec = load_pipelines()
    stages = Stages(profile=profile, enforce=enforce_vram)
    rgb = AnimateDiffFrames(resolve_frames(ANIMATEDIFF, spec_file=spec)).generate(
        poses, ref, prompt, seed, AD_SIZE[0], AD_SIZE[1], stages)
    alphas = BiRefNetMatter(resolve_matting("birefnet", spec_file=spec)).alphas(rgb, stages)
    rgba = [Image.fromarray(real_rgba(im, a)[0], "RGBA") for im, a in zip(rgb, alphas, strict=True)]
    box = None
    for im in rgba:
        bb = im.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
        if bb:
            box = bb if box is None else (min(box[0], bb[0]), min(box[1], bb[1]), max(box[2], bb[2]), max(box[3], bb[3]))
    if box is None:
        raise ValueError("the frames came out empty")
    pad = 4
    box = (max(0, box[0] - pad), max(0, box[1] - pad), min(AD_SIZE[0], box[2] + pad), min(AD_SIZE[1], box[3] + pad))
    frames = [im.crop(box) for im in rgba]
    anchor = (round(AD_SIZE[0] / 2 - box[0]), box[3] - box[1] - pad)
    strip = Image.new("RGB", (AD_SIZE[0] // 2 * len(poses), AD_SIZE[1] // 2))
    for i, im in enumerate(poses):
        strip.paste(im.resize((AD_SIZE[0] // 2, AD_SIZE[1] // 2)), (i * AD_SIZE[0] // 2, 0))
    del np
    return frames, anchor, strip


def make_frames(project: Path, ref: str, clip_path: str | Path, *, frames: int = 8, seeds: list[int] | None = None,
                profile: str = "lite", enforce_vram: bool = True, name: str | None = None,
                method: str = "auto") -> tuple[Path, list]:
    """A character item's accepted design drawn in ``frames`` poses of a clip (all in one sheet, so they stay
    alike). Writes ``<item>/frames/<name>/``: frames/, <name>.png (APNG), <name>.webp, <name>_sheet.png + .json,
    poses.png and report.json. Returns (folder, results best first)."""
    import shutil

    from ggs.export.animated import save_apng, save_webp
    from ggs.export.sheet import save_sprite_sheet
    from ggs.items import CharacterSettings, ItemError, design_dir, load
    from ggs.pipeline import progress
    from ggs.rig.clip import load_clip
    from ggs.rig.library import sidecar
    from ggs.rig.make import CHARACTER_PROMPT, VIEW_WORDS
    from ggs.rig.poses import rig_for

    project = Path(project)
    rec, d = load(project, ref)
    info_file = design_dir(d) / "design.json"
    if rec.accepted is None or not info_file.exists():
        raise ItemError(f"{rec.name!r} has no design yet: press 'Use this design' first")
    info = json.loads(info_file.read_text(encoding="utf-8"))
    drawing = info["drawing"]
    model = drawing.get("model") or "flux2-klein-4b"
    cs = CharacterSettings(**(drawing.get("character") or {}))
    clip_path = Path(clip_path)
    clip = load_clip(clip_path)
    name = name or clip_path.stem
    motion = name.replace("_", " ")
    side = sidecar(clip_path)
    if side.exists():
        try:
            motion = json.loads(side.read_text(encoding="utf-8")).get("name") or motion
        except ValueError:
            pass
    rig = rig_for("side")
    from ggs.items import character_guide

    _, guide, _ = character_guide(CharacterSettings(view="side", body_preset=cs.body_preset, body=cs.body,
                                                    pose="idle"), size=512)
    out = frames_dir(d) / name
    if out.exists():
        shutil.rmtree(out)
    work = out / "work"
    work.mkdir(parents=True)
    shutil.copy2(design_dir(d) / "design.png", work / "design.png")
    who = "; ".join(info.get("text") or [rec.description])
    prompt = CHARACTER_PROMPT.format(style="", description=who, body="", view=VIEW_WORDS["side"])
    spec = guide.spec.model_copy(update={"image": "design.png", "prompt": prompt, "seed": drawing["seed"],
                                         "model": model, "body": guide.spec.body or (cs.body or {})})
    if cs.body or cs.body_preset != "standard":
        from ggs.rig.body import PRESETS, BodyShape

        spec.body = (BodyShape(**cs.body) if cs.body else PRESETS[cs.body_preset]).model_dump()
    (work / "master.json").write_text(json.dumps(spec.model_dump(), indent=1), encoding="utf-8")
    method = frames_method() if method == "auto" else method
    if method == "animatediff" and model != "stub":
        progress.start("frames:animatediff", ["animate", "matte"])
        seed = (seeds or [drawing["seed"]])[0]
        fr, anchor, strip = _animatediff_frames(d, info, clip, min(frames, 16), seed, profile, enforce_vram)
        fps = len(fr) / clip.duration
        (out / "frames").mkdir()
        for k, f in enumerate(fr):
            f.save(out / "frames" / f"frame_{k:02d}.png")
        save_apng(fr, out / f"{name}.png", fps, True)
        save_webp(fr, out / f"{name}.webp", fps, True)
        save_sprite_sheet(fr, anchor, out / f"{name}_sheet.png", name, fps, True)
        strip.save(out / "poses.png")
        (out / "report.json").write_text(json.dumps({
            "format": "ggs.frames", "version": 1, "clip": str(clip_path), "frames": len(fr), "fps": round(fps, 3),
            "seed": seed, "passed": True, "anchor": list(anchor), "method": "animatediff", "checks": []},
            indent=1), encoding="utf-8")
        return out, []
    progress.start(f"frames:{model}", ["encode", "denoise", "matte"])
    results = make_sprites(work / "master.json", clip_path, rig, work, seeds=seeds or [drawing["seed"]],
                           model=model, profile=profile, settings=settings_for(frames),
                           style_ref=drawing.get("style"), style_weight=drawing.get("style_weight"),
                           project=project, motion=motion, enforce_vram=enforce_vram)
    best = results[0]
    fps = len(best.frames) / clip.duration
    (out / "frames").mkdir()
    for k, f in enumerate(best.frames):
        f.save(out / "frames" / f"frame_{k:02d}.png")
    save_apng(best.frames, out / f"{name}.png", fps, True)
    save_webp(best.frames, out / f"{name}.webp", fps, True)
    save_sprite_sheet(best.frames, best.anchor, out / f"{name}_sheet.png", name, fps, True)
    shutil.copy2(best.folder / "poses.png", out / "poses.png")
    shutil.copy2(best.folder / "sheet_raw.png", out / "sheet_raw.png")
    (out / "report.json").write_text(json.dumps({
        "format": "ggs.frames", "version": 1, "clip": str(clip_path), "frames": len(best.frames),
        "fps": round(fps, 3), "seed": best.seed, "passed": best.passed, "anchor": list(best.anchor),
        "checks": [{"index": c.index, "passed": c.passed, "why": c.why} for c in best.checks]}, indent=1),
        encoding="utf-8")
    return out, results


def pose_preview(clip_path: str | Path, frames: int, body: dict | None = None, body_preset: str = "standard"):
    """The stick-figure sheet the model is given for a clip (RGB), for showing before drawing."""
    from ggs.rig.body import PRESETS, BodyShape
    from ggs.rig.clip import load_clip
    from ggs.rig.make import guide_puppet
    from ggs.rig.poses import rig_for

    rig = rig_for("side")
    shape = BodyShape(**body) if body else PRESETS.get(body_preset, PRESETS["standard"])
    puppet, lengths = guide_puppet(rig, shape)
    sheet, _, _, _ = pose_sheet(rig, puppet, load_clip(clip_path), settings_for(frames), lengths)
    return sheet
