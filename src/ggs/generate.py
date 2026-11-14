"""Run a recipe: generate -> matte -> normalize (decontaminate, clean, trim) -> qa -> export.

Deterministic: same recipe, prompt, model files and seed give the same file. Heavy stages run one model at a
time (``ggs.pipeline.vram.Stages``). Assets that fail QA are not exported (they can be kept for inspection).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ggs import __version__
from ggs.pipeline.vram import Stages
from ggs.recipes import Recipe


@dataclass
class AssetResult:
    seed: int
    path: Path | None
    qa: dict
    passed: bool
    size: tuple[int, int] | None = None
    failures: list[str] = field(default_factory=list)


def build_backends(recipe: Recipe, model: str, profile: str):
    """(generator, matter) for ``model``: a pipeline from models/pipelines.yaml, a cloud model (Nano Banana,
    ``cloud:`` in the same file), or 'stub' (CPU, tests)."""
    if model == "stub":
        from ggs.pipeline.stub import StubGenerator, StubMatter

        return StubGenerator(recipe.background_rgb), StubMatter(recipe.background_rgb)
    from ggs.pipeline.matting import BiRefNetMatter
    from ggs.pipeline.models import load_pipelines, resolve_matting, resolve_pipeline

    spec_file = load_pipelines()
    if model in spec_file.cloud:  # drawn on Google's servers; matting still runs here (real alpha)
        from ggs.pipeline.cloud import api_key, make_cloud_generator

        api_key()  # fail before anything loads when there is no key
        return make_cloud_generator(spec_file.cloud[model]), BiRefNetMatter(
            resolve_matting(recipe.matting, spec_file=spec_file))
    files = resolve_pipeline(model, profile, spec_file=spec_file)
    from ggs.pipeline.backends import make_generator

    gen = make_generator(spec_file.pipelines[model], files)
    return gen, BiRefNetMatter(resolve_matting(recipe.matting, spec_file=spec_file))


def choose_model(recipe: Recipe, model: str | None = None, project: Path | None = None) -> str:
    """Explicit ``model`` > the project's chosen base model > the recipe's default."""
    if model:
        return model
    if project is not None:
        from ggs.style.package import load_project

        chosen = load_project(project).base_model
        if chosen:
            return chosen
    return recipe.base_model


def real_rgba(rgb_img, alpha_img, alpha_low: float = 0.04, alpha_high: float = 0.96, background=None):
    """Generator render + matte -> RGBA on the full canvas (no trim): known-background refinement and
    decontamination. Returns (rgba uint8 array, background colour, background spread). ``background`` (RGB)
    skips estimating it from the border, for art that touches the frame edges (background layers)."""
    import numpy as np

    from ggs.pipeline.post import (
        clean_alpha,
        decontaminate,
        estimate_background,
        refine_alpha_known_background,
        to_rgba,
    )

    rgb = np.asarray(rgb_img.convert("RGB"))
    alpha = np.asarray(alpha_img.convert("L"), dtype=np.float32) / 255.0
    if background is None:
        bg, spread = estimate_background(rgb)
    else:
        bg, spread = np.array(background, dtype=np.float32), 0.0
    a = clean_alpha(alpha, alpha_low, alpha_high)
    a = refine_alpha_known_background(rgb, a, bg)
    a = clean_alpha(a, alpha_low, 1.0)
    return to_rgba(decontaminate(rgb, a, bg), a), bg, spread


def process(rgb_img, alpha_img, recipe: Recipe):
    """Normalize + QA one image. Returns (trimmed RGBA array or None, QaResult, background)."""
    from ggs.pipeline.post import pad_to_pot, trim
    from ggs.qa.sprite import check_sprite

    rgba, bg, spread = real_rgba(rgb_img, alpha_img, recipe.post.alpha_low, recipe.post.alpha_high)
    qa = check_sprite(rgba, bg, spread, recipe.qa)
    if not rgba[..., 3].any():
        return None, qa, bg
    out, _ = trim(rgba, recipe.post.padding)
    if recipe.post.power_of_two:
        out = pad_to_pot(out)
    return out, qa, bg


def run_recipe(recipe: Recipe, description: str, out: Path, *, model: str | None = None, seeds: list[int],
               profile: str = "lite", style: str = "", keep_failed: bool = False, enforce_vram: bool = True,
               size: tuple[int, int] | None = None, debug: bool = False, style_ref: str | None = None,
               style_weight: float | None = None, project: Path | None = None, reference: list | None = None,
               prompt_is_final: bool = False) -> tuple[list[AssetResult], dict]:
    """``style`` = extra style words for the prompt; ``style_ref`` = a trained style (installed name or
    .ggsstyle path) whose LoRA is loaded and whose trigger word starts the prompt."""
    from PIL import Image

    model = choose_model(recipe, model, project)
    width, height = size or recipe.size
    lora = style_meta = None
    if style_ref:
        from ggs.style.package import StyleError, resolve_style

        style_meta, lora = resolve_style(style_ref, project)
        if model != "stub" and style_meta.base_model != model:
            raise StyleError(f"style {style_meta.name!r} was trained on {style_meta.base_model!r}; it cannot load "
                             f"on {model!r}")
        style = ", ".join(s for s in (style_meta.trigger, style) if s)
        if style_weight is None:
            style_weight = style_meta.recommended_weight
    if recipe.kind != "sprite":  # backgrounds, tiles, UI panels, VFX: ggs.assets
        from ggs.assets import run_asset

        return run_asset(recipe, description, Path(out), model=model, seeds=seeds, profile=profile, style=style,
                         keep_failed=keep_failed, enforce_vram=enforce_vram, size=size or recipe.size, debug=debug,
                         lora=lora, lora_scale=1.0 if style_weight is None else style_weight,
                         style_meta=style_meta)
    # an edit instruction (``prompt_is_final``) goes to the model as written, with the image as reference
    prompt = (f"{style}, {description}" if style else description) if prompt_is_final else recipe.prompt(
        description, style, model)
    stages = Stages(profile=profile, enforce=enforce_vram)
    t0 = time.perf_counter()
    gen, matter = build_backends(recipe, model, profile)
    extra = {"reference": reference} if reference else {}
    images = gen.generate(prompt, seeds, width, height, stages, lora=lora,
                          lora_scale=1.0 if style_weight is None else style_weight, **extra)
    alphas = matter.alphas(images, stages)
    from ggs.pipeline.cloud import cost as cloud_cost

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: list[AssetResult] = []
    for i, (seed, img, alpha) in enumerate(zip(seeds, images, alphas)):
        arr, qa, _ = process(img, alpha, recipe)
        name = out if len(seeds) == 1 else out.with_name(f"{out.stem}_{i + 1:02d}{out.suffix}")
        if debug:  # raw render + matte, to re-run post-processing without the GPU
            img.save(name.with_name(f"{name.stem}_raw.png"))
            alpha.save(name.with_name(f"{name.stem}_alpha.png"))
        path = None
        if arr is not None and (qa.passed or keep_failed):
            path = name if qa.passed else name.with_name(f"{name.stem}_QA_FAILED{name.suffix}")
            Image.fromarray(arr, "RGBA").save(path)
        results.append(AssetResult(seed, path, qa.to_dict(), qa.passed,
                                   None if arr is None else (arr.shape[1], arr.shape[0]), qa.failures))
    meta = {
        "generator": f"ggs {__version__}",
        "class": recipe.id,
        "model": model,
        "matting": getattr(matter, "name", recipe.matting),
        "profile": profile,
        "prompt": prompt,
        "description": description,
        "style": style,
        "style_package": None if style_meta is None else {"name": style_meta.name, "trigger": style_meta.trigger,
                                                          "weight": style_weight},
        "size": [width, height],
        "seconds": round(time.perf_counter() - t0, 2),
        "stages": stages.summary(),
        "cloud": cloud_cost(gen, width, height),
        "assets": [{"seed": r.seed, "file": r.path.name if r.path else None, "passed": r.passed,
                    "size": r.size, "failures": r.failures, "qa": r.qa} for r in results],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return results, meta
