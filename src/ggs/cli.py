"""Command line interface. Deterministic: no LLM interprets requests here."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ggs import __version__
from ggs.hardware import detect_vram_gb, profile_for
from ggs.models import load_registry
from ggs.paths import rigs_dir
from ggs.rig import load_rig_spec

DEFAULT_RIG = rigs_dir() / "humanoid_side_v1.yaml"

PROFILE_HELP = {
    "standard": "Full local workflow (generation and style-LoRA training).",
    "lite": "Quantized inference; train at 512px only, or export the dataset for cloud training.",
    "minimal": "Not enough VRAM for generation. Use a cloud runner.",
    "no-gpu": "No NVIDIA GPU found. Generation is unsupported.",
}


def cmd_doctor(_: argparse.Namespace) -> int:
    vram = detect_vram_gb()
    profile = profile_for(vram)
    print(f"ggs {__version__}")
    print(f"VRAM: {'unknown' if vram is None else f'{vram:.1f} GB'}")
    print(f"Profile: {profile} - {PROFILE_HELP[profile]}")
    return 0


def cmd_rig_validate(args: argparse.Namespace) -> int:
    try:
        spec = load_rig_spec(args.path)
    except ValueError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    sheet = spec.part_sheet
    print(
        f"OK {spec.id}: {len(spec.bones)} bones, {len(sheet.parts)} unique parts, "
        f"{len(sheet.head_variants)} head variants, {len(spec.sockets)} sockets, "
        f"sheet {sheet.cols}x{sheet.rows} cells of {sheet.cell_px}px"
    )
    return 0


def cmd_models_list(args: argparse.Namespace) -> int:
    registry = load_registry()
    for e in registry.values():
        if args.commercial_only and not e.bundle_eligible:
            continue
        vram = "" if e.min_vram_gb is None else f" >= {e.min_vram_gb:g}GB"
        print(f"{e.id:28} {e.kind:8} commercial={e.commercial:6} {e.license}{vram}")
    return 0


def cmd_models_dirs(_: argparse.Namespace) -> int:
    from ggs.pipeline.models import config_path, model_dirs

    print(f"config: {config_path()}")
    for d in model_dirs():
        print(f"{'ok     ' if d.is_dir() else 'missing'} {d}")
    return 0


def cmd_models_add_dir(args: argparse.Namespace) -> int:
    from ggs.pipeline.models import load_user_config, save_user_config

    d = Path(args.path).expanduser()
    if not d.is_dir():
        print(f"ERROR: {d} is not a folder", file=sys.stderr)
        return 1
    cfg = load_user_config()
    if str(d) not in cfg.model_dirs:
        cfg.model_dirs.append(str(d))
    print(f"added {d} -> {save_user_config(cfg)}")
    return 0


def cmd_models_check(args: argparse.Namespace) -> int:
    from ggs.pipeline.models import (
        MissingModelError,
        load_pipelines,
        resolve_matting,
        resolve_pipeline,
    )

    profile = args.profile or profile_for(detect_vram_gb())
    spec_file = load_pipelines()
    status = 0
    for name in spec_file.pipelines:
        try:
            files = resolve_pipeline(name, profile, spec_file=spec_file)
            print(f"{name} ({profile}): OK")
            for k, p in files.items():
                print(f"  {k:13} {p}")
        except (MissingModelError, ValueError) as exc:
            print(f"{name} ({profile}): {exc}")
            status = 1
    for name in spec_file.matting:
        try:
            print(f"matting {name}: OK  {resolve_matting(name, spec_file=spec_file)}")
        except MissingModelError as exc:
            print(f"matting {name}: {exc}")
            status = 1
    return status


DEFAULT_BASE = "flux2-klein-4b"


def cmd_generate(args: argparse.Namespace) -> int:
    from ggs.generate import run_recipe
    from ggs.pipeline.models import MissingModelError
    from ggs.pipeline.vram import VramBudgetError
    from ggs.recipes import load_recipe

    try:
        recipe = load_recipe(args.asset_class)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    profile = args.profile or profile_for(detect_vram_gb())
    try:
        from ggs.generate import choose_model

        model = choose_model(recipe, args.model, Path(args.project) if args.project else None)
    except ValueError as exc:  # StyleError: not a project / bad project.yaml
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if model != "stub" and profile not in ("standard", "lite"):
        print(f"ERROR: profile {profile!r}: {PROFILE_HELP.get(profile, '')}", file=sys.stderr)
        return 1
    if model != "stub" and recipe.min_profile == "standard" and profile != "standard":
        print(f"ERROR: class {recipe.id!r} needs the standard profile", file=sys.stderr)
        return 1
    size = None
    if args.size:
        try:
            w, h = (int(v) for v in args.size.lower().split("x"))
            size = (w, h)
        except ValueError:
            print("ERROR: --size must look like 1024x1024", file=sys.stderr)
            return 1
    seeds = [args.seed + i for i in range(args.count)]
    try:
        results, meta = run_recipe(recipe, args.prompt, Path(args.out), model=model, seeds=seeds,
                                   profile=profile, style=args.style_words, keep_failed=args.keep_failed,
                                   enforce_vram=not args.no_vram_check, size=size, debug=args.debug,
                                   style_ref=args.style, style_weight=args.style_weight,
                                   project=Path(args.project) if args.project else None)
    except (MissingModelError, VramBudgetError, ValueError, OSError) as exc:  # StyleError is a ValueError
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f'ERROR: {exc}. Install the GPU stack: see README ("gpu" extra).', file=sys.stderr)
        return 1
    for s in meta["stages"]:
        peak = "" if s["peak_gb"] is None else f", peak {s['peak_gb']:.2f} GB VRAM"
        print(f"stage {s['name']:8} {s['seconds']:6.1f} s{peak}")
    ok = 0
    for r in results:
        if r.passed:
            ok += 1
            print(f"seed {r.seed}: OK  {r.path}  {r.size[0]}x{r.size[1]}")
        else:
            where = f" (kept as {r.path})" if r.path else ""
            print(f"seed {r.seed}: QA FAILED{where}: " + "; ".join(r.failures))
    print(f"{ok}/{len(results)} passed QA; details in {Path(args.out).with_suffix('.json')}")
    return 0 if ok == len(results) else 2


def cmd_project_init(args: argparse.Namespace) -> int:
    from ggs.pipeline.models import load_pipelines
    from ggs.style.package import StyleError, init_project

    known = sorted(load_pipelines().pipelines)
    if args.model and args.model not in known:
        print(f"ERROR: unknown base model {args.model!r}; known: {known}", file=sys.stderr)
        return 1
    try:
        print(f"created {init_project(args.path, args.name, args.model)}")
    except StyleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_style_dataset_import(args: argparse.Namespace) -> int:
    from ggs.style.dataset import RIGHTS_WARNING, RightsNotConfirmed, import_dataset

    print(f"NOTE: {RIGHTS_WARNING}")
    try:
        m = import_dataset(args.src, args.out, rights_confirmed=args.i_have_the_rights, size=args.size)
    except RightsNotConfirmed as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    n = len(m.images)
    tip = "" if 20 <= n <= 40 else " (20-40 images recommended)"
    print(f"imported {n} images into {args.out}{tip}. Next: ggs style caption {args.out} --trigger <WORD>")
    return 0


def cmd_style_caption(args: argparse.Namespace) -> int:
    from PIL import Image

    from ggs.pipeline.models import MissingModelError, resolve_captioner
    from ggs.pipeline.vram import Stages
    from ggs.style.caption import QwenVLCaptioner, with_trigger
    from ggs.style.dataset import load_dataset, save_manifest

    try:
        manifest, items = load_dataset(args.dataset)
        todo = [(p, c) for p, c in items if args.overwrite or c is None]
        if todo:
            captioner = QwenVLCaptioner(resolve_captioner(args.captioner))
            stages = Stages(profile=args.profile or profile_for(detect_vram_gb()))
            caps = captioner.caption([Image.open(p) for p, _ in todo], stages)
            for (p, _), cap in zip(todo, caps):
                p.with_suffix(".txt").write_text(with_trigger(args.trigger, cap) + "\n", encoding="utf-8")
        for p, c in items:  # make sure every caption carries the trigger
            if c is not None and (p, c) not in todo:
                p.with_suffix(".txt").write_text(with_trigger(args.trigger, c) + "\n", encoding="utf-8")
        manifest.trigger = args.trigger
        save_manifest(args.dataset, manifest)
    except (MissingModelError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"captioned {len(todo)} images ({len(items) - len(todo)} kept). Edit the .txt files in "
          f"{Path(args.dataset) / 'images'} if needed, then: ggs style train {args.dataset} --name <name> --out <file>")
    return 0


def cmd_style_train(args: argparse.Namespace) -> int:
    from PIL import Image

    from ggs.pipeline.models import MissingModelError, load_pipelines, resolve_pipeline
    from ggs.pipeline.vram import Stages, VramBudgetError
    from ggs.style.dataset import load_dataset
    from ggs.style.package import StyleError, StyleMeta, load_project, write_package
    from ggs.style.train import TrainSettings, train_style

    profile = args.profile or profile_for(detect_vram_gb())
    if profile not in ("standard", "lite"):
        print(f"ERROR: profile {profile!r} cannot train", file=sys.stderr)
        return 1
    spec_file = load_pipelines()
    try:
        model = args.model or (load_project(args.project).base_model if args.project else None) or DEFAULT_BASE
    except StyleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    spec = spec_file.pipelines.get(model)
    if spec is None or spec.training is None:
        trainable = sorted(k for k, v in spec_file.pipelines.items() if v.training)
        print(f"ERROR: {model!r} cannot train styles; choose one of {trainable}", file=sys.stderr)
        return 1
    t = spec.training
    resolution = args.resolution or t.resolution
    max_res = t.max_resolution.get(profile, t.resolution)
    if resolution > max_res:
        print(f"ERROR: {resolution}px {model} training needs more VRAM than the {profile} profile has "
              f"(max {max_res}px). Use a lower --resolution, or export the dataset for cloud training.",
              file=sys.stderr)
        return 1
    if profile == "lite" and resolution > t.resolution:
        print(f"WARNING: {resolution}px on the lite profile is slow and close to the VRAM limit")
    steps = args.steps or t.steps
    try:
        manifest, items = load_dataset(args.dataset)
        if not manifest.trigger:
            raise ValueError("dataset has no trigger word yet; run `ggs style caption` first")
        missing = [p.name for p, c in items if not c]
        if missing:
            raise ValueError(f"images without captions: {missing[:5]}; run `ggs style caption`")
        StyleMeta.model_validate({"name": args.name, "trigger": "x", "base_model": "x", "rank": 1, "alpha": 1,
                                  "steps": 1, "resolution": 512, "learning_rate": 1, "images": 1})
        files = resolve_pipeline(model, profile, spec_file=spec_file)
        settings = TrainSettings(steps=steps, rank=args.rank or t.rank, lr=args.lr or t.lr, resolution=resolution,
                                 seed=args.seed, checkpoint_every=args.checkpoint_every,
                                 text_tokens=args.text_tokens)
        work = Path(args.work_dir) if args.work_dir else Path(args.out).with_suffix("").with_name(
            Path(args.out).stem + "_work")
        stages = Stages(profile=profile)
        every = max(1, steps // 30)
        print(f"training {args.name!r} on {model} ({profile}): {steps} steps at {resolution}px", flush=True)
        window: list[float] = []

        def progress(step: int, loss: float, elapsed: float) -> None:
            window.append(loss)
            if step % every == 0 or step == steps:
                eta = elapsed / step * (steps - step)
                print(f"step {step:5d}/{steps}  loss {sum(window) / len(window):.4f}  "
                      f"{elapsed / step:.2f} s/step  eta {eta / 60:.1f} min", flush=True)
                window.clear()

        images = [Image.open(p) for p, _ in items]
        res = train_style(spec.loader, images, [c for _, c in items], files, settings, work, stages,
                          manifest.trigger, progress, profile=profile, defaults=spec.defaults)
    except (MissingModelError, VramBudgetError, ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    tile = settings.resolution // 2
    sheet = Image.new("RGB", (tile * len(res.previews), tile), "white")
    for i, im in enumerate(res.previews):
        sheet.paste(im.resize((tile, tile)), (i * tile, 0))
    sheet.save(work / "preview.png")
    meta = StyleMeta(name=args.name, trigger=manifest.trigger, base_model=model, trained_on=res.trained_on,
                     rank=settings.rank, alpha=settings.alpha or settings.rank, steps=settings.steps,
                     resolution=settings.resolution, learning_rate=settings.lr, images=len(items))
    pkg = write_package(args.out, meta, res.lora, sheet, [c for _, c in items])
    for s in stages.summary():
        peak = "" if s["peak_gb"] is None else f", peak {s['peak_gb']:.2f} GB VRAM"
        print(f"stage {s['name']:8} {s['seconds'] / 60:6.1f} min{peak}")
    print(f"wrote {pkg} (preview: {work / 'preview.png'}). Install with: ggs style install {pkg} --project <dir>")
    return 0


def cmd_style_install(args: argparse.Namespace) -> int:
    from ggs.paths import shared_dir
    from ggs.style.package import StyleError, install, project_dir, require_project

    try:
        target = (project_dir(require_project(args.project)) / "styles") if args.project else (
            shared_dir() / "styles")
        dest = install(args.package, target, overwrite=args.overwrite)
    except (StyleError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"installed {dest.name} -> {dest}")
    return 0


def cmd_style_list(args: argparse.Namespace) -> int:
    from ggs.style.package import StyleError, list_styles

    try:
        styles = list_styles(args.project)
    except StyleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for meta, folder in styles:
        print(f"{meta.name:20} trigger={meta.trigger:14} base={meta.base_model:16} {meta.images} images, "
              f"{meta.steps} steps @ {meta.resolution}px  ({folder})")
    if not styles:
        print("no styles installed")
    return 0


def cmd_mocap_inspect(args: argparse.Namespace) -> int:
    from ggs.rig.bvh import BvhError, load_bvh

    try:
        motion = load_bvh(args.path)
        spec = load_rig_spec(args.rig)
    except (BvhError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{args.path}: {motion.n_frames} frames at {motion.fps:.2f} fps "
          f"({motion.n_frames * motion.frame_time:.2f} s), {len(motion.joints)} joints")
    print("joints: " + ", ".join(motion.joint_names))
    have = set(motion.joint_names)
    for name, mapping in spec.mocap_profiles.items():
        if not mapping:
            continue
        missing = sorted({j for pair in mapping.values() for j in pair} - have)
        status = "OK" if not missing else f"missing {missing}"
        print(f"profile {name}: {status}")
    return 0


def cmd_mocap_retarget(args: argparse.Namespace) -> int:
    from ggs.rig.bvh import BvhError, load_bvh
    from ggs.rig.retarget import RetargetError, retarget

    try:
        motion = load_bvh(args.path)
        spec = load_rig_spec(args.rig)
        clip = retarget(
            motion, spec, args.profile, up_axis=args.up, fps=args.fps,
            start=args.start, end=args.end, in_place=args.in_place,
            drop_tpose_frame=not args.keep_tpose_frame,
        )
    except (BvhError, RetargetError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    clip.save(args.out)
    print(f"wrote {args.out}: {clip.frame_count} frames at {clip.fps:g} fps ({clip.duration:.2f} s)")
    for w in clip.warnings:
        print(f"warning: {w}")
    return 0


def cmd_mocap_preview(args: argparse.Namespace) -> int:
    from ggs.rig.clip import load_clip
    from ggs.rig.preview import render_preview

    try:
        clip = load_clip(args.clip)
        spec = load_rig_spec(args.rig)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if clip.rig != spec.id:
        print(f"ERROR: clip is for rig {clip.rig!r}, not {spec.id!r}", file=sys.stderr)
        return 1
    try:
        out = render_preview(spec, clip, args.out, frames=args.frames, cell=args.cell)
    except ImportError:
        print('ERROR: preview needs Pillow: pip install -e ".[image]"', file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


def cmd_character_programmer_art(args: argparse.Namespace) -> int:
    from ggs.rig.character import make_programmer_art

    try:
        spec = load_rig_spec(args.rig)
        from ggs.rig.body import parse_body

        path = make_programmer_art(spec, args.out, px_per_height=args.px_per_height, name=args.name,
                                   body=parse_body(args.body) if args.body else None)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError:
        print('ERROR: needs Pillow: pip install -e ".[image]"', file=sys.stderr)
        return 1
    print(f"wrote {path} and {path.parent / 'parts.png'}")
    return 0


def _print_cut(label: str, outcome) -> None:
    r = outcome.report
    status = "OK " if r.passed else "REJECTED"
    print(f"{label}: {status} score {r.score:.3f}, {outcome.px_per_height:.0f} px/height -> {outcome.character}")
    for f in r.failures():
        print(f"    {f}")


def cmd_character_cut(args: argparse.Namespace) -> int:
    from ggs.rig.cutter import CutError
    from ggs.rig.make import cut_master

    try:
        outcome = cut_master(args.master, load_rig_spec(args.rig), args.out, args.name, stature=args.stature)
    except (CutError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_cut(args.name, outcome)
    print(f"report: {Path(args.out) / 'validation.json'} (picture: validation.png)")
    return 0 if outcome.report.passed else 2


def _print_refine(reports) -> None:
    for rep in reports:
        parts = ", ".join(f"{e.get('part') or e.get('variant')}{'' if e['applied'] else ' (kept)'}" for e in rep.edits)
        print(f"  refined {rep.char_dir}: {rep.applied}/{len(rep.edits)} edits applied [{parts}]")


def cmd_character_refine(args: argparse.Namespace) -> int:
    from ggs.pipeline.models import MissingModelError
    from ggs.pipeline.vram import VramBudgetError
    from ggs.rig.refine import RefineSettings, refine_characters

    dirs = [Path(c).parent if Path(c).suffix else Path(c) for c in args.characters]
    missing = [str(d) for d in dirs if not (d / "parts.png").exists()]
    if missing:
        print(f"ERROR: no parts.png in {missing}", file=sys.stderr)
        return 1
    profile = args.profile or profile_for(detect_vram_gb())
    try:
        reports = refine_characters(dirs, load_rig_spec(args.rig), model=args.model, profile=profile,
                                    settings=RefineSettings(fills=not args.no_fills,
                                                            expressions=not args.no_expressions, seed=args.seed),
                                    enforce_vram=not args.no_vram_check)
    except (MissingModelError, VramBudgetError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_refine(reports)
    return 0


def cmd_character_master(args: argparse.Namespace) -> int:
    from ggs.pipeline.models import MissingModelError
    from ggs.pipeline.vram import VramBudgetError
    from ggs.rig.body import parse_body
    from ggs.rig.cutter import CutError
    from ggs.rig.make import cut_master, guide_only_master, make_masters

    rig = load_rig_spec(args.rig)
    out = Path(args.out)
    try:
        body = parse_body(args.body)
        if args.guide_only:
            masters = [guide_only_master(rig, out / "guide_only", body=body, pose=args.pose)]
        else:
            if not args.prompt:
                print("ERROR: --prompt is required (or use --guide-only)", file=sys.stderr)
                return 1
            from ggs.generate import choose_model
            from ggs.recipes import load_recipe

            model = choose_model(load_recipe("sprite"), args.model, Path(args.project) if args.project else None)
            profile = args.profile or profile_for(detect_vram_gb())
            masters = make_masters(rig, args.prompt, out, seeds=[args.seed + i for i in range(args.count)],
                                   model=model, profile=profile, style_ref=args.style, style_weight=args.style_weight,
                                   project=Path(args.project) if args.project else None,
                                   enforce_vram=not args.no_vram_check, body=body, pose=args.pose)
        if args.no_cut:
            for m in masters:
                print(f"wrote {m}")
            return 0
        ok = 0
        valid = []
        for m in masters:
            name = f"{args.name}_{m.parent.name}" if len(masters) > 1 else args.name
            outcome = cut_master(m, rig, m.parent / "character", name, stature=args.stature)
            _print_cut(m.parent.name, outcome)
            ok += outcome.report.passed
            if outcome.report.passed:
                valid.append(m.parent / "character")
        if valid and not args.no_refine and not args.guide_only:
            from ggs.rig.refine import refine_characters

            _print_refine(refine_characters(valid, rig, model=model, profile=profile,
                                            enforce_vram=not args.no_vram_check))
    except (MissingModelError, VramBudgetError, CutError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f'ERROR: {exc}. Install the GPU stack: see README ("gpu" extra).', file=sys.stderr)
        return 1
    print(f"{ok}/{len(masters)} masters gave a valid part sheet (each seed folder has character/validation.png)")
    return 0 if ok else 2


def cmd_adapter_pairs(args: argparse.Namespace) -> int:
    from ggs.pipeline.models import MissingModelError
    from ggs.pipeline.vram import VramBudgetError
    from ggs.rig.factory import character_prompts, make_pairs

    profile = args.profile or profile_for(detect_vram_gb())
    try:
        summary = make_pairs(load_rig_spec(args.rig), args.out, character_prompts(args.count, args.start),
                             [args.seed + i for i in range(args.seeds)], model=args.model, profile=profile,
                             enforce_vram=not args.no_vram_check)
    except (MissingModelError, VramBudgetError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{summary['kept']} pairs kept, {summary['rejected']} sheets rejected -> {Path(args.out) / 'pairs'}")
    return 0 if summary["kept"] else 2


def _split(pairs: list, every: int) -> tuple[list, list]:
    """Deterministic hold-out: every ``every``-th pair (by order) is kept out of training."""
    hold = [p for i, p in enumerate(pairs) if every and i % every == every - 1]
    return [p for p in pairs if p not in hold], hold


def cmd_adapter_train(args: argparse.Namespace) -> int:
    from PIL import Image

    from ggs.pipeline.models import MissingModelError, load_pipelines, resolve_pipeline
    from ggs.pipeline.vram import Stages, VramBudgetError
    from ggs.rig.adapter import ADAPTER_PROMPT, AdapterMeta, write_adapter
    from ggs.rig.factory import SHEET_PX_PER_HEIGHT, load_pairs
    from ggs.rig.make import on_white
    from ggs.style.train import TrainSettings
    from ggs.style.train_edit import train_edit_lora

    rig = load_rig_spec(args.rig)
    profile = args.profile or profile_for(detect_vram_gb())
    pairs = load_pairs(args.pairs)
    train, hold = _split(pairs, args.holdout_every)
    if len(train) < 4:
        print(f"ERROR: only {len(train)} training pairs in {args.pairs}; make more with `ggs adapter pairs`",
              file=sys.stderr)
        return 1
    ps = rig.part_sheet
    tw = args.resolution
    th = round(tw * ps.rows / ps.cols / 16) * 16
    ref = (args.resolution, args.resolution)
    work = Path(args.out) / "work"
    every = max(1, args.steps // 30)
    window: list[float] = []

    def progress(step: int, loss: float, elapsed: float) -> None:
        window.append(loss)
        if step % every == 0 or step == args.steps:
            print(f"step {step:5d}/{args.steps}  loss {sum(window) / len(window):.4f}  {elapsed / step:.2f} s/step  "
                  f"eta {elapsed / step * (args.steps - step) / 60:.1f} min", flush=True)
            window.clear()

    try:
        files = resolve_pipeline(args.model, profile, spec_file=load_pipelines())
        refs = [on_white(Image.open(p / "master.png")) for p in train]
        targets = [on_white(Image.open(p / "parts.png")) for p in train]
        previews = [on_white(Image.open(p / "master.png")).resize(ref, Image.LANCZOS) for p in hold[:2]]
        settings = TrainSettings(steps=args.steps, rank=args.rank, lr=args.lr, resolution=args.resolution,
                                 seed=args.seed, checkpoint_every=args.checkpoint_every)
        print(f"training rig-sheet adapter on {len(train)} pairs ({len(hold)} held out), master {ref[0]}px -> sheet "
              f"{tw}x{th}", flush=True)
        stages = Stages(profile=profile)
        res = train_edit_lora(refs, targets, ADAPTER_PROMPT, files, settings, work, stages, ref, (tw, th),
                              progress, previews)
    except (MissingModelError, VramBudgetError, ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    preview = None
    if res.previews:
        preview = Image.new("RGB", (tw * len(res.previews), th), "white")
        for i, im in enumerate(res.previews):
            preview.paste(im, (i * tw, 0))
    meta = AdapterMeta(name=Path(args.out).name, rig=rig.id, base_model=args.model, ref_size=ref,
                       target_size=(tw, th), sheet_px_per_height=SHEET_PX_PER_HEIGHT, pairs=len(train),
                       steps=args.steps, rank=args.rank, learning_rate=args.lr, seconds=round(res.seconds, 1),
                       holdout=[p.name for p in hold])
    out = write_adapter(args.out, meta, res.lora, preview)
    for s in stages.summary():
        peak = "" if s["peak_gb"] is None else f", peak {s['peak_gb']:.2f} GB VRAM"
        print(f"stage {s['name']:8} {s['seconds'] / 60:6.1f} min{peak}")
    print(f"wrote adapter {out}")
    return 0


def cmd_adapter_eval(args: argparse.Namespace) -> int:
    import json

    import numpy as np
    from PIL import Image

    from ggs.pipeline.models import MissingModelError
    from ggs.pipeline.vram import VramBudgetError
    from ggs.rig.adapter import adapter_character, load_adapter, pose_fidelity, sheets_from_masters
    from ggs.rig.character import load_character
    from ggs.rig.factory import load_pairs
    from ggs.rig.make import cut_master

    rig = load_rig_spec(args.rig)
    profile = args.profile or profile_for(detect_vram_gb())
    try:
        meta, _ = load_adapter(args.adapter)
        pairs = load_pairs(args.pairs)
        if not args.all:
            pairs = [p for p in pairs if p.name in set(meta.holdout)]
        if args.limit:
            pairs = pairs[:args.limit]
        if not pairs:
            raise ValueError("no held-out pairs to evaluate (use --all)")
        masters = [np.asarray(Image.open(p / "master.png").convert("RGBA")) for p in pairs]
        size = None if not args.size else tuple(int(v) for v in args.size.lower().split("x"))
        sheets = sheets_from_masters(masters, args.adapter, profile=profile, size=size)
    except (MissingModelError, VramBudgetError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out)
    rows = []
    for p, sheet in zip(pairs, sheets):
        truth = load_character(p / "character.yaml")
        cut = cut_master(p / "master.json", rig, out / p.name / "cutter", f"cut_{p.name}")
        ada = adapter_character(p / "master.json", sheet, rig, out / p.name / "adapter", f"adapter_{p.name}")
        row = {"pair": p.name, "who": json.loads((p / "meta.json").read_text(encoding="utf-8")).get("who", "")}
        for label, o in (("cutter", cut), ("adapter", ada)):
            fid = pose_fidelity(load_character(o.character), truth, rig) if o.character else {
                "pose_iou": 0.0, "pose_colour_error": 1.0}
            row[label] = {"score": round(o.report.score, 4), "passed": o.report.passed, **fid}
        rows.append(row)
        print(f"{p.name} {row['who'][:28]:28}  cutter: valid={row['cutter']['passed']!s:5} pose IoU "
              f"{row['cutter']['pose_iou']:.3f}  |  adapter: valid={row['adapter']['passed']!s:5} pose IoU "
              f"{row['adapter']['pose_iou']:.3f}")
    mean = {k: {m: round(float(np.mean([r[k][m] for r in rows])), 4)
                for m in ("score", "pose_iou", "pose_colour_error")} for k in ("cutter", "adapter")}
    for k in ("cutter", "adapter"):
        mean[k]["valid"] = sum(r[k]["passed"] for r in rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps({"mean": mean, "pairs": rows}, indent=1), encoding="utf-8")
    for k in ("cutter", "adapter"):
        m = mean[k]
        print(f"{k:8} valid {m['valid']}/{len(rows)}  reassembly {m['score']:.3f}  pose IoU {m['pose_iou']:.3f}  "
              f"pose colour error {m['pose_colour_error']:.3f}")
    better = mean["adapter"]["pose_iou"] > mean["cutter"]["pose_iou"]
    print(f"adapter {'beats' if better else 'does not beat'} the cutter on pose IoU against the ground truth "
          f"({out / 'eval.json'})")
    return 0


def cmd_character_sprites(args: argparse.Namespace) -> int:
    import json

    from ggs.export.animated import save_apng, save_webp
    from ggs.export.sheet import save_sprite_sheet
    from ggs.pipeline.models import MissingModelError
    from ggs.pipeline.vram import VramBudgetError
    from ggs.rig.clip import load_clip
    from ggs.rig.sprites import SpriteSettings, make_sprites

    rig = load_rig_spec(args.rig)
    profile = args.profile or profile_for(detect_vram_gb())
    out = Path(args.out)
    try:
        s = SpriteSettings(frames=args.frames, cols=args.cols, rows=args.rows)
        results = make_sprites(args.master, args.clip, rig, out, seeds=[args.seed + i for i in range(args.count)],
                               model=args.model, profile=profile, settings=s, style_ref=args.style,
                               style_weight=args.style_weight, project=Path(args.project) if args.project else None,
                               motion=args.motion, enforce_vram=not args.no_vram_check)
        clip = load_clip(args.clip)
    except (MissingModelError, VramBudgetError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for r in sorted(results, key=lambda r: r.seed):
        bad = [f"frame {c.index}: {'; '.join(c.why)}" for c in r.checks if not c.passed]
        print(f"seed {r.seed}: {r.passed}/{len(r.checks)} frames pass" + (" -- " + " | ".join(bad) if bad else ""))
    best = results[0]
    fps = len(best.frames) / clip.duration
    name = args.name
    suffix = "" if best.passed == len(best.checks) else "_QA_FAILED"
    apng = save_apng(best.frames, out / f"{name}{suffix}.png", fps, True)
    webp = save_webp(best.frames, out / f"{name}{suffix}.webp", fps, True)
    sheet, atlas = save_sprite_sheet(best.frames, best.anchor, out / f"{name}{suffix}_sheet.png", name, fps, True)
    (out / "best.json").write_text(json.dumps({"seed": best.seed, "frames_passed": best.passed,
                                               "frames": len(best.checks), "fps": round(fps, 3),
                                               "anchor": list(best.anchor)}, indent=1), encoding="utf-8")
    print(f"best seed {best.seed} ({best.passed}/{len(best.checks)} frames pass): {apng}, {webp}, {sheet} + {atlas}")
    return 0 if best.passed == len(best.checks) else 2


def cmd_item_make(args: argparse.Namespace) -> int:
    from ggs.rig.equipment import load_item, make_item

    try:
        path = make_item(args.image, args.out, name=args.name, length=args.length, hold=args.hold,
                         grip_at=args.grip_at, hold_angle=args.hold_angle)
        item = load_item(path)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    s = item.spec
    print(f"wrote {path}: {s.hold}, grip {s.grip}" + (f", second grip {s.grip2}" if s.grip2 else "")
          + f", {s.px_per_height:.0f} px per standing height")
    return 0


def cmd_clip_keys(args: argparse.Namespace) -> int:
    from ggs.rig.character import load_character
    from ggs.rig.layers import keys_to_clip, load_keys

    try:
        rig = load_rig_spec(args.rig)
        lengths = load_character(args.character).spec.bone_lengths if args.character else None
        clip = keys_to_clip(load_keys(args.keys), rig, lengths)
        clip.save(args.out)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}: {clip.frame_count} frames at {clip.fps:g} fps{' (loop)' if clip.loop else ''}")
    return 0


def cmd_clip_layer(args: argparse.Namespace) -> int:
    from ggs.rig.clip import load_clip
    from ggs.rig.layers import BONE_SETS, layer_clips

    try:
        rig = load_rig_spec(args.rig)
        bones = args.bones if args.bones in BONE_SETS else tuple(b.strip() for b in args.bones.split(","))
        clip = layer_clips(load_clip(args.base), load_clip(args.over), rig, bones)
        clip.save(args.out)
        from ggs.rig.library import sidecar

        if sidecar(Path(args.base)).is_file():  # keep where the base motion came from (credits)
            shutil.copy2(sidecar(Path(args.base)), sidecar(Path(args.out)))
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}: {clip.frame_count} frames, {args.over} on {args.bones} over {args.base}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        from ggs.ui.app import main as ui_main
    except ImportError as exc:
        print(f'ERROR: {exc}. Install the desktop app: pip install -e ".[ui]"', file=sys.stderr)
        return 1
    return ui_main(["ggs", *(["--project", args.project] if args.project else [])])


def cmd_asset_new(args: argparse.Namespace) -> int:
    from ggs.items import CharacterSettings, ItemError, ItemSettings, create
    from ggs.rig.poses import DEFAULT

    try:
        settings = ItemSettings(size=args.size, style=args.style)
        if args.asset_class == "character":
            settings.character = CharacterSettings(view=args.view, body_preset=args.body,
                                                   pose=args.pose or DEFAULT[args.view])
        rec, d = create(Path(args.project), args.asset_class, args.name, args.describe or "", settings)
    except (ItemError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"created {rec.asset_class}/{rec.name} -> {d}")
    return 0


def cmd_asset_gen(args: argparse.Namespace) -> int:
    from ggs.items import ItemError, friendly, generate
    from ggs.pipeline.models import MissingModelError
    from ggs.pipeline.vram import VramBudgetError

    profile = args.profile or profile_for(detect_vram_gb())
    try:
        rec, new = generate(Path(args.project), args.item, action=args.action, text=args.text or "",
                            count=args.count, seed=args.seed, profile=profile, model=args.model,
                            enforce_vram=not args.no_vram_check)
    except (ItemError, MissingModelError, VramBudgetError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f'ERROR: {exc}. Install the GPU stack: see README ("gpu" extra).', file=sys.stderr)
        return 1
    for v in new:
        state = "OK" if v.passed else "QA FAILED: " + friendly(v.failures)
        print(f"v{v.n:03d} (seed {v.seed}): {state}  {v.file}")
    print(f"current version: v{rec.current:03d}" if rec.current else "no usable version yet")
    return 0 if any(v.passed for v in new) else 2


def cmd_asset_list(args: argparse.Namespace) -> int:
    from ggs.items import list_items

    rows = list_items(Path(args.project))
    for rec, d in rows:
        cur = f"v{rec.current:03d}" if rec.current else "-"
        print(f"{rec.asset_class:12} {rec.name:28} {len(rec.versions):3} versions, current {cur}  ({d})")
    if not rows:
        print("no items yet")
    return 0


def cmd_asset_export(args: argparse.Namespace) -> int:
    from ggs.items import ItemError, export_item

    try:
        out = export_item(Path(args.project), args.item, Path(args.out) if args.out else None, args.version,
                          args.size)
    except (ItemError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"exported {out}")
    return 0


def cmd_models_status(args: argparse.Namespace) -> int:
    from ggs.pipeline.download import load_packs, pack_ready

    missing_required = False
    for name, pack in load_packs().items():
        ok = pack_ready(pack)
        missing_required |= pack.required and not ok
        need = "required" if pack.required else "optional"
        print(f"{name:10} {'ready  ' if ok else 'missing'} {pack.size_gb:5.1f} GB  {need:8}  {pack.label}")
    return 1 if missing_required else 0


def cmd_models_fetch(args: argparse.Namespace) -> int:
    from ggs.pipeline.download import fetch_pack, load_packs, pack_ready

    packs = load_packs()
    names = [n for n, p in packs.items() if p.required] if args.pack == "required" else [args.pack]
    for n in names:
        if n not in packs:
            print(f"ERROR: unknown model pack {n!r}; known: {sorted(packs)}", file=sys.stderr)
            return 1
        print(f"@stage download_{n}", flush=True)
        try:
            fetch_pack(n, packs[n], progress=lambda m: print(m, flush=True))
        except OSError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"{n}: {'ready' if pack_ready(packs[n]) else 'downloaded, but still not complete'}")
    return 0


def cmd_clip_loop(args: argparse.Namespace) -> int:
    from ggs.rig.clip import load_clip
    from ggs.rig.cliptools import ClipError, cut_loop, find_loop

    try:
        clip = load_clip(args.clip)
        if (args.start is None) != (args.end is None):
            raise ClipError("give both --start and --end, or neither")
        if args.start is not None:
            start, end, cost = args.start, args.end, float("nan")
        else:
            start, end, cost = find_loop(clip, args.min_seconds, args.max_seconds)
        out = cut_loop(clip, start, end)
    except (ClipError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out.save(args.out)
    print(f"wrote {args.out}: loop frames {start}..{end - 1} ({out.frame_count} frames, "
          f"{out.duration:.2f} s), seam mismatch {cost:.1f} deg, travel {out.loop_travel:.3f} heights/loop")
    return 0


def cmd_motion_make(args: argparse.Namespace) -> int:
    """Text -> 3D motion (Kimodo) -> BVH -> 2D clip (+ .gif preview, + .motion.json with prompt and seed)."""
    from ggs.hardware import detect_vram_gb, profile_for
    from ggs.pipeline.models import MissingModelError
    from ggs.pipeline.motion import MODEL, MotionError, generate, save_bvh
    from ggs.pipeline.vram import Stages, VramBudgetError
    from ggs.rig.library import to_clip, write_sidecar
    from ggs.rig.preview import render_preview
    from ggs.rig.retarget import RetargetError

    out = Path(args.out)
    seeds = [args.seed + i for i in range(args.count)]
    try:
        spec = load_rig_spec(args.rig)
        seconds = [float(s) for s in args.seconds.split()] if isinstance(args.seconds, str) else args.seconds
        seconds = seconds[0] if len(seconds) == 1 else seconds
        stages = Stages(profile=profile_for(detect_vram_gb()))
        takes = generate(args.prompt, seconds, seeds, stages, steps=args.steps)
        out.parent.mkdir(parents=True, exist_ok=True)
        for take in takes:
            stem = out.stem if len(takes) == 1 else f"{out.stem}_{take.seed}"
            bvh = save_bvh(take, out.with_name(stem + ".bvh"))
            clip = to_clip(bvh, spec, loop=args.loop, in_place=args.in_place,
                           loop_seconds=(args.loop_min, args.loop_max))
            path = out.with_name(stem + ".json")
            clip.save(path)
            render_preview(spec, clip, path.with_suffix(".gif"), cell=args.cell)
            write_sidecar(path, {"prompt": args.prompt, "seconds": seconds, "seed": take.seed, "model": MODEL,
                                 "steps": args.steps, "loop": args.loop, "in_place": args.in_place,
                                 "frames_3d": take.frames})
            print(f"wrote {path}: {clip.frame_count} frames at {clip.fps:g} fps ({clip.duration:.2f} s), "
                  f"seed {take.seed}", flush=True)
            for w in clip.warnings:
                print(f"warning: {w}")
    except MissingModelError as exc:
        print(f"ERROR: {exc}\nDownload the motion packs: ggs models fetch motion (and motion_text for new "
              "prompts).", file=sys.stderr)
        return 1
    except (MotionError, RetargetError, VramBudgetError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_motion_library(args: argparse.Namespace) -> int:
    from ggs.pipeline.motion import find_embed, split_prompt
    from ggs.rig.library import library_clip, load_library

    try:
        lib = load_library()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    rows = []
    for m in lib.motions:
        clip = library_clip(m.name)
        rows.append({**m.model_dump(), "clip": str(clip) if clip.is_file() else None,
                     "preview": str(clip.with_suffix(".gif")) if clip.with_suffix(".gif").is_file() else None,
                     "embedded": all(find_embed(t) for t in split_prompt(m.prompt))})
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for r in rows:
        state = "ready" if r["clip"] else "no clip"
        print(f"{r['name']:<14} {r['category']:<9} {r['seconds']:4.1f} s  {state:<8} {r['prompt']}")
    return 0


def cmd_motion_add(args: argparse.Namespace) -> int:
    """Library names (``walk``) and catalog references (``cmu/02_01``) into a project's clips."""
    from ggs.rig import catalog
    from ggs.rig.library import add_to

    dest = Path(args.out) if args.out else Path(args.project) / "clips"
    try:
        refs = [n for n in args.names if "/" in n]
        names = [n for n in args.names if "/" not in n]
        for p in [*add_to(names, dest), *catalog.add_to(refs, dest)]:
            print(f"added {p}")
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_motion_catalog(args: argparse.Namespace) -> int:
    """Search the bundled motion catalogs (CMU, 100STYLE)."""
    from ggs.rig import catalog

    try:
        hits = catalog.search(" ".join(args.words), category=args.category, source=args.source)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps([{"ref": f"{i.source}/{m.id}", **m.model_dump()} for i, m in hits[: args.limit]], indent=1))
        return 0
    for i, m in hits[: args.limit]:
        print(f"{i.source + '/' + m.id:<22} {m.category:<13} {m.seconds:6.1f} s  {m.name}")
    print(f"{len(hits)} motions" + (f" (first {args.limit} shown)" if len(hits) > args.limit else ""))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from ggs.export.animated import ExportError, parse_color, save_apng, save_gif, save_webp
    from ggs.export.pack import PackError
    from ggs.export.rig_package import write_rig_package
    from ggs.export.sheet import save_sprite_sheet
    from ggs.rig.character import load_character
    from ggs.rig.clip import load_clip
    from ggs.rig.puppet import RenderError

    try:
        spec = load_rig_spec(args.rig)
        clip = load_clip(args.clip)
        character = load_character(args.character)
        if clip.rig != spec.id:
            raise ValueError(f"clip is for rig {clip.rig!r}, not {spec.id!r}")
        from ggs.rig.equipment import apply_two_hand, load_item

        items = [load_item(i) for i in args.item]
        for it in items:  # two-hand items: the far arm reaches the second grip (IK, every frame)
            if it.spec.hold == "two_hand":
                clip = apply_two_hand(clip, spec, character.spec.bone_lengths, it.spec)
        name = args.name or Path(args.clip).stem
        height = max(8, round(args.height * character.spec.stature))  # taller characters render taller
        if args.format in ("rig", "godot", "unity", "defold"):
            out = write_rig_package(spec, character, {name: clip}, args.out, height,
                                    padding=args.padding, items=items)
            print(f"wrote rig package {out}")
            if args.format == "godot":
                from ggs.export.engines.godot import write_godot

                print(f"wrote {write_godot(out)}")
            elif args.format == "unity":
                from ggs.export.engines.unity import write_unity

                print(f"wrote {write_unity(out)} and {out / 'GgsRigPlayer.cs'}")
            elif args.format == "defold":
                from ggs.export.engines.defold import write_defold

                print(f"wrote {write_defold(out, args.defold_path)}")
            if args.format != "rig":
                print(f"see {out / ('README_' + args.format.upper() + '.txt')}")
            _export_credits(args.clip, Path(out) / "CREDITS.txt")
            return 0
        if args.format == "gif" and not args.matte:
            raise ExportError("GIF has 1-bit alpha; pass --matte #RRGGBB (your game's background colour)")
        frames, anchor = character.puppet(spec, items).render_clip(clip, height, padding=args.padding)
        if args.format == "apng":
            out = save_apng(frames, args.out, clip.fps, clip.loop)
        elif args.format == "webp":
            out = save_webp(frames, args.out, clip.fps, clip.loop)
        elif args.format == "gif":
            out = save_gif(frames, args.out, clip.fps, parse_color(args.matte), clip.loop)
        else:
            out, atlas = save_sprite_sheet(frames, anchor, args.out, name, clip.fps, clip.loop,
                                           padding=args.padding)
            print(f"wrote {atlas}")
    except (ExportError, PackError, RenderError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError:
        print('ERROR: needs Pillow: pip install -e ".[image]"', file=sys.stderr)
        return 1
    w, h = frames[0].size
    print(f"wrote {out}: {clip.frame_count} frames of {w}x{h} px, pivot {anchor}")
    _export_credits(args.clip, Path(out).with_name(Path(out).stem + "_CREDITS.txt"))
    return 0


def _export_credits(clip: str, dest: Path) -> None:
    from ggs.rig.library import write_credits

    p = write_credits([Path(clip)], dest)
    if p:
        print(f"wrote {p}: credit lines for the motion libraries used")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ggs", description="Game Graphic Studio")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="detect GPU/VRAM and pick a profile").set_defaults(func=cmd_doctor)

    rig = sub.add_parser("rig", help="rig spec tools").add_subparsers(dest="rig_cmd", required=True)
    v = rig.add_parser("validate", help="validate a rig YAML file")
    v.add_argument("path")
    v.set_defaults(func=cmd_rig_validate)

    models = sub.add_parser("models", help="model registry").add_subparsers(dest="models_cmd", required=True)
    ls = models.add_parser("list", help="list registry entries")
    ls.add_argument("--commercial-only", action="store_true", help="only entries confirmed commercial-OK")
    ls.set_defaults(func=cmd_models_list)
    models.add_parser("dirs", help="folders searched for model files").set_defaults(func=cmd_models_dirs)
    ad = models.add_parser("add-dir", help="also search this folder for model files (e.g. ComfyUI models)")
    ad.add_argument("path")
    ad.set_defaults(func=cmd_models_add_dir)
    mst = models.add_parser("status", help="which model packs are ready (exit 1 if a required one is missing)")
    mst.set_defaults(func=cmd_models_status)
    mf = models.add_parser("fetch", help="download a model pack into ~/.ggs/models (resumable)")
    mf.add_argument("pack", help="pack name from models/downloads.yaml, or 'required'")
    mf.set_defaults(func=cmd_models_fetch)
    ck = models.add_parser("check", help="show which files each pipeline would load")
    ck.add_argument("--profile", choices=["standard", "lite"], default=None)
    ck.set_defaults(func=cmd_models_check)

    from ggs.recipes import available_recipes

    gen = sub.add_parser("generate", help="generate an asset of an explicit class (recipe)")
    gen.add_argument("--class", dest="asset_class", required=True, choices=available_recipes(),
                     help="asset class (recipe); never inferred from the prompt")
    gen.add_argument("--prompt", required=True, help="what to draw, e.g. 'a red health potion'")
    gen.add_argument("--out", required=True, help="output .png (several seeds add _01, _02 ...)")
    gen.add_argument("--style", default=None, help="trained style: installed name or a .ggsstyle file")
    gen.add_argument("--style-weight", type=float, default=None, help="LoRA strength (default: the style's)")
    gen.add_argument("--style-words", default="", help="extra style words added before the description")
    gen.add_argument("--project", default=None, help="project folder whose installed styles to use")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--count", type=int, default=1, help="number of variants (seeds seed..seed+count-1)")
    gen.add_argument("--model", default=None, help="pipeline id from models/pipelines.yaml, or 'stub' (CPU test)")
    gen.add_argument("--size", default=None, help="override generation size, e.g. 768x768")
    gen.add_argument("--profile", choices=["standard", "lite"], default=None, help="default: from ggs doctor")
    gen.add_argument("--keep-failed", action="store_true", help="also write assets that fail QA (*_QA_FAILED)")
    gen.add_argument("--no-vram-check", action="store_true", help="do not fail on VRAM budget overruns")
    gen.add_argument("--debug", action="store_true", help="also save the raw render and matte per asset")
    gen.set_defaults(func=cmd_generate)

    ast = sub.add_parser("asset", help="named project items: create, generate (chat actions), list, export")
    ast = ast.add_subparsers(dest="asset_cmd", required=True)
    an = ast.add_parser("new", help="create a named item in a project")
    an.add_argument("name")
    an.add_argument("--class", dest="asset_class", required=True, choices=[*available_recipes(), "character"])
    an.add_argument("--view", default="front", choices=["front", "side"], help="characters: front or side view")
    an.add_argument("--body", default="standard", help="characters: body preset (standard, chibi, heroic, ...)")
    an.add_argument("--pose", default=None, help="characters: t_pose, a_pose, idle (front); idle, walk, spread (side)")
    an.add_argument("--project", required=True)
    an.add_argument("--describe", default="", help="what it is (used by the first generation)")
    an.add_argument("--size", default="medium", choices=["small", "medium", "large", "full"],
                    help="export size: small 128, medium 256, large 512 px (longest side), full")
    an.add_argument("--style", default=None)
    an.set_defaults(func=cmd_asset_new)
    ag = ast.add_parser("gen", help="run a chat action on an item: new, refine, regenerate, variations")
    ag.add_argument("item", help="item name, or <class>/<name>")
    ag.add_argument("--project", required=True)
    ag.add_argument("--action", default="new", choices=["new", "refine", "regenerate", "variations"])
    ag.add_argument("--text", default="", help="new: the description; refine: what to change")
    ag.add_argument("--count", type=int, default=3, help="variations: how many")
    ag.add_argument("--seed", type=int, default=None, help="advanced: fix the seed")
    ag.add_argument("--model", default=None)
    ag.add_argument("--profile", choices=["standard", "lite"], default=None)
    ag.add_argument("--no-vram-check", action="store_true")
    ag.set_defaults(func=cmd_asset_gen)
    al = ast.add_parser("list", help="the project's items")
    al.add_argument("--project", required=True)
    al.set_defaults(func=cmd_asset_list)
    ae2 = ast.add_parser("export", help="copy an item's current version to <project>/exports at its size")
    ae2.add_argument("item")
    ae2.add_argument("--project", required=True)
    ae2.add_argument("--version", type=int, default=None)
    ae2.add_argument("--size", default=None, choices=["small", "medium", "large", "full"])
    ae2.add_argument("--out", default=None)
    ae2.set_defaults(func=cmd_asset_export)

    proj = sub.add_parser("project", help="project folders").add_subparsers(dest="project_cmd", required=True)
    pi = proj.add_parser("init", help="make a folder a GGS project (.ggs/)")
    pi.add_argument("path")
    pi.add_argument("--name", default=None)
    pi.add_argument("--model", default=None,
                    help="base model for this project's generation and style training (e.g. sdxl, flux2-klein-4b)")
    pi.set_defaults(func=cmd_project_init)

    st = sub.add_parser("style", help="train and manage styles").add_subparsers(dest="style_cmd", required=True)
    ds = st.add_parser("dataset", help="training images").add_subparsers(dest="dataset_cmd", required=True)
    di = ds.add_parser("import", help="copy and normalise training images (rights confirmation required)")
    di.add_argument("src", help="folder with 20-40 images (.png/.jpg/.webp); .txt captions next to them are kept")
    di.add_argument("--out", required=True, help="dataset folder to create")
    di.add_argument("--i-have-the-rights", action="store_true",
                    help="confirm you own or may use these images for training (required)")
    di.add_argument("--size", type=int, default=1024, help="stored image size (training can use less)")
    di.set_defaults(func=cmd_style_dataset_import)
    sc = st.add_parser("caption", help="auto-caption the dataset (content only) and add the trigger word")
    sc.add_argument("dataset")
    sc.add_argument("--trigger", required=True, help="a rare word that will stand for the style, e.g. GGSINK")
    sc.add_argument("--overwrite", action="store_true", help="recaption images that already have a caption")
    sc.add_argument("--captioner", default="qwen3-vl-2b")
    sc.add_argument("--profile", choices=["standard", "lite"], default=None)
    sc.set_defaults(func=cmd_style_caption)
    tr = st.add_parser("train", help="train a style LoRA and write a portable .ggsstyle package")
    tr.add_argument("dataset")
    tr.add_argument("--name", required=True, help="style name (lowercase, digits, - or _)")
    tr.add_argument("--out", required=True, help="output .ggsstyle file")
    tr.add_argument("--model", default=None,
                    help=f"base model to train on (default: the --project's, else {DEFAULT_BASE})")
    tr.add_argument("--project", default=None, help="take the base model from this project")
    tr.add_argument("--steps", type=int, default=None, help="default: per base model (models/pipelines.yaml)")
    tr.add_argument("--rank", type=int, default=None)
    tr.add_argument("--lr", type=float, default=None)
    tr.add_argument("--resolution", type=int, default=None, help="default: klein 512, sdxl 1024")
    tr.add_argument("--text-tokens", type=int, default=None,
                    help="klein: caption length used in training (default: longest caption; 512 = no cut)")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--checkpoint-every", type=int, default=500)
    tr.add_argument("--work-dir", default=None, help="checkpoints and logs (default: next to --out)")
    tr.add_argument("--profile", choices=["standard", "lite"], default=None)
    tr.set_defaults(func=cmd_style_train)
    si = st.add_parser("install", help="install a .ggsstyle into a project (or ~/.ggs/styles)")
    si.add_argument("package")
    si.add_argument("--project", default=None)
    si.add_argument("--overwrite", action="store_true")
    si.set_defaults(func=cmd_style_install)
    sl = st.add_parser("list", help="installed styles")
    sl.add_argument("--project", default=None)
    sl.set_defaults(func=cmd_style_list)

    mocap = sub.add_parser("mocap", help="motion capture tools").add_subparsers(
        dest="mocap_cmd", required=True
    )
    ins = mocap.add_parser("inspect", help="list a BVH file's joints and which rig profiles match")
    ins.add_argument("path")
    ins.add_argument("--rig", default=str(DEFAULT_RIG))
    ins.set_defaults(func=cmd_mocap_inspect)

    rt = mocap.add_parser("retarget", help="BVH -> 2D clip JSON for a side-view rig")
    rt.add_argument("path")
    rt.add_argument("--profile", required=True, help="mocap profile in the rig spec, e.g. cmu_bvh, mixamo")
    rt.add_argument("--out", required=True, help="output clip .json")
    rt.add_argument("--rig", default=str(DEFAULT_RIG))
    rt.add_argument("--up", default="y", choices=["x", "y", "z"], help="up axis of the file (default y)")
    rt.add_argument("--fps", type=float, default=None, help="resample to this frame rate")
    rt.add_argument("--start", type=int, default=0, help="first source frame (inclusive)")
    rt.add_argument("--end", type=int, default=None, help="last source frame (exclusive)")
    rt.add_argument("--in-place", action="store_true", help="remove forward travel from the root")
    rt.add_argument("--keep-tpose-frame", action="store_true",
                    help="keep frame 0 even when it is a T-pose (dropped by default)")
    rt.set_defaults(func=cmd_mocap_retarget)

    pv = mocap.add_parser("preview", help="stick-figure preview of a clip (.png strip or .gif)")
    pv.add_argument("clip")
    pv.add_argument("--out", required=True)
    pv.add_argument("--rig", default=str(DEFAULT_RIG))
    pv.add_argument("--frames", type=int, default=8, help="frames in a PNG strip")
    pv.add_argument("--cell", type=int, default=160, help="pixel size of one frame")
    pv.set_defaults(func=cmd_mocap_preview)

    mo = sub.add_parser("motion", help="animations from text (Kimodo) and the starter library").add_subparsers(
        dest="motion_cmd", required=True
    )
    mm = mo.add_parser("make", help="text -> motion -> clip .json (+ .bvh, .gif preview, .motion.json)")
    mm.add_argument("--prompt", required=True,
                    help='what the person does; several sentences play one after another ("A person walks. '
                         'The person jumps.")')
    mm.add_argument("--seconds", default="3", help="length (1-10 s); one value, or one per sentence: \"2 1.5\"")
    mm.add_argument("--seed", type=int, default=0)
    mm.add_argument("--count", type=int, default=1, help="takes with seeds seed..seed+count-1 (out_<seed>.json)")
    mm.add_argument("--steps", type=int, default=100, help="denoising steps")
    mm.add_argument("--loop", action="store_true", help="cut a seamless cycle out of the take")
    mm.add_argument("--loop-min", type=float, default=0.6, help="shortest loop, seconds")
    mm.add_argument("--loop-max", type=float, default=1.6, help="longest loop, seconds")
    mm.add_argument("--in-place", action="store_true", help="remove forward travel")
    mm.add_argument("--cell", type=int, default=160, help="preview frame size in pixels")
    mm.add_argument("--rig", default=str(DEFAULT_RIG))
    mm.add_argument("--out", required=True, help="clip .json")
    mm.set_defaults(func=cmd_motion_make)
    ml = mo.add_parser("library", help="list the starter animation library")
    ml.add_argument("--json", action="store_true")
    ml.set_defaults(func=cmd_motion_library)
    mc = mo.add_parser("catalog", help="search the bundled motion catalogs (CMU, 100STYLE)")
    mc.add_argument("words", nargs="*", help="words that must all appear, e.g. walk angry")
    mc.add_argument("--category", default=None)
    mc.add_argument("--source", default=None, help="cmu or 100style")
    mc.add_argument("--limit", type=int, default=50)
    mc.add_argument("--json", action="store_true")
    mc.set_defaults(func=cmd_motion_catalog)
    ma = mo.add_parser("add", help="copy library animations (walk) or catalog motions (cmu/02_01) into a project")
    ma.add_argument("names", nargs="+")
    ma_where = ma.add_mutually_exclusive_group(required=True)
    ma_where.add_argument("--project")
    ma_where.add_argument("--out", help="folder")
    ma.set_defaults(func=cmd_motion_add)

    char = sub.add_parser("character", help="character part sheets").add_subparsers(
        dest="character_cmd", required=True
    )
    pa = char.add_parser("programmer-art", help="draw a placeholder part sheet + character.yaml")
    pa.add_argument("--out", required=True, help="output folder")
    pa.add_argument("--rig", default=str(DEFAULT_RIG))
    pa.add_argument("--px-per-height", type=float, default=600.0)
    pa.add_argument("--name", default="programmer_art")
    pa.add_argument("--body", default=None, help="body shape: a preset (chibi, lanky, ...) or ratios")
    pa.set_defaults(func=cmd_character_programmer_art)
    cm = char.add_parser("master", help="draw a master image over the pose guide, then cut and validate it")
    cm.add_argument("--prompt", default=None, help="the character, e.g. 'a knight in silver armour'")
    cm.add_argument("--out", required=True, help="output folder (one seed_NNN folder per variant)")
    cm.add_argument("--name", default="character")
    cm.add_argument("--rig", default=str(DEFAULT_RIG))
    cm.add_argument("--seed", type=int, default=0)
    cm.add_argument("--count", type=int, default=1, help="variants to try (seeds seed..seed+count-1)")
    cm.add_argument("--style", default=None, help="trained style: installed name or a .ggsstyle file")
    cm.add_argument("--style-weight", type=float, default=None)
    cm.add_argument("--project", default=None)
    cm.add_argument("--model", default=None, help="needs reference editing: flux2-klein-4b (or 'stub')")
    cm.add_argument("--profile", choices=["standard", "lite"], default=None)
    cm.add_argument("--no-vram-check", action="store_true")
    cm.add_argument("--no-cut", action="store_true", help="only write the masters")
    cm.add_argument("--body", default=None,
                    help="body shape of the pose guide: standard, heroic, chibi, stocky, lanky, child, or ratios "
                         "like head=1.6,legs=0.7,arms=1,torso=1,width=1.2")
    cm.add_argument("--pose", default="walk", choices=["walk", "spread"],
                    help="master pose: walk (arms down, small stride) or spread (arm forward, wide stride: less "
                         "overlap, cleaner cuts)")
    cm.add_argument("--stature", type=float, default=1.0,
                    help="in-game height relative to a standard character (1.2 = 20%% taller)")
    cm.add_argument("--no-refine", action="store_true",
                    help="skip the klein edits (repainted fills, head expressions) on valid sheets")
    cm.add_argument("--guide-only", action="store_true", help="use the pose guide itself as the master (no GPU)")
    cm.set_defaults(func=cmd_character_master)
    cr = char.add_parser("refine", help="klein edits on a cut character: repaint filled areas, draw expressions")
    cr.add_argument("characters", nargs="+", help="character folders (or their character.yaml)")
    cr.add_argument("--model", default="flux2-klein-4b")
    cr.add_argument("--seed", type=int, default=0)
    cr.add_argument("--no-fills", action="store_true")
    cr.add_argument("--no-expressions", action="store_true")
    cr.add_argument("--rig", default=str(DEFAULT_RIG))
    cr.add_argument("--profile", choices=["standard", "lite"], default=None)
    cr.add_argument("--no-vram-check", action="store_true")
    cr.set_defaults(func=cmd_character_refine)
    csp = char.add_parser("sprites", help="frame-by-frame animation: the generator draws the character in every "
                                         "pose of a clip (not editable parts; cloth and hair move naturally)")
    csp.add_argument("master", help="master.json written by `ggs character master`")
    csp.add_argument("--clip", required=True, help="clip JSON (e.g. a walk cycle from `ggs clip loop`)")
    csp.add_argument("--out", required=True)
    csp.add_argument("--name", default="sprites")
    csp.add_argument("--frames", type=int, default=8, help="frames drawn in one image (they stay consistent)")
    csp.add_argument("--cols", type=int, default=4)
    csp.add_argument("--rows", type=int, default=2)
    csp.add_argument("--motion", default="walking", help="word for the prompt: walking, running, attacking ...")
    csp.add_argument("--seed", type=int, default=0)
    csp.add_argument("--count", type=int, default=3, help="seeds to try; the best sheet is exported")
    csp.add_argument("--style", default=None)
    csp.add_argument("--style-weight", type=float, default=None)
    csp.add_argument("--project", default=None)
    csp.add_argument("--model", default="flux2-klein-4b")
    csp.add_argument("--rig", default=str(DEFAULT_RIG))
    csp.add_argument("--profile", choices=["standard", "lite"], default=None)
    csp.add_argument("--no-vram-check", action="store_true")
    csp.set_defaults(func=cmd_character_sprites)
    cc = char.add_parser("cut", help="cut a master.json into a part sheet and validate it (exit 2 if rejected)")
    cc.add_argument("master", help="master.json written by `ggs character master`")
    cc.add_argument("--out", required=True, help="character folder to write")
    cc.add_argument("--name", default="character")
    cc.add_argument("--stature", type=float, default=1.0, help="in-game height relative to a standard character")
    cc.add_argument("--rig", default=str(DEFAULT_RIG))
    cc.set_defaults(func=cmd_character_cut)

    ad = sub.add_parser("adapter", help="rig-sheet adapter: pairs, training, evaluation").add_subparsers(
        dest="adapter_cmd", required=True)
    ap = ad.add_parser("pairs", help="pair factory: klein redraws the part-sheet guide, QA keeps clean sheets")
    ap.add_argument("--out", required=True, help="dataset folder (keep it under data/, which git ignores)")
    ap.add_argument("--count", type=int, default=40, help="characters from the built-in list")
    ap.add_argument("--start", type=int, default=0, help="first character of the list")
    ap.add_argument("--seeds", type=int, default=2, help="seeds per character")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="flux2-klein-4b")
    ap.add_argument("--rig", default=str(DEFAULT_RIG))
    ap.add_argument("--profile", choices=["standard", "lite"], default=None)
    ap.add_argument("--no-vram-check", action="store_true")
    ap.set_defaults(func=cmd_adapter_pairs)
    at = ad.add_parser("train", help="train the rig-sheet adapter (master -> part sheet edit LoRA)")
    at.add_argument("pairs", help="folder written by `ggs adapter pairs`")
    at.add_argument("--out", required=True, help="adapter folder to write (e.g. ~/.ggs/adapters/rigsheet)")
    at.add_argument("--model", default="flux2-klein-4b")
    at.add_argument("--steps", type=int, default=1000)
    at.add_argument("--rank", type=int, default=32)
    at.add_argument("--lr", type=float, default=1e-4)
    at.add_argument("--resolution", type=int, default=512, help="master side; the sheet is 4:3 at this width")
    at.add_argument("--seed", type=int, default=0)
    at.add_argument("--checkpoint-every", type=int, default=250)
    at.add_argument("--holdout-every", type=int, default=8, help="every n-th pair is kept out for `adapter eval`")
    at.add_argument("--rig", default=str(DEFAULT_RIG))
    at.add_argument("--profile", choices=["standard", "lite"], default=None)
    at.set_defaults(func=cmd_adapter_train)
    ae = ad.add_parser("eval", help="adapter vs fallback cutter on held-out pairs (ground truth = the pair sheet)")
    ae.add_argument("pairs")
    ae.add_argument("--adapter", required=True)
    ae.add_argument("--out", required=True)
    ae.add_argument("--all", action="store_true", help="evaluate every pair, not only the held-out ones")
    ae.add_argument("--limit", type=int, default=0)
    ae.add_argument("--size", default=None, help="sheet size to generate, e.g. 1024x768 (default: training size)")
    ae.add_argument("--rig", default=str(DEFAULT_RIG))
    ae.add_argument("--profile", choices=["standard", "lite"], default=None)
    ae.set_defaults(func=cmd_adapter_eval)

    clipp = sub.add_parser("clip", help="clip tools").add_subparsers(dest="clip_cmd", required=True)
    lp = clipp.add_parser("loop", help="cut a seamless loop (e.g. one walk cycle)")
    lp.add_argument("clip")
    lp.add_argument("--out", required=True)
    lp.add_argument("--min-seconds", type=float, default=0.6)
    lp.add_argument("--max-seconds", type=float, default=1.6)
    lp.add_argument("--start", type=int, default=None, help="explicit loop start frame (with --end)")
    lp.add_argument("--end", type=int, default=None, help="explicit loop end frame, exclusive")
    lp.set_defaults(func=cmd_clip_loop)
    ck = clipp.add_parser("keys", help="turn a keyframe file (clips/*.yaml: holds, swings) into a clip")
    ck.add_argument("keys")
    ck.add_argument("--out", required=True)
    ck.add_argument("--character", default=None, help="use this character's bone lengths")
    ck.add_argument("--rig", default=str(DEFAULT_RIG))
    ck.set_defaults(func=cmd_clip_keys)
    cl = clipp.add_parser("layer", help="play one clip on part of the body over another (swing while walking)")
    cl.add_argument("base", help="the clip for the rest of the body, e.g. a walk cycle")
    cl.add_argument("--over", required=True, help="the clip for --bones, e.g. a swing or a hold pose")
    cl.add_argument("--bones", default="upper", help="upper (torso, head, arms), arms, near_arm, or bone names")
    cl.add_argument("--out", required=True)
    cl.add_argument("--rig", default=str(DEFAULT_RIG))
    cl.set_defaults(func=cmd_clip_layer)

    ui = sub.add_parser("ui", help="open the desktop app")
    ui.add_argument("--project", default=None, help="project folder to open")
    ui.set_defaults(func=cmd_ui)

    itm = sub.add_parser("item", help="held equipment").add_subparsers(dest="item_cmd", required=True)
    im = itm.add_parser("make", help="item folder from an upright sprite (handle at the bottom)")
    im.add_argument("image", help="PNG from `ggs generate --class held_item`")
    im.add_argument("--out", required=True)
    im.add_argument("--name", required=True)
    im.add_argument("--length", type=float, required=True, help="item length in standing heights (sword ~0.45)")
    im.add_argument("--hold", choices=["one_hand", "two_hand"], default="one_hand")
    im.add_argument("--grip-at", type=float, default=0.12, help="grip height as a share of the item, from the bottom")
    im.add_argument("--hold-angle", type=float, default=-90.0, help="item direction relative to the hand")
    im.set_defaults(func=cmd_item_make)

    ex = sub.add_parser("export", help="render a clip with a character and export it")
    ex.add_argument("clip")
    ex.add_argument("--character", required=True, help="character.yaml")
    ex.add_argument("--format", required=True,
                    choices=["apng", "webp", "gif", "sheet", "rig", "godot", "unity", "defold"])
    ex.add_argument("--out", required=True,
                    help="file (.png/.webp/.gif) or folder for --format rig/godot/unity/defold")
    ex.add_argument("--defold-path", default=None,
                    help="Defold only: where the folder sits in the project (default /ggs/<name>)")
    ex.add_argument("--rig", default=str(DEFAULT_RIG))
    ex.add_argument("--item", action="append", default=[], help="held item folder or item.yaml (repeatable)")
    ex.add_argument("--height", type=int, default=256,
                    help="standing height in pixels of a stature-1.0 character (character.yaml stature scales it)")
    ex.add_argument("--padding", type=int, default=2)
    ex.add_argument("--matte", default=None, help="GIF only: background colour #RRGGBB for soft edges")
    ex.add_argument("--name", default=None, help="animation name (default: clip file name)")
    ex.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
