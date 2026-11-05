# Roadmap

Order follows "rig first, then come back": the animation side defines what the character generator must produce.
Each milestone is done only when its acceptance check passes. Update this file as milestones finish.

| # | Milestone | Acceptance check | Agent |
|---|---|---|---|
| M0 | Scaffold: rig spec + validator, model registry, hardware profiles, CLI skeleton | `python -m pytest` green; `ggs rig validate` OK | done |
| M1 | Retargeter: BVH parser, 3D joints -> side-plane 2D bone angles, near/far depth swap, clip JSON | Unit tests on synthetic BVH; CMU walk clip gives a plausible angle curve (leg angle cycles, no flips) | done (CMU 02_01 walk passes `tests/test_mocap_real.py`) |
| M2 | Renderer + exports on procedural programmer-art parts (no AI needed) | Walk cycle exports as APNG, sprite sheet + JSON atlas, and rig package; far limbs tinted; layer order correct when legs cross | done; engine exports for Godot, Unity and Defold pass file-level tests, not yet opened in the engines |
| M3 | Generation core: diffusers wrapper (klein-4B, Z-Image), VRAM manager (load/run/unload), LoRA loading, matting; `ggs generate --class sprite` | One prompt -> trimmed RGBA PNG on 16GB and on the lite profile; no chroma fringe | done on lite (8GB); standard (16GB) not run; LoRA loading and Z-Image moved to M4 |
| M4 | Style trainer: dataset import, rights checkbox, auto-captioning, training wrapper, portable style package | Train a style from ~30 images on 16GB; load it in a second project | built on lite (8GB) for klein and SDXL; klein inkwash run (1500 steps) learned the style; formal second-project eval skipped for now at the owner's request |
| M5 | Fallback cutter (SAM + inpaint) -> part-sheet normalizer + reassembly validator | A generated master image becomes a valid part sheet; validator scores it vs the master and rejects bad cuts | done on lite: 3/3 klein knights valid (score 0.97); bad poses and wrong joints rejected in tests. Pose guide replaced SAM; inpaint is a numpy fill |
| M6 | Pair factory + rig-sheet adapter (master -> part sheet) | Adapter beats the fallback cutter on validator score for new characters | changed by the owner (2026-11-04): zero-shot klein refiner on the cutter's parts (fills + expressions) is the default; adapter tools built, not trained. Body shapes + stature added |
| M7 | Equipment, IK, hold poses, animation layers, mounts, quadruped rig | Character rides a horse and swings a sword; upper/lower body layered correctly | equipment, two-hand IK, keyframe clips and layers done (knight walks with a sword, swings while walking, carries a spear two-handed; items in Godot/Unity/Defold). Mounts and the quadruped rig moved after v1 |
| M8 | Other recipes: background (+parallax layers), tile (seamless), UI kit (9-slice + states), VFX (sheets + particle textures) | Each recipe has preset, QA rules, and exporter tests | done for v1: background (+far layer, wrap), tile, ui_panel, ui_icon, vfx; matted parallax foreground layers and VFX flipbooks moved after v1 |
| M9 | Desktop UI: projects, collections, style/asset-type/format dropdowns, preview | Create project, train style, generate a rigged animated character without the CLI | v1 built (PySide6, `ggs ui`): every page drives the CLI; checked offscreen and with screenshots, not yet clicked through end to end by the owner. Collections and packaging (installer) after v1 |
| M10 | Text-to-motion (Kimodo) + motion library browser | Prompt -> BVH -> clip; license-auditor sign-off recorded in the registry | rig-engineer, license-auditor |
| v2 | 3D model generation | out of scope for now | |

## M1 notes
- CLI: `ggs mocap inspect`, `ggs mocap retarget --profile cmu_bvh`, `ggs mocap preview` (stick figure PNG/GIF).
- `cmu_bvh` verified on CMU 02_01 (Bruce Hahne BVH release, GitHub mirror meshula/cmu-mocap). Those files prepend a
  T-pose frame; `retarget` drops a T-pose frame 0 by default (`--keep-tpose-frame` keeps it).
- 02_01 result: thigh swing about -32..+19 deg, legs anti-phase (r = -0.89), max step 9 deg at 24 fps, no near/far swaps.
- `cmu-mocap` stays `commercial: verify`. The release README quotes CMU: free for research and commercial
  projects. Needs the license-auditor + owner decision before any clip is bundled.
- `cmu_asf` stays unused until an ASF/AMC parser exists (BVH only for now).
- Rig spec gained `attach` (where a bone starts on its parent) for 2D forward kinematics; M2 builds on `ggs.rig.pose`.

## M2 notes
- CLI: `ggs character programmer-art`, `ggs clip loop`, `ggs export --format apng|webp|gif|sheet|rig`.
  Formats: `docs/EXPORT_FORMATS.md`.
- CMU 02_01 -> one 26-frame walk cycle (1.08 s, 0.91 standing heights per loop).
  Rig package = 15 KB atlas + JSON, against 560 KB for the same walk as a sprite sheet.
- Rig spec gained `art_angle` per part. Clips gained `loop` and `loop_travel`.
- Characters whose legs differ from the mocap skeleton get the root track scaled by the leg-length ratio.
- Limits: no foot IK yet, so feet can slide a little, and head variants are packaged but no clip drives them
  yet (both belong to M7).

## M3 notes
- `ggs generate --class sprite`: FLUX.2 klein 4B. The stages are encode (Qwen3-4B streamed onto the GPU,
  embeddings cached on disk), denoise (Q4 GGUF + VAE), matte (BiRefNet), then normalize, QA and export.
  Each heavy stage runs inside `Stages`, which fails the run if it goes over the profile's VRAM budget or
  leaves memory behind.
- Real RGBA: the generator paints on a flat white background and BiRefNet gives the alpha. Known-background
  refinement then clears leftover background connected to the outside, tightens the edge band and
  un-mixes the background colour. QA checks halo, leftover background, cut-off, coverage and specks.
- Lite-profile results: 9/9 varied subjects pass QA, about 28 s per sprite from cold (4 steps at
  1024px), peak VRAM 5.0 GB. Weights are reused from the ComfyUI shared models folder via
  `ggs models add-dir`.
- Next steps on this track: LoRA loading with the registry base check (M4 needs it), keeping models
  resident across a batch of prompts, Z-Image / Qwen-Image as alternative pipelines, and an engine-ready
  sprite export (power-of-two, atlas).
- License gate relaxed for development (see `docs/LICENSE_REVIEW.md`).

## M4 notes
- CLI: `ggs style dataset import --i-have-the-rights`, `ggs style caption` (Qwen3-VL-2B), `ggs style train`,
  `ggs style install/list`, `ggs project init --model`. Details: `docs/STYLE_TRAINING.md`.
- Two base models, picked per project (`project init --model`) or per call (`--model`): FLUX.2 klein 4B
  (default) and SDXL base 1.0. A style only loads on the base it was trained on.
- klein inkwash test (30 synthetic images, 512px, 1500 steps): 101 min on the 8GB laptop (3.85-4.0 s/step,
  peak 3.7 GB). The preview shows the look on unseen subjects; the trigger word was once painted as text on a
  bottle. The laptop GPU runs at a 35 W power cap (50 W max), which limits all GPU timings.
- Speed changes made after that run, not measured yet: klein training cuts caption padding (512 -> ~128 tokens,
  about 1/3 fewer tokens per step). SDXL training at 1024px stores the frozen UNet Linears in fp8 on lite,
  uses min-SNR weighting, and draws previews with the 8-step Lightning LoRA.
- SDXL measured on lite: training 1.85 s/step at 1024 px (peak 3.8 GB, about 46 min for 1500 steps);
  sprite generation encode 9.5 s + denoise ~8 s per seed (peak 5.5 GB) + decode + matte. With the SDXL prompt
  template 3/6 test sprites pass QA (klein: 9/9); SDXL base drifts to several objects or full scenes.
- Open: resume from checkpoint; quality comparison klein vs SDXL; whether 1500 steps is more than needed.

## M5 notes
- Design and checks: `docs/PART_CUTTER.md`. CLI: `ggs character master --prompt ... --count N` (pose guide ->
  klein edit -> BiRefNet -> cut -> validate, one folder per seed), `ggs character cut master.json`.
- No segmentation or pose model: klein redraws a rendered pose guide (programmer-art puppet in the master pose),
  so joints and part masks are known. Three of three knight seeds followed the guide (guide agreement passed).
- Lite timings: encode 16 s, denoise ~25 s per seed with the reference image (peak 5.0 GB), matte 1.5 s per
  seed, cut + validate ~8 s per seed on the CPU.
- Cut knights walk with the CMU clip (APNG and Godot export checked by eye). Scale ~350 px per standing height
  in 256 px cells; a long cape makes the torso part large, which lowers the scale for every part.
- Rig spec: head, torso and limb pivots retuned so joint caps fit; characters may now store their own
  `pivots` (character.yaml), which the normalizer uses to centre each part in its cell. The rig package and
  engine exports read them.
- Far-side accessories (a cape behind the body) are moved to the torso, so they move with the body instead of
  being lost with the far limbs.
- Known limits: areas hidden in the master (torso under the arm) are filled with a smooth numpy fill, which
  looks blurry when the arm swings away; a gap can open where a far limb covered the cape; head variants are
  copies of the head; SDXL cannot draw masters (no reference editing here). M6's adapter or a diffusion
  inpainter should replace the fill.

## M6 notes
- Probe: klein can redraw a part-sheet guide zero-shot, but most results are restyled mannequins or leave guide
  parts untouched, so factory data would teach the adapter mannequin sheets. The owner chose the refiner:
  zero-shot klein edits on the cutter's parts (`ggs character refine`, see `docs/PART_CUTTER.md`).
- Knights: every fill repaint and all 9 expressions passed their checks; the smudged torso, pelvis and limb
  fills became clean armour. About 20 s per edit at 768 px on the lite profile.
- Body shapes (owner request): `--body` presets or ratios shape the pose guide, so characters get their own
  head/torso/arm/leg proportions and widths; `--stature` sets in-game height. Klein followed the shaped guides:
  chibi knight 2/2 valid (0.98-0.99), stocky dwarf 2/2 (0.99), lanky wizard 1/2 (a long robe hanging behind
  the legs is partly lost with the far limbs). All walk with the same CMU clip at their own proportions.
- Adapter tools kept for later: `ggs adapter pairs` (pair factory with per-cell QA), `ggs adapter train`
  (klein edit-LoRA with reference tokens), `ggs adapter eval` (pose fidelity against ground-truth sheets).
- Owner review of the cuts: content-unaware labels (belt on the hand, robe cut into legs) are the main quality
  problem. Tried: a spread master pose (`--pose spread`, helps belts, not robes or beards), klein drawing parts
  from the master (fails zero-shot) and frame-by-frame sprites (`ggs character sprites`, experimental, parked).
  Next for this track: clothing-aware segmentation or garment parts in the rig.
- Windows note: the streamed text encoder sometimes fails with "CUDA error: resource already mapped" while
  pinning memory; it now retries without the stream.

## M7 notes
- `ggs generate --class held_item`, `ggs item make`, `ggs clip keys`, `ggs clip layer`, `ggs export --item`.
- Items are in the rig package (`items[]`) and all three engine exports, drawn above every part; a far-hand item
  would draw in front of the body (no far-hand item slot yet).
- Deferred past v1 (owner: "until the first version of the software"): mounts, the quadruped rig, the seated
  rider pose and gait-driven bob; pistol/rifle hold poses; aiming.

## M8 notes
- Recipes gained `kind` (sprite, background, tile, ui_panel, vfx); `ggs.assets` handles the new kinds with the
  same generate -> post -> QA -> export contract and JSON sidecar.
- Seamless: roll half a period, klein repaints the middle seam band, the repaint is colour-matched to the
  original before blending; tiles are downscaled as a 3x3 repeat so the edges stay continuous.
- Backgrounds: the far layer is the scene with its foreground edited away (works well). Tried for nearer layers:
  the far layer as reference (the model redraws the whole scene), BiRefNet on the scene (finds one object),
  drawn layers on white (painted haze, not flat white): none reliable yet.
- VFX alpha = brightness on black (additive particles): flagged for the owner against hard rule 5 (it is not a
  cut-out of an object; solid objects use the sprite class).
- Lite timings: tile ~1 min for 2 seeds (2 passes), background ~1.5 min per seed, panel/icon/vfx ~30-40 s.

## M9 notes
- `src/ggs/ui/`: `commands.py` (pure-Python command builders, unit-tested), `jobs.py` (QProcess queue, one job
  at a time), `widgets.py` (thumbnail grid with QA badges, log, path pickers), `pages.py`, `app.py`.
- The app writes into the project folder: `assets/<class>/`, `characters/<name>_<time>/seed_NNN/`, `clips/`
  (BVH imports, looped), `items/`, `exports/<character>/`, `style_data/` (datasets and .ggsstyle files).
- PySide6 is LGPL-3.0 (dynamic linking is fine for a closed app; ship its licence) - logged in LICENSE_REVIEW.
- Installer: done in v2 (portable Python + Inno Setup, first-run model downloads; see V2_PLAN section 4).
- Not in v1: collections (labels bound to recipes), drag-and-drop,
  per-asset re-run/upscale, thumbnails of animations in the grid, cancel that also cleans partial outputs.

## Open decisions for the owner
- Hard rule 5 vs VFX: is brightness-as-alpha for additive particle textures acceptable? (It is how emissive
  particles are authored; nothing is keyed out of an object.)
- Engines (decided 2026-11-03): Unity, Godot and Defold. All three have M2 exporters; M8 recipes must export
  to all three.
- Default base model: klein for now; the owner expects most users to pick SDXL (faster). Decide after a
  side-by-side comparison. The UI (M9) lets users choose per project.
- v1 rig view is side-view only; front and top-down need their own rig specs.
- Desktop UI toolkit (Qt/PySide, Dear PyGui, or web-based local UI) before M9.
