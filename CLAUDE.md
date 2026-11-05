# GGS — Game Graphic Studio

Local-first desktop app + CLI that lets a person with no graphic skills produce game-ready 2D assets
(characters, items, backgrounds, UI, VFX, animations) using open-weight image models and LoRAs.
"GGS" is a working name; rename freely.

Owner: a solo game developer who is NOT an artist. The product goal is: open app, pick a style,
pick an asset type and output format, type a prompt, get a file that drops into the game engine.

## Hard rules (do not break without asking the owner)

1. **Deterministic backend.** Every CLI call takes an explicit asset class (`--class sprite`). The CLI never
   asks an LLM to interpret the request. It must behave identically when called by Claude, another AI, or a script.
2. **One heavy model resident at a time.** Load, run, unload (free VRAM) before the next stage. The 8GB profile
   must work for inference; 16GB is the primary dev target.
3. **License gate.** Anything bundled or enabled by default must be `commercial: yes` in `models/registry.yaml`.
   `no` is never default. `verify` needs the license-auditor agent and a human decision. Never guess a license.
   **Development-phase exception (owner, 2026-11-03):** while developing, any model may be used if it gives better
   results. Every model used still gets a registry entry with its license recorded honestly, plus a row in
   `docs/LICENSE_REVIEW.md`, which the owner checks before release.
4. **diffusers directly.** No ComfyUI dependency in production code. (ComfyUI is fine for throwaway prototyping.)
5. **Real RGBA.** Transparency comes from native alpha or a matting model, never from a chroma-key shortcut
   that leaves fringes. Every asset passes the QA stage before export.
6. **Windows first.** pathlib everywhere, no bash-only scripts, no symlinks, no hard-coded separators.
7. **Copyright stance.** Users are responsible for the training images they supply. The style-training UI shows a
   warning and requires a "I have the rights to these images" checkbox. The app does not police content.
8. **Never commit** weights, datasets, generated outputs, or `.ggs/` project data.
9. GPU tests are marked `@pytest.mark.gpu` and are skipped by default. Tests on real mocap clips
   (`@pytest.mark.mocap_data`) skip when `data/mocap/` is empty; `data/` is git-ignored.

## Architecture in one screen

```
Project (.ggs/ in the project folder; shared models/adapters in ~/.ggs/)
 ├─ Style        user images -> auto-caption -> style LoRA (portable file)
 ├─ Collections  user labels (Characters, Weapons, Trees...) each bound to a RECIPE
 └─ Recipes      sprite | rig_character | background | tile | ui_kit | vfx | key_art
                 each = {base model, style LoRA, task adapter, size, prompt template,
                         post-process chain, QA rules, exporter}
```

Pipeline stages (all recipes): `generate -> matte -> normalize -> qa -> export`.

Character animation is **rig-first** (cut-out puppet):
pose guide -> master image drawn over it (joints at known pixels) -> part sheet (fixed grid) -> normalize to bone
lengths -> validate by reassembly -> animate with retargeted mocap (3D joints projected to the side plane -> 2D bone angles).
See `docs/RIG_SPEC_V1.md` and `rigs/humanoid_side_v1.yaml`.

Exports: GIF/APNG/WebP, sprite sheet + JSON atlas, or a rig package (parts atlas + skeleton + clips) with an engine importer.
Target engines: **Unity, Godot and Defold** (all three, every exporter).

## Hardware profiles

- `standard` (>=12GB, primary 16GB): klein-4B bf16/fp8, local style-LoRA training at 768px+.
- `lite` (~8GB): GGUF/FP8 inference; local training only at 512px (warn user) or export the dataset for cloud training.
  Measured on the owner's RTX 4060 Laptop (8GB), FLUX.2 klein Q4 at 1024px: the text encoder is streamed
  (0.7GB peak), denoise peaks at 5.0GB, BiRefNet at 1.6GB, and each stage is unloaded before the next.
  SDXL on the same laptop: denoise 5.5GB (bf16 UNet + Lightning LoRA); style training at 1024px 3.8GB with the
  UNet Linears stored in fp8, 1.85 s/step (klein: 3.9 s/step at 512px).
- `minimal` (<7GB) / `no-gpu`: unsupported for generation; `ggs doctor` explains why.

## Repo layout

```
CLAUDE.md            this file
docs/                ARCHITECTURE, ASSET_CLASSES, RIG_SPEC_V1, CLIP_FORMAT_V1, EXPORT_FORMATS, STYLE_TRAINING, PART_CUTTER, ANIMATION_V3,
                     MODELS_AND_LICENSES, LICENSE_REVIEW, ROADMAP, KICKOFF
rigs/                machine-readable rig specs (YAML) — source of truth for bones/parts/sockets
models/registry.yaml model/LoRA/motion registry with license + VRAM metadata
src/ggs/             package (cli.py, hardware.py, generate.py, recipes.py, rig/, export/, pipeline/, qa/, style/,
                     models/, assets/ (M8 asset kinds), ui/ (M9 desktop app))
recipes/             asset-class presets (YAML), e.g. sprite.yaml, held_item.yaml
clips/               keyframed clips (holds, swings) for `ggs clip keys`
models/pipelines.yaml where each pipeline's files are found on disk, per profile
motions/             animation library: starter set (Kimodo), catalogs/*.zip (CMU, 100STYLE), CREDITS.md
tests/               pytest
.claude/agents/      specialist subagents
.claude/skills/      repeatable procedures
```

## Commands

```
pip install -e ".[dev]"          # core + tests
pip install -e ".[dev,image]"    # + pillow/numpy
pip install -e ".[dev,image,gpu]"# + torch/diffusers (install the CUDA build of torch first on Windows)
python -m pytest                 # GPU tests skipped by default
python -m ggs doctor             # detect GPU/VRAM and pick a profile
python -m ggs rig validate rigs/humanoid_side_v1.yaml
python -m ggs models list --commercial-only
python -m ggs models add-dir <ComfyUI models folder>   # reuse weights already on disk
python -m ggs models check                               # which files each pipeline would load
python -m ggs generate --class sprite --prompt "a red health potion" --out potion.png  # --count, --seed, --style-words
python -m ggs style dataset import <imgs> --out <ds> --i-have-the-rights   # then: style caption, style train
python -m ggs project init <dir> --model sdxl; python -m ggs style install x.ggsstyle --project <dir>
python -m ggs style train <ds> --name ink --out ink.ggsstyle --project <dir>   # trains on the project's base model
python -m ggs generate --class sprite --prompt "a lantern" --style <name> --project <dir> --out lantern.png
python -m pytest -m gpu                                  # real-model tests on this machine
python -m ggs mocap inspect data/mocap/cmu/02_01_walk.bvh
python -m ggs mocap retarget data/mocap/cmu/02_01_walk.bvh --profile cmu_bvh --fps 24 --out walk.json
python -m ggs mocap preview walk.json --out walk.gif
python -m ggs clip loop walk.json --out cycle.json
python -m ggs character programmer-art --out chars/pa
python -m ggs character master --prompt "a knight with a red cape" --count 3 --out chars/knight  # guide -> klein -> cut -> validate
python -m ggs character cut chars/knight/seed_000/master.json --out chars/knight_v2   # re-cut; exit 2 = rejected
python -m ggs character master --prompt "a dwarf" --body stocky --stature 0.75 --out chars/dwarf  # body: preset or head=1.6,legs=0.7
python -m ggs character refine chars/knight/seed_000/character   # klein repaints fills + draws expressions
python -m ggs character sprites chars/knight/seed_000/master.json --clip cycle.json --out sprites/knight  # experimental
python -m ggs generate --class held_item --prompt "a steel longsword" --out sword.png; python -m ggs item make sword.png --out items/sword --name sword --length 0.5
python -m ggs clip keys clips/swing_one_hand.yaml --out swing.json; python -m ggs clip layer cycle.json --over swing.json --bones near_arm --out walk_swing.json
python -m ggs export walk_swing.json --character chars/knight/character.yaml --item items/sword --format godot --out build/knight
python -m ggs generate --class tile --prompt "grey cobblestone" --out tiles/cobble.png   # also background, background_wrap, ui_panel, ui_icon, vfx
python -m ggs ui            # desktop app (pip install -e ".[ui]"); same commands underneath
python -m ggs models status; python -m ggs models fetch required   # first-run model packs (models/downloads.yaml)
.venv\Scripts\python tools\build_app.py   # standalone app folder dist/GGS (portable Python + CUDA torch, no models)
ISCC tools\installer.iss                   # -> dist/GGS-Setup-<ver>.exe (+ .bin slices), per-user install
python -m ggs export cycle.json --character chars/pa/character.yaml --format apng --out walk.png  # or webp|gif|sheet|rig
python -m ggs motion make --prompt "A person waves." --seconds 2 --count 2 --out clips/wave.json   # Kimodo text->clip
python -m ggs motion catalog walk angry; python -m ggs motion add walk cmu/02_01 100style/Angry_FW --project <dir>
python tools/make_motion_catalogs.py cmu --src data/mocap/src/cmu-mocap.zip --index data/mocap/src/cmu-index.txt
python tools/make_motion_library.py   # starter set (motions/clips) with Kimodo; tools/install_kimodo.py installs it
python -m ggs export cycle.json --character chars/pa/character.yaml --format godot --out build/hero  # or unity|defold
python -m ggs asset gen Hero --project <dir> --action new --text "a clown"   # refine|regenerate|goto --version N
python -m ggs asset accept Hero --project <dir>                                # keep the design (items/character/hero/design)
python -m ggs asset delete Hero --project <dir> [--current|--drawing N|--sliced|--frames walk]  # to the Recycle Bin
python -m ggs motion remove wave --project <dir>; python -m ggs project delete <dir>
python -m ggs character slice Hero --project <dir> --layout auto [--extras skirt,hair_back,cape] [--recut]  # docs/BODY_LAYOUTS.md
python -m ggs character warp Hero --project <dir> --detail normal   # warp rig from the design (<item>/warp/skin.json)
python -m ggs export walk.json --character <item>/warp/skin.json --format sheet --out walk.png  # the design bent
python -m ggs character frames Hero --project <dir> --clip motions/clips/walk.json --frames 8   # drawn frames
python -m ggs export walk.json punch.json --character <item>/sliced/character.yaml --format godot --out build/hero
python -m ggs export --character <item>/sliced/character.yaml --format unity --out build/parts  # parts only, no clips
```

## Conventions

- Python >=3.10, type hints, pydantic v2 models for every config file, `ruff` for lint.
- Config files are YAML and validated on load; a bad file fails loudly with the path and field.
- Coordinates: screen space, +x right, +y down, angles in degrees clockwise from +y (down). Characters face right;
  the **near side is the character's right side**.
- New asset class, rig, or model => use the matching skill in `.claude/skills/`.
- Before large work, write a short plan in the PR/commit message or `docs/` and update `docs/ROADMAP.md`.

## Agents (in `.claude/agents/`)

| Agent | Use for |
|---|---|
| rig-engineer | rig specs, BVH parsing, 3D->2D retargeting, bone renderer |
| pipeline-engineer | diffusers pipelines, LoRA loading, VRAM management, matting |
| lora-trainer | dataset import, captioning, training wrapper, style package format |
| export-engineer | GIF/APNG/WebP, sprite sheets, atlases, engine importers |
| qa-reviewer | automatic asset checks and reviewing diffs against the hard rules |
| license-auditor | checking licenses of models, LoRAs, datasets, motion data before they are added |

## Definition of done for any task

Tests pass (`python -m pytest`), `ggs rig validate` passes for every rig file, new models have registry entries,
docs updated, no hard rule violated.
