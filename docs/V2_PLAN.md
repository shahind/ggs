# GGS v2 plan (owner review of the v1 app, 2026-11-04)

Target user: a game developer who is not an artist and not an AI person. Few settings, abstract choices, pictures
instead of dropdowns, plain-language errors. The engine and CLI stay deterministic (hard rule 1); the app keeps
driving the CLI.

## 1. App shell (chat-centred)
- **Start screen:** list of projects (cards) + "New project". No project -> the start screen only.
- **Project window:** top bar (project name, base model dropdown, settings), left **tree** like Unity's hierarchy:
  Characters, Assets (grouped by type), Animations, Styles. Centre: the selected item's **outputs** (big preview,
  version strip) with the **chat box** under it. Right: an **Advanced** drawer, closed by default (seed, size,
  style strength, model, steps).
- **Create wizard** (one page): pick the type from icon tiles (sprite, item, icon, background, tile, UI panel,
  VFX, character), the size from tiles (small / medium / large / wide), give a name, describe it -> creates a
  named item in the project and starts the first generation.
- **Items are stored**: `<project>/items/<kind>/<name>/item.json` + `versions/vNNN.png` (+ sidecars). Every
  generation is a version with its prompt and seed, so exports can be redone later.
- **Chat actions**: typing a new message = **Refine** (same seed, new prompt: "make it blue"); **Regenerate**
  button = same prompt, new seed; **Variations** = several new seeds. Seeds never show outside Advanced.
- **Progress bar** with a plain step name ("Drawing...", "Cutting out...", "Checking quality..."), no log.
  Errors become sentences with a suggestion; the raw log stays in a hidden "details" box.

## 2. Characters (three ways)
- **Body and pose editor** (character page): a skeleton viewer with sliders (head, torso, arms, legs, width) and
  draggable joints; pose presets T-pose, A-pose, walk, idle; saved with the character.
- **a. Single image** in the chosen pose and body shape. For engines that bend one sprite with their own bones
  (Unity 2D Animation / Skinning Editor, Spine-like tools). Optionally export the skeleton as JSON.
- **b. Sliced sheet** from a **template**: ~5 predefined layouts for common characters (simple side view;
  detailed side view with hair, cloth and joints; front view; three-quarter view; face expressions). The user
  sees each template's preview and picks one; the model draws the sheet in that layout. The sheet is an output
  on its own (for the user's own rig) and feeds our animator:
  - **Animation library** per project (pre-made clips from motion capture + keyframed holds/attacks; later
    text-to-animation, M10), previews, save/share between projects.
  - **Export**: pick several animations -> engine package (Godot/Unity/Defold, clips included) or sprite sheets
    (one per animation or all in one, chosen frame count).
- **c. Frame by frame**: the model draws every frame of a chosen animation (frame count, body shape) in one
  sheet using the pose sheet as reference (the experimental `character sprites`).

Research risk: in M6 probes klein drew part sheets zero-shot only sometimes (restyled mannequins, leftover guide
parts). Plan: purpose-made templates (neutral grey silhouettes, clear spacing, T-pose master as identity
reference), per-cell QA, several seeds, and a fallback that cuts a T-pose master (limbs apart, so the cut is
clean, unlike the walk pose).

## 3. Animated assets
VFX and other animated assets: the model draws a flipbook (N frames in a grid with spacing); VFX keep the
brightness-as-alpha transparency (owner: fine).

## 4. Delivery
- Own git repository for GGS, commits per step.
- Installable standalone app (owner, 2026-11-04: bundle everything except the main AI models; LoRAs only if ~50 MB).
  Done: `tools/build_app.py` makes `dist/GGS` = a private copy of Python 3.13 (isolated by `python313._pth`) +
  the venv's site-packages (CUDA torch, Qt trimmed by ~580 MB) + ggs + data, 4.35 GB; no PyInstaller, because
  torch/diffusers lazy imports and BiRefNet's remote code want a real Python. `tools/installer.iss` packs it
  into a per-user Inno Setup installer (Start menu / desktop shortcuts to `pythonw -m ggs.ui.app`, optional
  `ggs` on PATH). LoRAs are bundled (`bundled:` in `models/downloads.yaml`, now the 394 MB SDXL-Lightning LoRA,
  copied to `<app>/models/`, searched after ~/.ggs/models). Models: `models/downloads.yaml` packs (drawing
  10.1 GB + cut-out 0.4 GB required; SDXL 6.7 GB and the captioner 4 GB optional), downloaded by `ggs models fetch` (HTTP range resume) from the app's
  first-run "Set up the AI models" screen, which can also point at an existing ComfyUI models folder.

## Order
1 git -> 2 item model + app shell -> 3 wizard, chat actions, progress, errors -> 4 body/pose editor + single
image -> 5 templates + sliced sheets + animation library + export -> 6 frame by frame + VFX flipbooks ->
7 installer. Text-to-animation (M10) after that.

## Status (2026-11-04)
- Done: own git repo; project items with chat versions (`ggs asset ...`); app shell (start screen, tree, chat
  workspace, wizard tiles, Refine/Regenerate/Variations, Advanced, progress, plain errors); characters way **a**
  (front-view rig with T/A-pose, side poses, body & pose editor, single image + skeleton.json, pose QA).
- Done (M10, 2026-11-04): animation library (starter set made with Kimodo, CMU and 100STYLE catalogs, search,
  previews, add to project) and text to animation ("Describe a motion"), in the Animations section.
- Next: way **b** (slice templates, model-drawn sheets, multi-clip export), way **c** in the
  character workspace, VFX flipbooks, installer.
- Known: chibi T-pose can grow extra hands at the sides (the limb check sees missing limbs, not extra ones);
  the classic Animate and Styles pages are still the old look inside the new shell.
