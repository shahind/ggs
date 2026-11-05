# Architecture

## Layers
1. **Engine** (`src/ggs/`): pure Python, no UI. Everything the app does is a function the CLI can call.
2. **CLI** (`ggs ...`): explicit arguments, deterministic. Also the interface Claude/scripts use.
3. **Desktop UI** (later): thin shell over the engine. Same project files, same presets.

## Project data
```
<project>/.ggs/
  project.yaml            name, target engine, view, platform, audience, chosen base model
  style/                  dataset + trained style LoRA + metadata (base model id, trigger word)
  collections/<name>.yaml label -> recipe binding + defaults
  outputs/                generated assets (git-ignored)
~/.ggs/                   models, adapters, presets shared by all projects
```

## Recipe = preset
`{recipe, base model, style LoRA + weight, task adapter, size, prompt template, post-process chain, QA rules, exporter}`.
Dropdowns in the UI select presets; the free-text prompt only fills in the object description.

## Stages and VRAM discipline
`generate -> matte -> normalize -> qa -> export`. A scheduler runs one heavy model at a time and unloads between stages
(text encoder on CPU where needed). This is what makes the 8GB profile possible.

## Character pipeline (rig-first)
pose guide -> master image (klein edit + style LoRA) -> part sheet (fallback cutter now; rig-sheet adapter in M6)
-> normalize -> reassembly QA
-> assemble rig -> animate (retargeted mocap, hold poses, IK, layers) -> export.

## Modules
Built: `rig/` (spec, bvh, retarget, clip, cliptools, pose, puppet renderer, character, master, cutter, validate, make), `export/` (animated,
sheet, rig_package, pack, engines/ for Godot, Unity and Defold), `pipeline/` (model files, VRAM stages, backends,
flux2, sdxl, fp8, matting, post), `qa/`, `style/` (dataset, caption, train, train_sdxl,
package + projects), `models/` (registry),
`recipes.py` + `recipes/*.yaml`, `generate.py`.
`assets/` (M8: background, tile, ui_panel, vfx, seamless helpers), `ui/` (M9 desktop app over the CLI).
Planned: collections, installer, mounts/quadruped rig, matted parallax foreground layers.
