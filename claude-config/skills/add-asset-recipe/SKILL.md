---
name: add-asset-recipe
description: Add or change an asset recipe (sprite, background, tile, ui_kit, vfx, key_art, rig_character). Use when a new asset class or output format is requested, or when presets for an existing class change.
---

# Add an asset recipe

1. Read `docs/ASSET_CLASSES.md`. Decide whether the request is a new recipe or a new preset of an existing one.
   Object types (tree, dog, button) are NOT recipes; they are free-text descriptions inside a recipe.
2. Define the preset (YAML, validated by a pydantic model): base model id, style-LoRA slot, task adapter id (optional),
   size and aspect ratio, prompt template with `{style_trigger}` and `{description}`, post-process chain, QA rules,
   exporter, minimum VRAM profile.
3. Every model/adapter id must exist in `models/registry.yaml` with `commercial: "yes"` if it is a default.
4. Add stages in order: generate -> matte -> normalize -> qa -> export. Each stage must be independently testable.
5. Add QA rules (see the qa-reviewer agent): alpha fringe, trim/padding, palette limit, tile edge match, etc.
6. Add a CLI path: `ggs generate --class <recipe> --style <name> --format <fmt> --prompt "..."`.
   The class is always explicit; never infer it from the prompt.
7. Tests: a CPU-only test with a stub generator that proves the stage chain and exporter; a `@pytest.mark.gpu` test for the real model.
8. Update `docs/ASSET_CLASSES.md` and `docs/ROADMAP.md`.
