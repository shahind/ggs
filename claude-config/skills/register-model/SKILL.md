---
name: register-model
description: Add a base model, LoRA, motion model, segmentation/matting model or dataset to the registry. Use whenever the code is about to depend on a model or data source that is not in models/registry.yaml.
---

# Register a model

1. Find the primary source (model card or repo LICENSE). Delegate to the `license-auditor` agent for anything that is
   not clearly Apache-2.0/MIT with a confirmed source.
2. Add an entry to `models/registry.yaml` with: `id` (kebab-case), `kind`, `name`, `license`, `commercial`
   (`"yes"` | `"no"` | `verify`; quote yes/no because YAML reads bare yes/no as booleans), `min_vram_gb`,
   `source`, `notes`. LoRAs must set `base` to an existing base model id.
3. Never mark `"yes"` from a blog post or aggregator alone.
4. Run `python -m pytest` (the registry tests enforce that LoRAs have known bases and that `no`/`verify` entries are never bundle-eligible).
5. Download weights into `~/.ggs/models/`, never into the repo.
6. If the model changes VRAM needs, update the profile notes in `CLAUDE.md`.
