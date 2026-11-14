# Models and licenses (research snapshot: 2026-11-04)

Machine-readable version: `models/registry.yaml`. This file records the reasoning and open items.
**Release checklist of everything actually used: `docs/LICENSE_REVIEW.md`** (development-phase decision,
2026-11-03: any model may be used during development; licenses are cleared before release).

## Defaults (commercial-OK per sources checked)
- FLUX.2 klein 4B (Apache 2.0): generation + reference editing. The 9B variant is non-commercial: never default.
- SDXL base 1.0 (CreativeML Open RAIL++-M): commercial use allowed, but the license's use restrictions must be
  passed on to users. Used with the SDXL VAE fp16 fix (MIT) and the SDXL-Lightning 8-step LoRA (openrail++).
- Z-Image Turbo / Base (Apache 2.0).
- Qwen-Image 2512 and Edit-2511 (Apache 2.0).

## Avoid for a shipped game
- Qwen-Image 2.1: native RGBA output but research license, non-commercial only.
- MoMask / MDM / T2M-GPT: trained on AMASS-derived data whose license forbids commercial use and commercial training.

## Must be resolved before use (status `verify`)
- Kimodo SOMA models: G1 model card says ready for commercial use under the NVIDIA Open Model License; confirm SOMA.
- HY-Motion 1.0: sources conflict (MIT / Apache / custom community license with territorial exclusions). Read LICENSE.
- CMU mocap: reported as free for all uses; confirm on CMU's site; do not resell raw data.
- Community LoRAs (sprite-sheet, 4-walk): check each repo's license; a LoRA inherits constraints from its base model.
- Mixamo: free, but Adobe terms need reading; do not use it as ML training data.
- Google Nano Banana (Gemini API, cloud; added 2026-11-13): a hosted service used with the user's own API key,
  opt-in, never default, nothing bundled. Read the Gemini API Additional Terms on output ownership and commercial
  use before release. Outputs carry Google's SynthID watermark. See `docs/CLOUD_MODELS.md`.

## Process
Every new model/LoRA/dataset gets a registry entry and a license-auditor review. Record the source URL and date.
Users supply their own training images and are responsible for their rights; the app shows a warning + checkbox.
