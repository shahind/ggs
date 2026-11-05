# License review list — check before release

**Development-phase decision (owner, 2026-11-03):** during development any model may be used if it gives
better results, whatever its license. Every model actually used must still have a registry entry, with its
license recorded as known. Before a release, the owner goes through this list and replaces or licenses
anything that is not commercial-OK. Until then nothing here counts as cleared.

Status key: **used** = the code loads it today; **candidate** = on disk or planned, not used yet.

| Model / data | Where it comes from | License as found | Commercial? | Status | Used for |
|---|---|---|---|---|---|
| FLUX.2 klein 4B (distilled) | black-forest-labs/FLUX.2-klein-4B (configs, tokenizer, VAE) | Apache-2.0 (model card, 2026-11-03) | yes per card | **used** | sprite generation (M3) |
| FLUX.2 klein 4B GGUF Q4_K_M | unsloth/FLUX.2-klein-4B-GGUF | apache-2.0 on the repo; quantised copy of the above | yes per card | **used** | lite-profile transformer |
| Qwen3-4B text encoder | ComfyUI `text_encoders/qwen_3_4b.safetensors` (the klein text encoder) | Qwen3-4B is Apache-2.0 per Qwen's release; not re-checked for this file | likely yes; **verify** | **used** | prompt encoding |
| BiRefNet | ZhengPeng7/BiRefNet | MIT (model card) | yes per card | **used** | matting / real alpha |
| Qwen3-VL-2B-Instruct | Qwen/Qwen3-VL-2B-Instruct | apache-2.0 (model card, 2026-11-03) | yes per card | **used** | auto-captioning style datasets (training time only, never shipped) |
| SDXL base 1.0 | stabilityai/stable-diffusion-xl-base-1.0 (fp16 files) | CreativeML Open RAIL++-M (LICENSE.md, 2026-11-03) | yes, with use restrictions that must be passed on to users | **used** | second base model: generation + style training |
| SDXL VAE fp16 fix | madebyollin/sdxl-vae-fp16-fix | MIT (hub card, 2026-11-03) | yes per card | **used** | SDXL decoding/encoding |
| SDXL-Lightning 8-step LoRA | ByteDance/SDXL-Lightning | openrail++ (hub card, 2026-11-03) | yes per card | **used** | 8-step SDXL generation |
| CMU motion capture (BVH) | meshula/cmu-mocap mirror of the Bruce Hahne release | CMU: "free for research and commercial projects" (quoted in the release README) | likely yes; **verify** on CMU's site | **used** (dev data) | M1/M2 walk clip |
| FLUX.2 klein base 4B | ComfyUI `diffusion_models/flux-2-klein-base-4b.safetensors` | not checked | **verify** | candidate | style-LoRA training (M4) |
| Qwen-Image 2.1 (native RGBA) | ComfyUI `qwen_image_2.1_int8_convrot` + `qwen3.5_9b` text encoder | Qwen Research License (registry) | **no** | candidate | could replace matting (native transparency) |
| Krea 2 turbo | ComfyUI `krea2_turbo_fp8_scaled` | not checked | **verify** | candidate | alternative base model |
| fal klein 4B sprite-sheet LoRA | fal/flux-2-klein-4b-spritesheet-lora | unknown | **verify** | candidate | multi-angle sheets |
| svntax pixel 4-walk LoRA | svntax-dev/pixel_spritesheet_4walk_small_lora_v1 | unknown | **verify** | candidate | pixel walk sheets |
| Kimodo SOMA | nvidia/Kimodo | NVIDIA Open Model License | **verify** | candidate | text-to-motion (M10) |

Not usable, even in the final product, without a separate license: RMBG-2.0 (CC BY-NC 4.0), FLUX.2 klein 9B
(non-commercial), and MoMask / MDM / T2M-GPT (AMASS-derived, non-commercial).

Software dependencies to ship with their licences: PySide6 / Qt (LGPL-3.0, desktop app), diffusers,
transformers, peft, torch (Apache-2.0 / BSD-style).

**The installer (tools/build_app.py, 2026-11-04) bundles**, besides GGS itself: CPython 3.13 (PSF licence, copied
to `licenses/`), torch 2.11 + the NVIDIA CUDA 12.8 / cuDNN 9 runtime DLLs it ships with (redistributable under
NVIDIA's CUDA EULA / cuDNN SLA "distributable" terms: **verify** before release), PySide6 / Qt trimmed to
Core/Gui/Widgets/Svg/Network/OpenGL (LGPL-3.0: the DLLs stay separate files the user can replace, which is what
the LGPL needs; add the LGPL text and a Qt source offer to `licenses/` before release), numpy, pillow, pydantic,
pyyaml, requests, huggingface_hub, diffusers, transformers, accelerate, peft, safetensors, gguf, timm, kornia,
einops and their dependencies (permissive licences; collect each package's licence file into `licenses/` before
release). **No model weights or LoRAs are bundled**: the app downloads them on first run from the repos in
`models/downloads.yaml` (the same rows as the table above). The SDXL-Lightning LoRA (394 MB) is in the optional
SDXL download, not in the installer, until the owner decides otherwise.

When a new model is used, add a row here in the same commit.
