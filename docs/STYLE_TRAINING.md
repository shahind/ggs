# Training a style (M4)

A *style* is a small LoRA (about 33 MB at rank 16) that teaches the generator your look. It's stored in a
portable `.ggsstyle` file you can install into any project.

## Steps

```powershell
# 1. Collect 20-40 images in one consistent look (any size; PNG/JPG/WebP). Only use images you have the rights to.
python -m ggs style dataset import my_art --out styles\inkwash_ds --i-have-the-rights

# 2. Auto-caption (Qwen3-VL-2B). Captions describe content only; the trigger word stands for the style.
python -m ggs style caption styles\inkwash_ds --trigger GGSINKWASH
#    -> review/edit styles\inkwash_ds\images\*.txt if you like

# 3. Train on a base model: flux2-klein-4b (default) or sdxl. --project takes the project's base model.
python -m ggs style train styles\inkwash_ds --name inkwash --out inkwash.ggsstyle --model sdxl

# 4. Use it in any project on the same base model
python -m ggs project init C:\games\my_game --model sdxl
python -m ggs style install inkwash.ggsstyle --project C:\games\my_game
python -m ggs generate --class sprite --prompt "a lantern" --style inkwash --project C:\games\my_game --out lantern.png
```

`--style` also accepts a path to a `.ggsstyle` file. Styles installed without `--project` go to
`~/.ggs/styles` and are available to every project. Set the strength with `--style-weight` (default 1.0).

## Choosing a base model
| | FLUX.2 klein 4B | SDXL base 1.0 |
|---|---|---|
| Generation | 4 steps, best prompt following | 8 steps with the SDXL-Lightning LoRA (30 + CFG without it) |
| Training default | 512 px, 1500 steps | 1024 px, 1500 steps, min-SNR loss |
| Lite (8 GB) training | Q4 GGUF base, 3.9-4.0 s/step at 512 px, peak 3.7 GB; 1500 steps = 101 min | UNet Linears in fp8, 1.85 s/step at 1024 px, peak 3.8 GB; 1500 steps ~ 46 min |
| Lite sprite time | ~28 s, 9/9 test subjects pass QA | ~13 s per extra seed; 3/6 pass QA (extra objects, full-frame scenes) |
| Ecosystem | new, few LoRAs | very many community LoRAs and fine-tunes |

A project stores its choice in `.ggs/project.yaml` (`base_model`); `generate` and `style train --project` use it
unless `--model` is given. Styles are locked to the base they were trained on.

## Speed
- klein: the pipeline pads every caption to 512 text tokens. Training cuts that padding to the longest caption
  rounded up to 64 (usually 128), so each step processes about a third fewer tokens at 512 px. Generation still
  uses the full 512. `--text-tokens 512` turns the cut off. The speed-up has not been measured yet.
- SDXL: about 2x faster per step than klein, at 4x the pixels (measured on the 8 GB laptop). On the lite profile the frozen UNet
  Linear weights are stored as fp8 (like kohya's `--fp8_base`), so it fits in 8 GB; the LoRA stays fp32.
- Laptop GPUs: the RTX 4060 Laptop used for testing runs at a 35 W power cap (the driver allows 50 W).
  The laptop maker's performance mode, with the charger plugged in, can make every GPU task faster.
- Fewer steps: checkpoints are saved every 500 steps (`--checkpoint-every`); a style often looks right before
  the last step.

## How it works
- **Rights gate:** `dataset import` refuses to run without `--i-have-the-rights` (exit code 3). The desktop UI will
  show the same warning with a checkbox. The confirmation and its time are stored in `dataset.yaml`.
- **Normalisation:** transparency is flattened onto white, each image is padded to a square with its border
  colour and resized (default 1024 px). Exact duplicates are skipped, and a `.txt` caption next to a source image
  is kept.
- **Captions** follow BFL's klein guidance: describe only what is in the image, never the style, and start every
  caption with the trigger word. Sentences that mention the medium (watercolour, ink, illustration...) are
  dropped automatically.
- **Training (SDXL)** follows diffusers' SDXL LoRA example: epsilon prediction on DDPM noise with min-SNR
  weighting (gamma 5), LoRA on the UNet attention projections, captions encoded once with both CLIP encoders.
  Previews are drawn by the normal generator (Lightning LoRA + the new style).
- **Training (klein)** follows diffusers' klein LoRA example: flow-matching loss with LoRA on the attention layers and the
  single-block projections. The difference is that the LoRA trains on top of the frozen transformer that
  generation uses (the 4-bit GGUF on the lite profile), so the style is learned against the exact weights it will
  run with. One model is on the GPU at a time: the text encoder (captions, streamed), then the VAE (latents), then
  the transformer and LoRA (peak about 3.7 GB at 512 px). Checkpoints are saved every 500 steps in the work folder.
- **Package** (`.ggsstyle` is a zip): `style.yaml` (trigger, base model, settings, image count, rights flag),
  `lora.safetensors`, `preview.png` (4 samples drawn right after training) and `captions.txt`.
- **Base-model lock:** a style records the base it was trained on (`flux2-klein-4b` or `sdxl`) and won't load
  on any other base model.

## Settings
| Flag | Default | Notes |
|---|---|---|
| `--model` | project's, else `flux2-klein-4b` | `sdxl` or `flux2-klein-4b` |
| `--steps` | 1500 | BFL suggests 1500-2500 for klein styles |
| `--rank` | 16 | Higher learns more detail; the file grows with rank |
| `--lr` | 1e-4 | BFL range 8e-5 to 1e-4 |
| `--resolution` | klein 512, SDXL 1024 | klein: lite up to 768 (slower), standard up to 1024 |
| `--text-tokens` | auto | klein only: caption length used in training |

Defaults per base model live in `models/pipelines.yaml` (`training:`).

## Known limits
- On the lite profile, klein training at 1024 px is blocked; 768 px is allowed with a warning.
- The trigger word can show up as painted text in an image (seen once on a bottle in the klein test). The
  sprite prompt asks for no text; pick trigger words that don't look like real words.
- There's no resume from a checkpoint yet. A stopped run starts over.
- The sprite recipe's QA was built for crisp game art. Very soft or splattery styles (watercolour) and white
  or glass objects on the white generation background may fail QA. Use `--keep-failed` to inspect them.
