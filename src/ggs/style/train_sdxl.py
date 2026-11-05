"""Style LoRA training for SDXL (needs the ``gpu`` extra).

Stages, one model at a time:
  encode   captions -> CLIP-L + bigG embeddings and pooled embeddings (cached)
  latents  images -> VAE latents x scaling factor
  train    UNet + LoRA on the attention projections (bf16 autocast, gradient checkpointing, AdamW).
           On the lite profile the frozen UNet Linear weights are stored in fp8 (``ggs.pipeline.fp8``),
           which is what lets 1024 px training fit in 8 GB.
  previews the normal generator (speed LoRA included) draws the preview prompts with the new style
Loss follows diffusers' SDXL LoRA example: epsilon prediction on DDPM noise, with min-SNR weighting
(gamma 5), which converges in fewer steps than plain MSE.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from pathlib import Path

from ggs.pipeline.sdxl import LORA_TARGETS, SdxlGenerator, load_unet, load_vae
from ggs.pipeline.vram import Stages
from ggs.style.train import TrainResult, TrainSettings

SNR_GAMMA = 5.0


def encode_latents(images: list, files: dict[str, Path], resolution: int, stages: Stages) -> list:
    import numpy as np
    import torch
    from PIL import Image

    out = []
    with stages.stage("latents"):
        vae = load_vae(files).to("cuda").eval()
        with torch.no_grad():
            for img in images:
                im = img.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
                x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
                lat = vae.encode(x.to("cuda", vae.dtype)).latent_dist.mode() * vae.config.scaling_factor
                out.append(lat.to(torch.bfloat16).cpu())
        del vae
    return out


def train_sdxl_lora(images: list, captions: list[str], files: dict[str, Path], settings: TrainSettings,
                    work_dir: Path, stages: Stages, trigger: str,
                    progress: Callable[[int, float, float], None] | None = None, *, profile: str = "lite",
                    defaults: dict | None = None) -> TrainResult:
    import torch
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    if len(images) != len(captions):
        raise ValueError("one caption per image is required")
    d = defaults or {}
    t_start = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    gen = SdxlGenerator(files, steps=int(d.get("steps", 30)), guidance=float(d.get("guidance", 6.0)),
                        speed_steps=int(d.get("speed_steps", 8)), negative=str(d.get("negative", "")))
    embeds = gen.encode_many(captions + [""], stages)
    empty, embeds = embeds[-1], embeds[:-1]
    latents = encode_latents(images, files, settings.resolution, stages)

    rng = random.Random(settings.seed)
    torch.manual_seed(settings.seed)
    bf16 = torch.bfloat16
    losses: list[float] = []
    lora_path = work_dir / "pytorch_lora_weights.safetensors"
    with stages.stage("train"):
        unet = load_unet(files)
        unet.requires_grad_(False)
        alpha = settings.alpha or settings.rank
        unet.add_adapter(LoraConfig(r=settings.rank, lora_alpha=alpha, init_lora_weights="gaussian",
                                    target_modules=LORA_TARGETS))
        params = [p for p in unet.parameters() if p.requires_grad]
        for p in params:
            p.data = p.data.float()  # fp32 master weights for the adapter
        if profile == "lite":
            from ggs.pipeline.fp8 import linears_to_fp8

            linears_to_fp8(unet)  # after add_adapter, so the LoRA weights stay full precision
        unet.enable_gradient_checkpointing()
        unet.to("cuda").train()
        opt = torch.optim.AdamW(params, lr=settings.lr, weight_decay=1e-4)
        ddpm = DDPMScheduler.from_pretrained(files["configs"] / "scheduler")
        alphas_cumprod = ddpm.alphas_cumprod.to("cuda")
        n_train = ddpm.config.num_train_timesteps
        res = settings.resolution
        time_ids = torch.tensor([[res, res, 0, 0, res, res]], device="cuda", dtype=bf16)  # size, crop, target

        def save(path: Path, model=unet) -> None:
            state = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            state = {k: v.detach().to(bf16).cpu().contiguous() for k, v in state.items()}
            StableDiffusionXLPipeline.save_lora_weights(save_directory=str(path.parent), unet_lora_layers=state,
                                                        weight_name=path.name)

        t0 = time.perf_counter()
        for step in range(1, settings.steps + 1):
            i = rng.randrange(len(latents))
            x0 = latents[i].to("cuda", torch.float32)
            e = empty if rng.random() < settings.caption_dropout else embeds[i]
            noise = torch.randn_like(x0)
            t = torch.tensor([rng.randrange(n_train)], device="cuda", dtype=torch.long)
            noisy = ddpm.add_noise(x0, noise, t).to(bf16)
            with torch.autocast("cuda", dtype=bf16):
                pred = unet(noisy, t, encoder_hidden_states=e["prompt_embeds"].to("cuda", bf16),
                            added_cond_kwargs={"text_embeds": e["pooled"].to("cuda", bf16), "time_ids": time_ids},
                            return_dict=False)[0]
            snr = alphas_cumprod[t] / (1.0 - alphas_cumprod[t])
            weight = torch.clamp(snr, max=SNR_GAMMA) / snr
            mse = torch.nn.functional.mse_loss(pred.float(), noise, reduction="none").mean(dim=(1, 2, 3))
            loss = (weight * mse).mean()
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"training diverged at step {step} (loss {loss.item()})")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(loss.item(), 5))
            if progress:
                progress(step, losses[-1], time.perf_counter() - t0)
            if step % settings.checkpoint_every == 0 and step < settings.steps:
                save(work_dir / f"checkpoint_{step:05d}.safetensors")
        save(lora_path)
        # drop every reference to the UNet (``save`` holds one) so the stage really frees the GPU
        del save, unet, opt, params, pred, loss, mse, noisy, x0, noise

    # previews: the real generation path (fresh UNet, speed LoRA, the new style LoRA)
    jobs = [(f"{trigger}, {p}", settings.seed + k) for k, p in enumerate(settings.preview_prompts)]
    previews = gen.render(jobs, res, res, stages, lora=lora_path)
    (work_dir / "losses.json").write_text(json.dumps(losses), encoding="utf-8")
    tr = files["transformer"]
    return TrainResult(lora_path, losses, previews, time.perf_counter() - t_start, f"{tr.parent.name}/{tr.name}")
