"""Style LoRA training (needs the ``gpu`` extra). ``train_style`` picks the trainer for the base model's loader:
FLUX.2 klein here, SDXL in ``ggs.style.train_sdxl``.

FLUX.2 klein:

The LoRA is trained directly on top of the frozen transformer that generation uses (on the lite profile the
4-bit GGUF), so the style learns against exactly the weights it will run with. Stages, one model at a time:
  encode   captions -> embeddings (text encoder streamed, cached)
  latents  images -> VAE latents (patchified + batch-norm normalised, as the pipeline expects)
  train    transformer + LoRA (bf16 autocast, gradient checkpointing, AdamW), then preview images
Loss follows diffusers' klein LoRA example: flow matching, x_t = (1-s) x0 + s noise, target = noise - x0.
Speed: captions are padded to 512 tokens by the pipeline; training cuts the padding to the longest caption
(rounded up to 64), which removes about a third of the tokens per step at 512 px (``text_tokens`` overrides).
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from ggs.pipeline.flux2 import (
    Flux2KleinGenerator,
    load_transformer,
    load_vae,
    prompt_token_counts,
    scheduler,
    text_token_budget,
)
from ggs.pipeline.vram import Stages


class TrainSettings(BaseModel):
    steps: int = Field(1500, ge=1)
    rank: int = Field(16, ge=1, le=256)
    alpha: float | None = None  # default = rank
    lr: float = Field(1e-4, gt=0)
    resolution: int = 512
    seed: int = 0
    checkpoint_every: int = Field(500, ge=1)
    caption_dropout: float = Field(0.0, ge=0, lt=1)
    text_tokens: int | None = Field(None, ge=1, le=512)  # klein only; None = cut padding automatically
    preview_prompts: list[str] = Field(default_factory=lambda: [
        "a wooden treasure chest", "a small dragon sitting", "a glass potion bottle", "a knight with a sword"])


@dataclass
class TrainResult:
    lora: Path
    losses: list[float]
    previews: list = field(default_factory=list)
    seconds: float = 0.0
    trained_on: str = ""


def lora_targets(num_single_layers: int) -> list[str]:
    """Attention projections of the double blocks plus the fused qkv/mlp input and output of single blocks."""
    return ["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj",
            *[f"single_transformer_blocks.{i}.attn.to_out" for i in range(num_single_layers)]]


def encode_latents(images: list, files: dict[str, Path], resolution: int | tuple[int, int], stages: Stages) -> list:
    """VAE latents as the klein pipeline makes them. ``resolution``: square side, or (width, height)."""
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image

    size = (resolution, resolution) if isinstance(resolution, int) else tuple(resolution)
    out = []
    with stages.stage("latents"):
        vae = load_vae(files).to("cuda").eval()
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to("cuda")
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to("cuda")
        with torch.no_grad():
            for img in images:
                im = img.convert("RGB").resize(size, Image.LANCZOS)
                x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
                lat = vae.encode(x.to("cuda", torch.bfloat16)).latent_dist.mode()
                lat = Flux2KleinPipeline._patchify_latents(lat)
                out.append(((lat.float() - mean) / std).to(torch.bfloat16).cpu())
        del vae, mean, std
    return out


def train_lora(images: list, captions: list[str], files: dict[str, Path], settings: TrainSettings, work_dir: Path,
               stages: Stages, trigger: str, progress: Callable[[int, float, float], None] | None = None
               ) -> TrainResult:
    import torch
    from diffusers import Flux2KleinPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    if len(images) != len(captions):
        raise ValueError("one caption per image is required")
    t_start = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    gen = Flux2KleinGenerator(files)
    preview_prompts = [f"{trigger}, {p}" for p in settings.preview_prompts]
    embeds = gen.encode_many(captions + [""] + preview_prompts, stages)  # one text-encoder load for everything
    pemb = embeds[len(captions) + 1:]
    empty = embeds[len(captions)]
    embeds = embeds[:len(captions)]
    n_text = settings.text_tokens or text_token_budget(prompt_token_counts(files, captions))
    embeds = [e[:, :n_text] for e in embeds]  # right padding only, so the real tokens stay
    empty = empty[:, :n_text]
    latents = encode_latents(images, files, settings.resolution, stages)

    rng = random.Random(settings.seed)
    torch.manual_seed(settings.seed)
    bf16 = torch.bfloat16
    losses: list[float] = []
    lora_path = work_dir / "pytorch_lora_weights.safetensors"
    previews = []
    with stages.stage("train"):
        tr = load_transformer(files)
        tr.requires_grad_(False)
        alpha = settings.alpha or settings.rank
        tr.add_adapter(LoraConfig(r=settings.rank, lora_alpha=alpha, init_lora_weights="gaussian",
                                  target_modules=lora_targets(tr.config["num_single_layers"])))
        params = [p for p in tr.parameters() if p.requires_grad]
        for p in params:
            p.data = p.data.float()  # fp32 master weights for the adapter
        tr.enable_gradient_checkpointing()
        tr.to("cuda").train()
        opt = torch.optim.AdamW(params, lr=settings.lr, weight_decay=1e-4)
        sched = scheduler(files)
        sigmas_table = sched.sigmas.to("cuda")
        n_train = sched.config.num_train_timesteps

        def save(path: Path, model=tr) -> None:
            state = {k: v.detach().to(bf16).cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
            Flux2KleinPipeline.save_lora_weights(save_directory=str(path.parent), transformer_lora_layers=state,
                                                 weight_name=path.name)

        t0 = time.perf_counter()
        for step in range(1, settings.steps + 1):
            i = rng.randrange(len(latents))
            x0 = latents[i].to("cuda")
            emb = empty if rng.random() < settings.caption_dropout else embeds[i]
            emb = emb.to("cuda", bf16)
            txt_ids = Flux2KleinPipeline._prepare_text_ids(emb).to("cuda")
            img_ids = Flux2KleinPipeline._prepare_latent_ids(x0).to("cuda")
            noise = torch.randn_like(x0)
            idx = min(int(rng.random() * n_train), n_train - 1)
            sigma = sigmas_table[idx].to(bf16).view(1, 1, 1, 1)
            noisy = (1.0 - sigma) * x0 + sigma * noise
            packed = Flux2KleinPipeline._pack_latents(noisy)
            with torch.autocast("cuda", dtype=bf16):
                pred = tr(hidden_states=packed, timestep=sigma.flatten(), guidance=None,
                          encoder_hidden_states=emb, txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
            pred = Flux2KleinPipeline._unpack_latents_with_ids(pred[:, : packed.size(1)], img_ids)
            loss = torch.nn.functional.mse_loss(pred.float(), (noise - x0).float())
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

        # previews with the trained adapter, same transformer still on the GPU
        tr.eval()
        vae = load_vae(files).to("cuda")
        pipe = Flux2KleinPipeline(scheduler=scheduler(files), vae=vae, text_encoder=None, tokenizer=None,
                                  transformer=tr, is_distilled=True)
        pipe.set_progress_bar_config(disable=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=bf16):  # adapter weights are fp32 masters
            for k, e in enumerate(pemb):
                previews.append(pipe(prompt_embeds=e.to("cuda", bf16), height=settings.resolution,
                                     width=settings.resolution, num_inference_steps=4, guidance_scale=1.0,
                                     generator=torch.Generator("cuda").manual_seed(settings.seed + k)).images[0])
        # drop every reference to the transformer and the last step's tensors (``save`` holds the model) so the
        # stage really frees the GPU
        del pipe, vae, tr, opt, params, save, sigmas_table
        del x0, emb, txt_ids, img_ids, noise, sigma, noisy, packed, pred, loss
    (work_dir / "losses.json").write_text(json.dumps(losses), encoding="utf-8")
    return TrainResult(lora_path, losses, previews, time.perf_counter() - t_start, files["transformer"].name)


def train_style(loader: str, images: list, captions: list[str], files: dict[str, Path], settings: TrainSettings,
                work_dir: Path, stages: Stages, trigger: str,
                progress: Callable[[int, float, float], None] | None = None, *, profile: str = "lite",
                defaults: dict | None = None) -> TrainResult:
    """Train a style LoRA on the base model behind ``loader`` (from models/pipelines.yaml)."""
    if loader == "flux2_klein":
        return train_lora(images, captions, files, settings, work_dir, stages, trigger, progress)
    if loader == "sdxl":
        from ggs.style.train_sdxl import train_sdxl_lora

        return train_sdxl_lora(images, captions, files, settings, work_dir, stages, trigger, progress,
                               profile=profile, defaults=defaults or {})
    raise ValueError(f"no style trainer for loader {loader!r}")
