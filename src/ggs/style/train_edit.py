"""Edit-LoRA training for FLUX.2 klein (needs the ``gpu`` extra): learn "reference image -> target image".

Used by the rig-sheet adapter (M6): reference = master image, target = its part sheet. Same recipe as style
training (``ggs.style.train``), plus the reference: its VAE latents are packed and appended to the noisy
target tokens with klein's reference ids (time offset 10), exactly as ``Flux2KleinPipeline`` does at inference.
The loss is taken on the target tokens only.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from pathlib import Path

from ggs.pipeline.flux2 import (
    Flux2KleinGenerator,
    load_transformer,
    prompt_token_counts,
    scheduler,
    text_token_budget,
)
from ggs.pipeline.vram import Stages
from ggs.style.train import TrainResult, TrainSettings, encode_latents, lora_targets


def train_edit_lora(refs: list, targets: list, prompt: str, files: dict[str, Path], settings: TrainSettings,
                    work_dir: Path, stages: Stages, ref_size: tuple[int, int], target_size: tuple[int, int],
                    progress: Callable[[int, float, float], None] | None = None,
                    previews: list | None = None) -> TrainResult:
    """``refs``/``targets``: PIL images (RGB on white). ``previews``: reference images drawn with the result."""
    import torch
    from diffusers import Flux2KleinPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    if len(refs) != len(targets) or not refs:
        raise ValueError("one target per reference image is required")
    t_start = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    gen = Flux2KleinGenerator(files)
    emb = gen.encode_many([prompt], stages)[0]
    emb = emb[:, :settings.text_tokens or text_token_budget(prompt_token_counts(files, [prompt]))]
    ref_lat = encode_latents(refs, files, ref_size, stages)
    tgt_lat = encode_latents(targets, files, target_size, stages)

    rng = random.Random(settings.seed)
    torch.manual_seed(settings.seed)
    bf16 = torch.bfloat16
    losses: list[float] = []
    lora_path = work_dir / "pytorch_lora_weights.safetensors"
    with stages.stage("train"):
        tr = load_transformer(files)
        tr.requires_grad_(False)
        tr.add_adapter(LoraConfig(r=settings.rank, lora_alpha=settings.alpha or settings.rank,
                                  init_lora_weights="gaussian",
                                  target_modules=lora_targets(tr.config["num_single_layers"])))
        params = [p for p in tr.parameters() if p.requires_grad]
        for p in params:
            p.data = p.data.float()
        tr.enable_gradient_checkpointing()
        tr.to("cuda").train()
        opt = torch.optim.AdamW(params, lr=settings.lr, weight_decay=1e-4)
        sigmas_table = scheduler(files).sigmas.to("cuda")
        n_train = scheduler(files).config.num_train_timesteps
        e = emb.to("cuda", bf16)
        txt_ids = Flux2KleinPipeline._prepare_text_ids(e).to("cuda")

        def save(path: Path, model=tr) -> None:
            state = {k: v.detach().to(bf16).cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
            Flux2KleinPipeline.save_lora_weights(save_directory=str(path.parent), transformer_lora_layers=state,
                                                 weight_name=path.name)

        t0 = time.perf_counter()
        for step in range(1, settings.steps + 1):
            i = rng.randrange(len(tgt_lat))
            x0 = tgt_lat[i].to("cuda")
            ref = ref_lat[i].to("cuda")
            noise = torch.randn_like(x0)
            sigma = sigmas_table[min(int(rng.random() * n_train), n_train - 1)].to(bf16).view(1, 1, 1, 1)
            packed = Flux2KleinPipeline._pack_latents((1.0 - sigma) * x0 + sigma * noise)
            hidden = torch.cat([packed, Flux2KleinPipeline._pack_latents(ref)], dim=1)
            ids = torch.cat([Flux2KleinPipeline._prepare_latent_ids(x0),
                             Flux2KleinPipeline._prepare_image_ids([ref])], dim=1).to("cuda")
            with torch.autocast("cuda", dtype=bf16):
                pred = tr(hidden_states=hidden, timestep=sigma.flatten(), guidance=None, encoder_hidden_states=e,
                          txt_ids=txt_ids, img_ids=ids, return_dict=False)[0]
            pred = Flux2KleinPipeline._unpack_latents_with_ids(pred[:, :packed.size(1)], ids[:, :packed.size(1)])
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
        del save, tr, opt, params, pred, loss, hidden, packed, x0, ref, noise
    images = []
    if previews:
        images = gen.render([(prompt, settings.seed, [r]) for r in previews], target_size[0], target_size[1],
                            stages, lora=lora_path)
    (work_dir / "losses.json").write_text(json.dumps(losses), encoding="utf-8")
    return TrainResult(lora_path, losses, images, time.perf_counter() - t_start, files["transformer"].name)
