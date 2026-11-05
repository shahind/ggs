"""SDXL text-to-image, staged for 8 GB GPUs (needs the ``gpu`` extra).

Stage ``encode``: CLIP-L + OpenCLIP bigG text encoders (bf16, ~1.6 GB), then unloaded. Embeddings are cached.
Stage ``denoise``: UNet (+ the SDXL-Lightning speed LoRA when present, + an optional style LoRA) for all
    seeds, producing latents; then unloaded.
Stage ``decode``: VAE (fp16-fix) latents -> images, then unloaded.

With the speed LoRA (``extras.speed_lora`` in models/pipelines.yaml) generation runs ``speed_steps`` steps
with CFG off and a "trailing" Euler schedule, as the Lightning authors recommend; without it, ``steps`` steps
with CFG ``guidance`` and the negative prompt.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ggs.paths import shared_dir
from ggs.pipeline.vram import Stages

STYLE_ADAPTER = "style"
SPEED_ADAPTER = "speed"
LORA_TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]  # UNet attention, as in diffusers' SDXL LoRA example


def _embed_cache(model_key: str, prompt: str, negative: str) -> Path:
    h = hashlib.sha256(f"{model_key}|{negative}|{prompt}".encode()).hexdigest()[:24]
    return shared_dir() / "cache" / "embeds" / f"sdxl_{h}.pt"


def _variant(folder: Path) -> str | None:
    """'fp16' when the folder holds the half-precision variant files (as downloaded from the hub)."""
    if folder.is_dir() and any(f.name.endswith(".fp16.safetensors") for f in folder.iterdir()):
        return "fp16"
    return None


def load_text_encoders(files: dict[str, Path]):
    import torch
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    root, cfg = files["text_encoder"], files["configs"]
    bf16 = torch.bfloat16
    te1 = CLIPTextModel.from_pretrained(root / "text_encoder", variant=_variant(root / "text_encoder"), dtype=bf16)
    te2 = CLIPTextModelWithProjection.from_pretrained(root / "text_encoder_2",
                                                      variant=_variant(root / "text_encoder_2"), dtype=bf16)
    return te1.eval(), te2.eval(), CLIPTokenizer.from_pretrained(cfg / "tokenizer"), CLIPTokenizer.from_pretrained(
        cfg / "tokenizer_2")


def load_unet(files: dict[str, Path]):
    """SDXL UNet from a diffusers folder or a single .safetensors file (bf16)."""
    import torch
    from diffusers import UNet2DConditionModel

    path = files["transformer"]
    if path.is_dir():
        return UNet2DConditionModel.from_pretrained(path, variant=_variant(path), torch_dtype=torch.bfloat16)
    return UNet2DConditionModel.from_single_file(str(path), config=str(files["configs"]), subfolder="unet",
                                                 torch_dtype=torch.bfloat16)


def load_vae(files: dict[str, Path]):
    import torch
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(files["vae"], variant=_variant(files["vae"]), torch_dtype=torch.bfloat16)


def scheduler(files: dict[str, Path], speed: bool):
    from diffusers import EulerDiscreteScheduler

    folder = files["configs"] / "scheduler"
    if speed:  # SDXL-Lightning needs trailing timesteps
        return EulerDiscreteScheduler.from_pretrained(folder, timestep_spacing="trailing")
    return EulerDiscreteScheduler.from_pretrained(folder)


def _pipeline(files: dict[str, Path], speed: bool, **parts):
    from diffusers import StableDiffusionXLPipeline

    kw = {k: None for k in ("vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet")}
    kw.update(parts)
    return StableDiffusionXLPipeline(scheduler=scheduler(files, speed), add_watermarker=False, **kw)


class SdxlGenerator:
    name = "sdxl"

    def __init__(self, files: dict[str, Path], steps: int = 30, guidance: float = 6.0, speed_steps: int = 8,
                 negative: str = ""):
        self.files = files
        self.speed_lora = files.get("speed_lora")
        self.steps = speed_steps if self.speed_lora else steps
        self.guidance = 0.0 if self.speed_lora else guidance
        self.negative = negative

    def encode_many(self, prompts: list[str], stages: Stages) -> list[dict]:
        """Per prompt: prompt/negative embeddings + pooled embeddings. Text encoders load once, if needed."""
        import torch

        key = f"{self.name}:{self.files['text_encoder'].name}"
        caches = [_embed_cache(key, p, self.negative) for p in prompts]
        todo = [i for i, c in enumerate(caches) if not c.exists()]
        if todo:
            with stages.stage("encode"):
                te1, te2, tok1, tok2 = load_text_encoders(self.files)
                pipe = _pipeline(self.files, False, text_encoder=te1.to("cuda"), text_encoder_2=te2.to("cuda"),
                                 tokenizer=tok1, tokenizer_2=tok2)
                with torch.no_grad():
                    for i in todo:
                        pe, ne, pp, npool = pipe.encode_prompt(prompts[i], device=torch.device("cuda"),
                                                               num_images_per_prompt=1,
                                                               do_classifier_free_guidance=True,
                                                               negative_prompt=self.negative or None)
                        caches[i].parent.mkdir(parents=True, exist_ok=True)
                        torch.save({"prompt_embeds": pe.cpu(), "pooled": pp.cpu(), "negative_embeds": ne.cpu(),
                                    "negative_pooled": npool.cpu()}, caches[i])
                del pipe, te1, te2
        return [torch.load(c, map_location="cpu", weights_only=True) for c in caches]

    def encode(self, prompt: str, stages: Stages) -> dict:
        return self.encode_many([prompt], stages)[0]

    def render(self, jobs: list[tuple[str, int]], width: int, height: int, stages: Stages,
               lora: Path | None = None, lora_scale: float = 1.0) -> list:
        """Images for (prompt, seed) pairs with one UNet load and one VAE load."""
        import torch

        embeds = self.encode_many([p for p, _ in jobs], stages)
        bf16 = torch.bfloat16
        latents = []
        with stages.stage("denoise"):
            unet = load_unet(self.files)
            pipe = _pipeline(self.files, bool(self.speed_lora), unet=unet)
            names, weights = [], []
            if self.speed_lora:
                pipe.load_lora_weights(str(self.speed_lora), adapter_name=SPEED_ADAPTER)
                names.append(SPEED_ADAPTER)
                weights.append(1.0)
            if lora is not None:
                pipe.load_lora_weights(str(lora), adapter_name=STYLE_ADAPTER)
                names.append(STYLE_ADAPTER)
                weights.append(lora_scale)
            if names:
                pipe.set_adapters(names, weights)
            pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            with torch.no_grad():
                for (_, seed), e in zip(jobs, embeds):
                    out = pipe(prompt_embeds=e["prompt_embeds"].to("cuda", bf16),
                               pooled_prompt_embeds=e["pooled"].to("cuda", bf16),
                               negative_prompt_embeds=e["negative_embeds"].to("cuda", bf16),
                               negative_pooled_prompt_embeds=e["negative_pooled"].to("cuda", bf16),
                               height=height, width=width, num_inference_steps=self.steps,
                               guidance_scale=self.guidance, output_type="latent",
                               generator=torch.Generator("cuda").manual_seed(seed))
                    latents.append(out.images.cpu())
            del pipe, unet, out
        return self.decode(latents, stages)

    def decode(self, latents: list, stages: Stages) -> list:
        import torch
        from diffusers.image_processor import VaeImageProcessor

        images = []
        with stages.stage("decode"):
            vae = load_vae(self.files).to("cuda").eval()
            proc = VaeImageProcessor(vae_scale_factor=8)
            with torch.no_grad():
                for lat in latents:
                    x = vae.decode(lat.to("cuda", vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
                    images.append(proc.postprocess(x.float(), output_type="pil")[0].convert("RGB"))
            del vae
        return images

    supports_reference = False

    def generate(self, prompt: str, seeds: list[int], width: int, height: int, stages: Stages,
                 lora: Path | None = None, lora_scale: float = 1.0, reference: list | None = None) -> list:
        if reference:
            raise ValueError("SDXL here has no reference-image editing; use flux2-klein-4b for this")
        return self.render([(prompt, s) for s in seeds], width, height, stages, lora, lora_scale)
