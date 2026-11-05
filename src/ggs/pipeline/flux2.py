"""FLUX.2 klein 4B text-to-image, staged for 8 GB GPUs (needs the ``gpu`` extra).

Stage ``encode``: Qwen3-4B text encoder streamed block by block onto the GPU (exact bf16, ~0.7 GB peak),
    then unloaded. Prompt embeddings are cached on disk, so repeating a prompt skips this stage.
Stage ``denoise``: transformer (GGUF on the lite profile) + VAE (+ optional style LoRA) on the GPU for all
    seeds, then unloaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ggs.paths import shared_dir
from ggs.pipeline.vram import Stages

TEXT_LAYERS = (9, 18, 27)
MAX_TOKENS = 512
LORA_ADAPTER = "style"


def _embed_cache(model_key: str, prompt: str) -> Path:
    h = hashlib.sha256(f"{model_key}|{TEXT_LAYERS}|{MAX_TOKENS}|{prompt}".encode()).hexdigest()[:24]
    return shared_dir() / "cache" / "embeds" / f"{h}.pt"


def text_token_budget(counts: list[int], multiple: int = 64, cap: int = MAX_TOKENS) -> int:
    """Text length for training: the longest prompt, rounded up to ``multiple``. Prompts are padded on the
    right to 512 tokens, so cutting the padding drops ~1/3 of all tokens per step at 512 px."""
    longest = max(counts, default=1)
    return min(cap, max(multiple, -(-longest // multiple) * multiple))


def prompt_token_counts(files: dict[str, Path], prompts: list[str]) -> list[int]:
    """Tokens each prompt really uses (chat template included, as the pipeline encodes it). CPU only."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(files["configs"] / "tokenizer")
    out = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        out.append(min(MAX_TOKENS, len(tok(text)["input_ids"])))
    return out


def _load_text_encoder(te_path: Path, configs: Path):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    cfg = AutoConfig.from_pretrained(configs / "text_encoder")
    if te_path.is_dir():  # diffusers layout
        te = Qwen3ForCausalLM.from_pretrained(te_path, torch_dtype=torch.bfloat16)
    else:  # single file (ComfyUI layout uses the same key names)
        with torch.device("meta"):
            te = Qwen3ForCausalLM(cfg)
        missing, unexpected = te.load_state_dict(load_file(str(te_path)), strict=False, assign=True)
        bad = [k for k in missing if k != "lm_head.weight"] + list(unexpected)
        if bad:
            raise ValueError(f"{te_path}: not a Qwen3 text encoder ({bad[:5]} ...)")
        te.model.rotary_emb = Qwen3RotaryEmbedding(config=cfg)  # non-persistent buffer
    te.lm_head = torch.nn.Identity()  # the pipeline only reads hidden states
    left = [n for n, t in [*te.named_parameters(), *te.named_buffers()] if t.is_meta]
    if left:
        raise ValueError(f"{te_path}: weights missing for {left[:5]}")
    return te.eval()


def load_transformer(files: dict[str, Path]):
    """Flux2 transformer from a GGUF file, a diffusers folder or a single safetensors file (bf16 compute)."""
    import torch
    from diffusers import Flux2Transformer2DModel

    bf16 = torch.bfloat16
    path, configs = files["transformer"], str(files["configs"])
    if path.suffix == ".gguf":
        from diffusers import GGUFQuantizationConfig

        return Flux2Transformer2DModel.from_single_file(
            str(path), quantization_config=GGUFQuantizationConfig(compute_dtype=bf16), config=configs,
            subfolder="transformer", torch_dtype=bf16)
    if path.is_dir():
        return Flux2Transformer2DModel.from_pretrained(path, torch_dtype=bf16)
    return Flux2Transformer2DModel.from_single_file(str(path), config=configs, subfolder="transformer",
                                                    torch_dtype=bf16)


def load_vae(files: dict[str, Path]):
    import torch
    from diffusers import AutoencoderKLFlux2

    return AutoencoderKLFlux2.from_pretrained(files["vae"], torch_dtype=torch.bfloat16)


def scheduler(files: dict[str, Path]):
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_pretrained(files["configs"] / "scheduler")


class Flux2KleinGenerator:
    name = "flux2-klein-4b"

    def __init__(self, files: dict[str, Path], steps: int = 4, guidance: float = 1.0):
        self.files = files
        self.steps = steps
        self.guidance = guidance

    def encode_many(self, prompts: list[str], stages: Stages) -> list:
        """Embeddings for every prompt; the text encoder is loaded once, only if some prompt is not cached."""
        import torch

        key = f"{self.name}:{self.files['text_encoder'].name}"
        caches = [_embed_cache(key, p) for p in prompts]
        todo = [i for i, c in enumerate(caches) if not c.exists()]
        if todo:
            from diffusers import Flux2KleinPipeline
            from diffusers.hooks import apply_group_offloading
            from transformers import AutoTokenizer

            with stages.stage("encode"):
                te = _load_text_encoder(self.files["text_encoder"], self.files["configs"])
                try:
                    apply_group_offloading(te, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                           offload_type="leaf_level", use_stream=True)
                except RuntimeError as exc:  # seen on Windows: "CUDA error: resource already mapped" while pinning
                    if "already mapped" not in str(exc):
                        raise
                    torch.cuda.synchronize()
                    te = _load_text_encoder(self.files["text_encoder"], self.files["configs"])
                    apply_group_offloading(te, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                           offload_type="leaf_level", use_stream=False)
                tok = AutoTokenizer.from_pretrained(self.files["configs"] / "tokenizer")
                pipe = Flux2KleinPipeline(scheduler=scheduler(self.files), vae=None, text_encoder=te,
                                          tokenizer=tok, transformer=None, is_distilled=True)
                with torch.no_grad():
                    for i in todo:
                        emb, _ = pipe.encode_prompt(prompts[i], device=torch.device("cuda"),
                                                    max_sequence_length=MAX_TOKENS,
                                                    text_encoder_out_layers=TEXT_LAYERS)
                        caches[i].parent.mkdir(parents=True, exist_ok=True)
                        torch.save(emb.to("cpu"), caches[i])
                del pipe, te, tok
        return [torch.load(c, map_location="cpu", weights_only=True) for c in caches]

    def encode(self, prompt: str, stages: Stages):
        return self.encode_many([prompt], stages)[0]

    supports_reference = True

    def generate(self, prompt: str, seeds: list[int], width: int, height: int, stages: Stages,
                 lora: Path | None = None, lora_scale: float = 1.0, reference: list | None = None) -> list:
        """``reference``: RGB images the model edits/follows (klein's multi-reference editing)."""
        return self.render([(prompt, s, reference) for s in seeds], width, height, stages, lora, lora_scale)

    def render(self, jobs: list[tuple], width: int, height: int, stages: Stages, lora: Path | None = None,
               lora_scale: float = 1.0) -> list:
        """Images for (prompt, seed[, reference images]) jobs with one transformer load."""
        import torch
        from diffusers import Flux2KleinPipeline

        embeds = self.encode_many([j[0] for j in jobs], stages)
        images = []
        with stages.stage("denoise"):
            tr = load_transformer(self.files)
            vae = load_vae(self.files)
            pipe = Flux2KleinPipeline(scheduler=scheduler(self.files), vae=vae, text_encoder=None, tokenizer=None,
                                      transformer=tr, is_distilled=True)
            if lora is not None:
                pipe.load_lora_weights(str(lora), adapter_name=LORA_ADAPTER)
                pipe.set_adapters([LORA_ADAPTER], [lora_scale])
            pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            out = None
            with torch.no_grad():
                for job, emb in zip(jobs, embeds):
                    reference = job[2] if len(job) > 2 else None
                    extra = {"image": list(reference)} if reference else {}
                    out = pipe(prompt_embeds=emb.to("cuda", torch.bfloat16), height=height, width=width, **extra,
                               num_inference_steps=self.steps, guidance_scale=self.guidance,
                               generator=torch.Generator("cuda").manual_seed(job[1]))
                    images.append(out.images[0].convert("RGB"))
            del pipe, tr, vae, out
        return images
