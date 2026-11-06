"""Pose-driven frames: One-to-All Animation 1.3B (CVPR 2026, Apache-2.0) on Wan2.1-T2V-1.3B (diffusers).

The character's design is the reference image, the clip's skeleton is drawn one pose image per frame in
One-to-All's own pose style (OpenPose-18 colours), and the model draws the design in those poses. It is
"alignment-free": the reference and the poses may differ in layout and proportions, and it was trained with
cartoon data.

Stages, one heavy model on the GPU at a time:
- ``encode``: the umT5-xxl text encoder (fp16 file, kept in RAM) is streamed onto the GPU layer by layer. The
  result is cached per prompt (``~/.ggs/cache/one_to_all``), so a character's later runs skip this stage.
- ``animate``: the One-to-All transformer (about 2B parameters, bf16) and the Wan VAE on the GPU; three
  transformer passes per step (reference + pose guidance), 30 steps.
The vendored inference code is ``ggs.third_party.one_to_all``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
from pathlib import Path

from ggs.pipeline.vram import Stages

NEGATIVE = ("black background, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, "
            "bad teeth, bad eyes, bad limbs, distortion, extra objects, backpack, bag, props")
MAX_SEQ = 512

# One-to-All's pose drawing (opensora/dataset/utils.py: draw_bodypose_aligned), body only
LIMBS = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
         [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
COLOURS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def _stickwidth(w: int, h: int, base: int = 3) -> int:
    m = max(w, h)
    ratio = 1.0 if m < 512 else 1.5 if m < 1080 else 2.0 if m < 2160 else 2.5 if m < 3240 else 3.5 if m < 4320 \
        else 4.5 if m < 5400 else 4.0
    return int(base * ratio)


def draw_pose(points: list, size: tuple[int, int]):
    """An 18-point OpenPose skeleton (pixels, ``None`` = not shown) drawn the way One-to-All was trained: on a
    canvas whose short side is 2160 px, then scaled to ``size`` (w, h). RGB PIL image."""
    import cv2
    import numpy as np
    from PIL import Image

    w, h = size
    sr = 2160 / min(w, h)
    cw, ch = int(w * sr), int(h * sr)
    canvas = np.zeros((ch, cw, 3), np.uint8)
    sw = _stickwidth(cw, ch, 3)
    pts = [None if p is None else (p[0] * sr, p[1] * sr) for p in points]
    for i, (a, b) in enumerate(LIMBS):
        pa, pb = pts[a - 1], pts[b - 1]
        if pa is None or pb is None:
            continue
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        length = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        angle = math.degrees(math.atan2(pa[1] - pb[1], pa[0] - pb[0]))
        poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), sw), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, COLOURS[i])
    canvas = (canvas * 0.6).astype(np.uint8)
    for i, p in enumerate(pts):
        if p is not None:
            cv2.circle(canvas, (int(p[0]), int(p[1])), sw, COLOURS[i], thickness=-1)
    return Image.fromarray(cv2.resize(canvas, (w, h)))


def cache_dir() -> Path:
    return Path.home() / ".ggs" / "cache" / "one_to_all"


class OneToAllFrames:
    def __init__(self, files: dict[str, Path], steps: int = 30, ref_cfg: float = 2.5, pose_cfg: float = 1.5):
        self.files, self.steps, self.ref_cfg, self.pose_cfg = files, steps, ref_cfg, pose_cfg

    # ------------------------------------------------------------------ text
    def embeds(self, prompts: list[str], stages: Stages) -> list:
        """bf16 (1, 512, 4096) umT5 embeddings, from the cache or computed (streamed) and cached."""
        import torch

        from ggs.third_party.one_to_all.pipeline import prompt_clean

        out: dict[str, object] = {}
        todo = []
        for p in prompts:
            f = cache_dir() / f"{hashlib.sha1(prompt_clean(p).encode('utf-8')).hexdigest()}.pt"
            if f.exists():
                out[p] = torch.load(f, map_location="cpu", weights_only=True)
            else:
                todo.append((p, f))
        if todo:
            with stages.stage("encode"):
                enc, tok = self._load_text_encoder()
                cuda = torch.cuda.is_available()
                if cuda:
                    from diffusers.hooks import apply_group_offloading

                    apply_group_offloading(enc, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                           offload_type="leaf_level", use_stream=False)
                dev = "cuda" if cuda else "cpu"
                cache_dir().mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    for p, f in todo:
                        t = tok([prompt_clean(p)], padding="max_length", max_length=MAX_SEQ, truncation=True,
                                add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
                        n = int(t.attention_mask.gt(0).sum())
                        e = enc(t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state[0, :n]
                        e = torch.cat([e, e.new_zeros(MAX_SEQ - n, e.shape[1])]).unsqueeze(0)
                        e = e.to("cpu", torch.bfloat16)
                        torch.save(e, f)
                        out[p] = e
                del enc
        return [out[p] for p in prompts]

    def _load_text_encoder(self):
        import torch
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

        base = self.files["base"]
        cfg = UMT5Config.from_pretrained(base / "text_encoder")
        with init_empty_weights():
            enc = UMT5EncoderModel(cfg)
        sd = {k: v.to(torch.bfloat16) for k, v in load_file(str(self.files["text_encoder"])).items()}
        if "encoder.embed_tokens.weight" not in sd and "shared.weight" in sd:
            sd["encoder.embed_tokens.weight"] = sd["shared.weight"]
        missing, _unexpected = enc.load_state_dict(sd, strict=False, assign=True)
        if missing:
            raise RuntimeError(f"text encoder file is missing weights: {missing[:5]}")
        enc.eval().requires_grad_(False)
        return enc, AutoTokenizer.from_pretrained(base / "tokenizer")

    # ------------------------------------------------------------------ frames
    def generate(self, poses: list, reference, ref_pose, prompt: str, seed: int, width: int, height: int,
                 stages: Stages) -> list:
        """RGB frames (PIL) for ``poses`` (PIL pose images, 4k+1 of them), drawn like ``reference`` (PIL RGB on
        white, the character in the pose ``ref_pose``)."""
        import numpy as np
        import torch
        from diffusers import AutoencoderKLWan
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from PIL import Image
        from safetensors.torch import load_file

        from ggs.third_party.one_to_all import __file__ as pkg
        from ggs.third_party.one_to_all.pipeline import WanPipeline
        from ggs.third_party.one_to_all.transformer import WanTransformer3DModel

        if (len(poses) - 1) % 4:
            raise ValueError("One-to-All draws 4k+1 frames")
        pos, neg = self.embeds([prompt, NEGATIVE], stages)
        cfg = Path(pkg).parent / "configs"
        dt = torch.bfloat16
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        with stages.stage("animate"):
            model = WanTransformer3DModel.from_config(WanTransformer3DModel.load_config(cfg / "wan2.1_t2v_1.3b.json"))
            model = model.to(dt)
            model.set_up_controlnet(str(cfg / "wan2.1_t2v_1.3b_controlnet_2.json"), dt)
            model.set_up_refextractor(str(cfg / "wan2.1_t2v_1.3b_refextractor_2d_withmask2.json"), dt)
            sd = {}
            for f in sorted(Path(self.files["model"]).glob("*.safetensors")):
                sd.update(load_file(str(f)))
            model.load_state_dict(sd, strict=True)
            del sd
            model = model.to(dt).eval().requires_grad_(False)
            vae = AutoencoderKLWan.from_pretrained(self.files["base"] / "vae", torch_dtype=dt)
            sched = FlowMatchEulerDiscreteScheduler(shift=7.0, num_train_timesteps=1000, use_dynamic_shifting=False)
            pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=model, vae=vae, scheduler=sched)
            pipe.to(dev, dtype=dt)
            pipe.set_progress_bar_config(disable=True)

            def to_t(img):
                a = np.asarray(img.convert("RGB").resize((width, height)), np.float32) / 127.5 - 1.0
                return torch.from_numpy(a).permute(2, 0, 1)

            control = torch.stack([to_t(p) for p in poses], dim=1).unsqueeze(0)  # (1, 3, T, H, W)
            src_pose = to_t(ref_pose).unsqueeze(0).unsqueeze(2)
            mask = torch.zeros(1, 1, 1, height, width)
            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):  # the pipeline prints every step
                out = pipe(image=reference.convert("RGB").resize((width, height)), image_mask=mask,
                           control_video=control, prompt=None, prompt_embeds=pos.to(dev),
                           negative_prompt_embeds=neg.to(dev), height=height, width=width, num_frames=len(poses),
                           image_guidance_scale=self.ref_cfg, pose_guidance_scale=self.pose_cfg,
                           num_inference_steps=self.steps, generator=torch.Generator(device=dev).manual_seed(seed),
                           black_image_cfg=True, black_pose_cfg=True, controlnet_conditioning_scale=1.0,
                           return_tensor=True, case1=False, token_replace=False, prev_frames=None,
                           image_pose=src_pose).frames
            video = (out[0].float().cpu() / 2 + 0.5).clamp(0, 1).permute(1, 2, 3, 0).numpy()
            frames = [Image.fromarray((v * 255).round().astype(np.uint8)) for v in video]
            del pipe, model, vae, out
        return frames
