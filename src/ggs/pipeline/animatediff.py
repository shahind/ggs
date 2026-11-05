"""Pose-driven frames: AnimateDiff + ControlNet OpenPose + IP-Adapter on Stable Diffusion 1.5 (diffusers).

The character's design goes in through IP-Adapter (so every frame keeps its look), the animation's skeleton as
one OpenPose image per frame (so every frame follows the clip), and AnimateDiff's motion module keeps the frames
consistent with each other. 16 frames at 384x512 take about 75 s on the 8 GB laptop with a 6.4 GB peak (model
CPU offload: one sub-model on the GPU at a time).
"""

from __future__ import annotations

from pathlib import Path

from ggs.pipeline.vram import Stages

NEGATIVE = "blurry, deformed, extra limbs, extra fingers, background scenery, text, watermark, frame, border"


class AnimateDiffFrames:
    supports_reference = True

    def __init__(self, files: dict[str, Path], steps: int = 25, ip_scale: float = 0.7, pose_scale: float = 1.0):
        self.files, self.steps, self.ip_scale, self.pose_scale = files, steps, ip_scale, pose_scale

    def generate(self, poses: list, reference, prompt: str, seed: int, width: int, height: int,
                 stages: Stages) -> list:
        """One RGB frame per OpenPose image in ``poses`` (at most 16 per call)."""
        import torch
        from diffusers import (
            AnimateDiffControlNetPipeline,
            ControlNetModel,
            DDIMScheduler,
            MotionAdapter,
        )
        from transformers import CLIPVisionModelWithProjection

        if not 1 <= len(poses) <= 16:
            raise ValueError("draw between 1 and 16 frames at a time")
        f = self.files
        dt = torch.float16
        with stages.stage("animate"):
            adapter = MotionAdapter.from_pretrained(f["motion"], torch_dtype=dt, variant="fp16")
            controlnet = ControlNetModel.from_pretrained(f["controlnet"], torch_dtype=dt, variant="fp16")
            encoder = CLIPVisionModelWithProjection.from_pretrained(f["ip_adapter"] / "models" / "image_encoder",
                                                                    torch_dtype=dt)
            pipe = AnimateDiffControlNetPipeline.from_pretrained(f["base"], motion_adapter=adapter,
                                                                 controlnet=controlnet, image_encoder=encoder,
                                                                 torch_dtype=dt, variant="fp16")
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear",
                                                       clip_sample=False, timestep_spacing="linspace", steps_offset=1)
            pipe.load_ip_adapter(str(f["ip_adapter"]), subfolder="models",
                                 weight_name="ip-adapter-plus_sd15.safetensors")
            pipe.set_ip_adapter_scale(self.ip_scale)
            pipe.vae.enable_slicing()
            pipe.enable_model_cpu_offload()
            pipe.set_progress_bar_config(disable=True)
            res = pipe(prompt=prompt, negative_prompt=NEGATIVE, num_frames=len(poses), width=width, height=height,
                       conditioning_frames=poses, ip_adapter_image=reference, num_inference_steps=self.steps,
                       guidance_scale=7.5, controlnet_conditioning_scale=self.pose_scale,
                       generator=torch.Generator("cpu").manual_seed(seed), decode_chunk_size=4)
            frames = list(res.frames[0])
            pipe.remove_all_hooks()
            del pipe, adapter, controlnet, encoder, res
        return frames
