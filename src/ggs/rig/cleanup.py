"""Cut-out animation, cleaned by a model: the puppet draws every frame exactly (pose, shape, colours), and SDXL
repaints only the places where the pieces do not meet: gaps inside the outline and the seams between pieces.

Like real-time neural rendering, the motion and the design come from the renderer, so the model only fills
small areas. Everything outside the mask stays the drawn design pixel for pixel.

    frames = cutout_frames(character_yaml, clip, n)       # RGBA frames + where to repaint
    clean = clean_frames(frames, prompt, files, stages)    # SDXL inpainting (Lightning, ~6 steps)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ggs.pipeline.vram import Stages

GAP_CLOSE = 0.04  # holes and notches narrower than this (x standing height) are gaps between pieces
JOINT_CLOSE = 0.12  # near a joint, notches up to this wide are closed too
JOINT_REACH = 0.07  # how far from a joint that applies
JOINT_RADIUS = 0.028  # repaint disc at each joint (x standing height)
WORK_HEIGHT = 768  # px per standing height while repainting (the frames are scaled back after)
STRENGTH = 0.6  # how much of a masked area is redrawn (1 = from scratch)


@dataclass
class CutFrame:
    rgba: object  # PIL RGBA, the puppet's frame
    mask: np.ndarray  # bool: repaint here (gaps and seams)
    solid: np.ndarray  # bool: the character's filled outline (gaps closed)

    @property
    def gaps(self) -> np.ndarray:
        return self.solid & (np.asarray(self.rgba.getchannel("A")) < 128)


def frame_masks(layers: list[np.ndarray], height_px: float, joints: list[tuple[float, float]] | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(alpha union, filled outline, repaint mask) for one frame's piece layers (uint8 alpha arrays). The gaps
    are the filled outline minus the alpha union.

    - gaps: inside the outline once holes and notches are closed, but covered by no piece (a bad cut);
    - joints: a small disc where two pieces meet (``joints`` in frame pixels), inside the outline; the cuts
      show there. Drawn outlines elsewhere (an arm over the shirt) are left as they are."""
    import cv2

    union = np.zeros(layers[0].shape, np.uint8)
    for a in layers:
        union = np.maximum(union, a)
    on = union >= 128
    k = max(3, round(GAP_CLOSE * height_px) | 1)
    solid = cv2.morphologyEx(on.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                            (k, k))).astype(bool)
    solid |= on
    # enclosed holes (background not connected to the frame's edge) are gaps whatever their size
    from scipy import ndimage

    bg, _n = ndimage.label(~on)
    edge = np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]))
    holes = (bg > 0) & ~np.isin(bg, edge)
    solid |= holes
    if joints:  # near a joint a wider notch is still a bad cut (the back of a shoulder): close it there only
        kj = max(3, round(JOINT_CLOSE * height_px) | 1)
        near = np.zeros(on.shape, np.uint8)
        for x, y in joints:
            cv2.circle(near, (round(x), round(y)), max(3, round(JOINT_REACH * height_px)), 1, -1)
        wide = cv2.morphologyEx(on.astype(np.uint8), cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kj, kj))).astype(bool)
        solid |= wide & near.astype(bool)
    gaps = solid & ~on
    grow = max(2, round(0.01 * height_px))
    mask = cv2.dilate(gaps.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                       (2 * grow + 1, 2 * grow + 1)))
    r = max(3, round(JOINT_RADIUS * height_px))
    for x, y in joints or []:
        cv2.circle(mask, (round(x), round(y)), r, 1, -1)
    return on, solid, mask.astype(bool) & solid


def cutout_frames(character_yaml: Path, clip, n: int | None = None, height_px: int = 512) -> tuple[list, tuple]:
    """The sliced character playing ``clip``: ``n`` evenly spaced frames (all when None), cropped to one box,
    each with its repaint mask. Returns (frames, anchor)."""
    from PIL import Image

    from ggs.rig.character import load_character
    from ggs.rig.sprites import frame_indices

    ch = load_character(character_yaml)
    rig = ch.rig()
    clip = ch.clip(clip)
    puppet = ch.puppet(rig)
    w, h = int(2.0 * height_px), int(1.7 * height_px)
    origin = (w / 2, 1.35 * height_px)
    idx = list(range(clip.frame_count)) if not n or n >= clip.frame_count else frame_indices(clip, n)
    raw = []
    from ggs.rig.pose import pose_bones, root_scale

    for f in idx:
        layers = [np.asarray(im.getchannel("A")) for _, im in puppet.render_layers(clip, f, height_px, (w, h), origin)]
        frame = puppet.render_frame(clip, f, height_px, (w, h), origin)
        pose = pose_bones(puppet.bones, clip, f, puppet.lengths, root_scale(puppet.bones, clip, puppet.lengths), True)
        joints = [(origin[0] + s_[0] * height_px, origin[1] + s_[1] * height_px)
                  for b, (s_, _e) in pose.items() if b != puppet.bones[0].name and not b.startswith("hand")]
        # (no disc at the wrists: the hand overlaps its forearm anyway, and a fist in front of the face would be
        # repainted with the face)
        _on, solid, mask = frame_masks(layers, height_px, joints)
        raw.append((frame, mask, solid))
    box = None
    for frame, _m, solid in raw:
        bb = Image.fromarray((solid * 255).astype(np.uint8)).getbbox()
        if bb:
            box = bb if box is None else (min(box[0], bb[0]), min(box[1], bb[1]), max(box[2], bb[2]), max(box[3], bb[3]))
    if box is None:
        raise ValueError("every frame is empty")
    pad = 6
    x0, y0, x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad), min(w, box[2] + pad), min(h, box[3] + pad)
    frames = [CutFrame(fr.crop((x0, y0, x1, y1)), m[y0:y1, x0:x1], s[y0:y1, x0:x1]) for fr, m, s in raw]
    return frames, (round(origin[0] - x0), box[3] - y0)


def clean_frames(frames: list[CutFrame], prompt: str, files: dict[str, Path], stages: Stages, seed: int = 7,
                 strength: float = STRENGTH, height_px: int = 512) -> list:
    """Each frame with its masked areas repainted by SDXL (on white, at WORK_HEIGHT); RGBA PIL frames."""
    import cv2
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline
    from PIL import Image

    from ggs.pipeline.sdxl import SPEED_ADAPTER, load_unet, load_vae, scheduler

    gen_embeds = _embeds(prompt, files, stages)
    k = WORK_HEIGHT / height_px
    out = []
    with stages.stage("denoise"):
        unet, vae = load_unet(files), load_vae(files)
        speed = files.get("speed_lora")
        pipe = StableDiffusionXLInpaintPipeline(vae=vae, text_encoder=None, text_encoder_2=None, tokenizer=None,
                                                tokenizer_2=None, unet=unet, scheduler=scheduler(files, bool(speed)))
        if speed:
            pipe.load_lora_weights(str(speed), adapter_name=SPEED_ADAPTER)
            pipe.set_adapters([SPEED_ADAPTER], [1.0])
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        bf16 = torch.bfloat16
        steps = 8 if speed else 30
        for fr in frames:
            if not fr.mask.any():
                out.append(fr.rgba)
                continue
            w0, h0 = fr.rgba.size
            W, H = (round(w0 * k / 8) * 8), (round(h0 * k / 8) * 8)
            white = Image.new("RGB", (w0, h0), "white")
            white.paste(fr.rgba, mask=fr.rgba.getchannel("A"))
            gaps = fr.gaps
            if gaps.any():  # gaps start from the colours around them (not white), so the repaint closes them
                arr = np.asarray(white).copy()
                arr = cv2.inpaint(arr, gaps.astype(np.uint8), 5, cv2.INPAINT_TELEA)
                white = Image.fromarray(arr)
            img = white.resize((W, H), Image.LANCZOS)
            m = Image.fromarray((fr.mask * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
            with torch.no_grad():
                res = pipe(prompt_embeds=gen_embeds["prompt_embeds"].to("cuda", bf16),
                           pooled_prompt_embeds=gen_embeds["pooled"].to("cuda", bf16),
                           negative_prompt_embeds=gen_embeds["negative_embeds"].to("cuda", bf16),
                           negative_pooled_prompt_embeds=gen_embeds["negative_pooled"].to("cuda", bf16),
                           image=img, mask_image=m, width=W, height=H, strength=strength,
                           num_inference_steps=steps, guidance_scale=0.0 if speed else 5.0,
                           generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            res = np.asarray(res.convert("RGB").resize((w0, h0), Image.LANCZOS)).astype(np.float32)
            base = np.asarray(fr.rgba.convert("RGBA")).astype(np.float32)
            soft = cv2.GaussianBlur(fr.mask.astype(np.float32), (0, 0), 1.2)[..., None]  # feathered seam edges
            on_white = np.asarray(white).astype(np.float32)
            rgb = on_white * (1 - soft) + res * soft
            alpha = np.maximum(base[..., 3], np.where(fr.solid & fr.mask, 255.0, 0.0))
            a = alpha[..., None] / 255.0
            # straight colours: where the frame was see-through but is now filled, the repaint's colour
            rgb = np.where(a > 0, np.clip((rgb - 255.0 * (1 - a)) / np.maximum(a, 1e-3), 0, 255), 0)
            out.append(Image.fromarray(np.dstack([rgb, alpha]).round().astype(np.uint8), "RGBA"))
        del pipe, unet, vae
    return out


def _embeds(prompt: str, files: dict[str, Path], stages: Stages) -> dict:
    from ggs.pipeline.sdxl import SdxlGenerator

    neg = "blurry, smudge, extra limbs, deformed, gradient, shading noise"
    return SdxlGenerator(files, negative=neg).encode(prompt, stages)
