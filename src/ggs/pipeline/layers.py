"""Layer decomposition with See-through (SIGGRAPH 2026): one character picture -> complete RGBA layers.

The layer model (LayerDiffuse "3D" on SDXL, 4-bit NF4 weights, ``ggs.third_party.see_through``) draws every
layer whole, including what is hidden behind other layers: the shirt under the arms, the trousers under the shirt
hem, the head under the hair. Two passes, as See-through does: the body tags on the whole picture, then the face
tags on a crop around the head. The depth model See-through uses for drawing order is not needed: the rig decides
the order.

About 6–7 minutes at 768 px on the 8 GB laptop, 4.6 GB peak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ggs.pipeline.vram import Stages

BODY_TAGS = ["front hair", "back hair", "head", "neck", "neckwear", "topwear", "handwear", "bottomwear", "legwear",
             "footwear", "tail", "wings", "objects"]
HEAD_TAGS = ["headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash", "eyewear", "ears", "earwear", "nose",
             "mouth"]


class SeeThroughLayers:
    def __init__(self, files: dict[str, Path], resolution: int = 768, steps: int = 30):
        self.files, self.resolution, self.steps = files, resolution, steps

    def _pipeline(self):
        import torch

        from ggs.third_party.see_through.layerdiffuse.diffusers_kdiffusion_sdxl import (
            KDiffusionStableDiffusionXLPipeline,
        )
        from ggs.third_party.see_through.layerdiffuse.layerdiff3d import UNetFrameConditionModel
        from ggs.third_party.see_through.layerdiffuse.vae import TransparentVAE

        repo = str(self.files["model"])
        unet = UNetFrameConditionModel.from_pretrained(repo, subfolder="unet")
        trans_vae = TransparentVAE.from_pretrained(repo, subfolder="trans_vae")
        pipe = KDiffusionStableDiffusionXLPipeline.from_pretrained(repo, trans_vae=trans_vae, unet=unet,
                                                                   scheduler=None)
        pipe.vae.to(dtype=torch.bfloat16, device="cuda")
        pipe.trans_vae.to(dtype=torch.bfloat16, device="cuda")
        pipe.cache_tag_embeds()  # the tag prompts once; the text encoders are unloaded
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def decompose(self, image, stages: Stages, seed: int = 42) -> dict[str, np.ndarray]:
        """``image``: PIL RGBA (the character on transparency). Returns tag -> RGBA uint8 array the size of the
        image; empty layers are left out."""
        import cv2
        import torch

        from ggs.third_party.see_through.helpers import center_square_pad_resize, smart_resize

        res = self.resolution
        src = np.asarray(image.convert("RGBA"))
        h0, w0 = src.shape[:2]
        with stages.stage("layers"):
            pipe = self._pipeline()
            full, pad_size, pad_pos = center_square_pad_resize(src, res, return_pad_info=True)
            scale = pad_size[0] / res
            rng = torch.Generator(device=pipe.unet.device).manual_seed(seed)
            body = pipe(strength=1.0, num_inference_steps=self.steps, batch_size=1, generator=rng, guidance_scale=1.0,
                        prompt=BODY_TAGS, negative_prompt="", fullpage=full, group_index=0).images
            layers = dict(zip(BODY_TAGS, body, strict=True))
            head = layers["head"]
            found = cv2.findNonZero((head[..., -1] > 15).astype(np.uint8))
            if found is not None:  # the face pass on a crop around the head (See-through's run_layerdiff)
                hx0, hy0, hw, hh = cv2.boundingRect(found)
                hx, hy = int(hx0 * scale) - pad_pos[0], int(hy0 * scale) - pad_pos[1]
                hw, hh = int(hw * scale), int(hh * scale)
                crop, (x1, y1, _x2, _y2) = _crop_head(src, (hx, hy, hw, hh))
                hx1, hy1 = int(x1 / scale + pad_pos[0] / scale), int(y1 / scale + pad_pos[1] / scale)
                ih, iw = crop.shape[:2]
                crop_sq, c_pad_size, c_pad_pos = center_square_pad_resize(crop, res, return_pad_info=True)
                out = pipe(strength=1.0, num_inference_steps=self.steps, batch_size=1, generator=rng,
                           guidance_scale=1.0, prompt=HEAD_TAGS, negative_prompt="", fullpage=crop_sq,
                           group_index=1).images
                # the crop (in picture pixels) back into the body pass's square, at the body pass's scale
                py1, py2, px1, px2 = (np.array([c_pad_pos[1], c_pad_pos[1] + ih, c_pad_pos[0], c_pad_pos[0] + iw])
                                      / scale).astype(np.int64)
                size = (int(c_pad_size[0] / scale), int(c_pad_size[1] / scale))
                for rst, tag in zip(out, HEAD_TAGS, strict=True):
                    rst = smart_resize(rst, size)[py1:py2, px1:px2]
                    canvas = np.zeros((res, res, 4), np.uint8)
                    h_, w_ = min(rst.shape[0], res - hy1), min(rst.shape[1], res - hx1)
                    canvas[hy1:hy1 + h_, hx1:hx1 + w_] = rst[:h_, :w_]
                    layers[tag] = canvas
            del pipe
        # back from the padded square to the picture's own pixels
        sz = max(h0, w0)
        px, py = (sz - w0) // 2, (sz - h0) // 2
        result = {}
        for tag, lay in layers.items():
            if lay[..., 3].max() < 16:
                continue
            big = cv2.resize(lay, (sz, sz), interpolation=cv2.INTER_LINEAR if sz > res else cv2.INTER_AREA)
            result[tag] = np.ascontiguousarray(big[py:py + h0, px:px + w0])
        return result


def _crop_head(img: np.ndarray, xywh):
    """See-through's head crop: the head box grown by a fifth on each side (inside the picture)."""
    x, y, w, h = xywh
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = x, y, x + w, y + h
    if w < iw // 2:
        px = min(iw - x - w, x, w // 5)
        x1, x2 = min(max(x - px, 0), iw), min(max(x + w + px, 0), iw)
    if h < ih // 2:
        py = min(ih - y - h, y, h // 5)
        y2, y1 = min(max(y + h + py, 0), ih), min(max(y - py, 0), ih)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def save_layers(layers: dict[str, np.ndarray], folder: Path) -> None:
    from PIL import Image

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for tag, arr in layers.items():
        Image.fromarray(arr, "RGBA").save(folder / f"{tag.replace(' ', '_')}.png")


def load_layers(folder: Path) -> dict[str, np.ndarray]:
    from PIL import Image

    folder = Path(folder)
    if not folder.is_dir():
        return {}
    return {f.stem.replace("_", " "): np.asarray(Image.open(f).convert("RGBA")) for f in sorted(folder.glob("*.png"))}
