"""Matting: an alpha mask for an RGB image (needs the ``gpu`` extra). BiRefNet runs at 1024x1024."""

from __future__ import annotations

from pathlib import Path

from ggs.pipeline.vram import Stages

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class BiRefNetMatter:
    name = "birefnet"

    def __init__(self, path: Path, size: int = 1024):
        self.path = path
        self.size = size

    def alphas(self, images: list, stages: Stages) -> list:
        """One 'L' mask per image (same size as the image)."""
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoModelForImageSegmentation

        out = []
        with stages.stage("matte"):
            net = AutoModelForImageSegmentation.from_pretrained(str(self.path), trust_remote_code=True)
            # no NVIDIA GPU (a cloud base model on a laptop): the CPU in fp32, slower but the same cut-out
            dev, dtype = ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)
            net.to(dev, dtype).eval()
            mean = torch.tensor(MEAN).view(1, 3, 1, 1)
            std = torch.tensor(STD).view(1, 3, 1, 1)
            with torch.no_grad():
                for img in images:
                    x = np.asarray(img.convert("RGB").resize((self.size, self.size), Image.BICUBIC),
                                   dtype=np.float32) / 255.0
                    t = (torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0) - mean) / std
                    pred = net(t.to(dev, dtype))[-1].sigmoid()[0, 0].float().cpu().numpy()
                    mask = Image.fromarray((pred * 255).round().astype("uint8"), "L")
                    out.append(mask.resize(img.size, Image.BICUBIC))
            del net
        return out
