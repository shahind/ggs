"""Body parsing: which pixels are face, hair, hat, skin of the arms and legs, shoes, clothes (SegFormer-B2 trained
on ATR "clothes" labels, ~110 MB, MIT).

It reads soft-shaded and 3D-rendered characters well, where outlines and flat colour areas do not exist, and
flat cartoon art mostly well. It is used two ways in ``ggs.rig.joints.region_labels``:
- trusted classes decide their part directly: face, hair, hat and sunglasses go with the head (the neck is part
  of the face), shoes with the nearest foot, arm and leg skin with the nearest arm or leg, cut at the joints;
- clothing classes only split colour areas further (shirt from shorts); the skeleton still decides who owns the
  clothing, so a costume the model misreads cannot move pixels to the wrong limb.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PARSE_MODEL = "segformer-b2-clothes"
BACKGROUND, HAT, HAIR, GLASSES, UPPER, SKIRT, PANTS, DRESS, BELT = range(9)
L_SHOE, R_SHOE, FACE, L_LEG, R_LEG, L_ARM, R_ARM, BAG, SCARF = range(9, 18)
HEAD_CLASSES = (HAT, HAIR, GLASSES, FACE)
SHOES = (L_SHOE, R_SHOE)
ARM_SKIN = (L_ARM, R_ARM)
LEG_SKIN = (L_LEG, R_LEG)


def parse_bodies(images: list, files: dict[str, Path] | None = None, stages=None) -> list[np.ndarray]:
    """Per image (RGBA), an int8 array of class ids (0 = background; transparent pixels are background)."""
    import torch
    from PIL import Image
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

    from ggs.pipeline.models import resolve_pose
    from ggs.pipeline.vram import Stages

    src = str(files["parse"]) if files and files.get("parse") else str(resolve_pose(PARSE_MODEL))
    stages = stages or Stages(enforce=False)
    out = []
    with stages.stage("parse"):
        proc = SegformerImageProcessor.from_pretrained(src)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSemanticSegmentation.from_pretrained(src).to(dev).eval()
        for rgba in images:
            rgba = rgba.convert("RGBA")
            im = Image.new("RGB", rgba.size, "white")
            im.paste(rgba, mask=rgba.getchannel("A"))
            inp = proc(images=im, return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = model(**inp).logits
            up = torch.nn.functional.interpolate(logits, size=im.size[::-1], mode="bilinear", align_corners=False)
            seg = up.argmax(1)[0].cpu().numpy().astype(np.int8)
            seg[np.asarray(rgba.getchannel("A")) < 128] = BACKGROUND
            out.append(seg)
        del model, proc
    return out


def save_parse(seg: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(seg.astype(np.uint8), "L").save(path)


def load_parse(path: Path) -> np.ndarray | None:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")).astype(np.int8) if Path(path).exists() else None
