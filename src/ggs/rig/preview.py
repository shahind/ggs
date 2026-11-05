"""Stick-figure preview of a clip (debug tool, not an exporter). Needs Pillow: pip install -e ".[image]"."""

from __future__ import annotations

from pathlib import Path

from ggs.rig.clip import Clip
from ggs.rig.pose import pose_frame
from ggs.rig.retarget import draw_order_for
from ggs.rig.spec import RigSpec

NEAR = (230, 90, 60, 255)
FAR = (120, 60, 50, 255)
BODY = (60, 110, 200, 255)
BG = (250, 250, 250, 255)
GROUND = (200, 200, 200, 255)


def _draw(spec: RigSpec, clip: Clip, frame: int, cell: int):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (cell, cell), BG)
    d = ImageDraw.Draw(img)
    scale = cell * 0.8
    ox = cell / 2 - (0.0 if clip.in_place else clip.root.x[frame] * scale)
    oy = cell * 0.92
    d.line([(0, oy), (cell, oy)], fill=GROUND, width=1)
    pose = pose_frame(spec, clip, frame)
    swapped = {g: v[frame] for g, v in clip.swaps.items()}
    width = max(2, cell // 40)
    for name in draw_order_for(spec, swapped):
        (x0, y0), (x1, y1) = pose[name]
        color = FAR if name.endswith("_far") else NEAR if name.endswith("_near") else BODY
        pts = [(ox + x0 * scale, oy + y0 * scale), (ox + x1 * scale, oy + y1 * scale)]
        d.line(pts, fill=color, width=width)
        r = width * 0.7
        for px, py in pts:
            d.ellipse([px - r, py - r, px + r, py + r], fill=color)
    d.text((4, 4), str(frame), fill=(0, 0, 0, 255))
    return img


def render_preview(
    spec: RigSpec, clip: Clip, out: str | Path, frames: int = 8, cell: int = 160
) -> Path:
    """``.gif`` -> animated GIF of every frame; anything else -> PNG strip of ``frames`` evenly spaced frames."""
    from PIL import Image

    out = Path(out)
    if out.suffix.lower() == ".gif":
        imgs = [_draw(spec, clip, i, cell).convert("RGB") for i in range(clip.frame_count)]
        imgs[0].save(
            out, save_all=True, append_images=imgs[1:], duration=round(1000 / clip.fps), loop=0
        )
        return out
    k = max(1, min(frames, clip.frame_count))
    picks = [round(i * (clip.frame_count - 1) / max(1, k - 1)) for i in range(k)]
    strip = Image.new("RGBA", (cell * k, cell), BG)
    for n, i in enumerate(picks):
        strip.paste(_draw(spec, clip, i, cell), (n * cell, 0))
    strip.save(out)
    return out
