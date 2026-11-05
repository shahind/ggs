"""Animated image export: APNG and WebP keep full alpha; GIF needs a matte colour (1-bit alpha, 256 colours)."""

from __future__ import annotations

from pathlib import Path

from ggs.rig.puppet import frame_durations_ms


class ExportError(ValueError):
    pass


def parse_color(text: str) -> tuple[int, int, int]:
    t = text.strip().lstrip("#")
    if len(t) != 6:
        raise ExportError(f"colour must be #RRGGBB, got {text!r}")
    try:
        return int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)
    except ValueError as exc:
        raise ExportError(f"colour must be #RRGGBB, got {text!r}") from exc


def save_apng(frames: list, path: str | Path, fps: float, loop: bool = True) -> Path:
    path = Path(path)
    frames[0].save(path, format="PNG", save_all=True, append_images=frames[1:],
                   duration=frame_durations_ms(len(frames), fps), loop=0 if loop else 1,
                   disposal=1, blend=0)
    return path


def save_webp(frames: list, path: str | Path, fps: float, loop: bool = True, lossless: bool = True) -> Path:
    path = Path(path)
    frames[0].save(path, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=frame_durations_ms(len(frames), fps), loop=0 if loop else 1,
                   lossless=lossless, quality=100 if lossless else 90, method=4, exact=True)
    return path


def save_gif(frames: list, path: str | Path, fps: float, matte: tuple[int, int, int],
             loop: bool = True, alpha_cutoff: int = 1) -> Path:
    """Soft edges are flattened onto ``matte``; pixels with alpha < ``alpha_cutoff`` become transparent.
    Pick a matte close to the game's background."""
    from PIL import Image

    path = Path(path)
    flat, masks = [], []
    for im in frames:
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (*matte, 255))
        flat.append(Image.alpha_composite(bg, im).convert("RGB"))
        masks.append(im.getchannel("A").point(lambda a: 255 if a < alpha_cutoff else 0))
    w, h = flat[0].size
    combined = Image.new("RGB", (w, h * len(flat)))
    for i, im in enumerate(flat):
        combined.paste(im, (0, i * h))
    palette_img = combined.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = palette_img.getpalette()[: 255 * 3]
    pal += [*matte]  # index 255 = transparent
    palette_img.putpalette(pal)
    out = []
    for im, mask in zip(flat, masks):
        q = im.quantize(palette=palette_img, dither=Image.Dither.NONE)
        q.paste(255, mask=mask)
        out.append(q)
    extra = {"loop": 0} if loop else {}
    out[0].save(path, format="GIF", save_all=True, append_images=out[1:],
                duration=frame_durations_ms(len(out), fps), transparency=255, disposal=2,
                optimize=False, **extra)
    return path
