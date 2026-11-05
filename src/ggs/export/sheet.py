"""Sprite sheet + JSON atlas (TexturePacker "JSON hash" layout plus an ``animations`` table and a ``ggs`` meta
block). See docs/EXPORT_FORMATS.md."""

from __future__ import annotations

import json
from pathlib import Path

from ggs import __version__
from ggs.export.pack import grid_layout, paste_extruded
from ggs.rig.puppet import frame_durations_ms


def save_sprite_sheet(frames: list, anchor: tuple[int, int], png_path: str | Path, name: str,
                      fps: float, loop: bool, padding: int = 2, extrude: int = 1) -> tuple[Path, Path]:
    """Uniform frames in a power-of-two grid. Writes ``<png>`` and ``<png stem>.json``."""
    from PIL import Image

    if extrude > padding:
        raise ValueError("extrude must not exceed padding")
    png_path = Path(png_path)
    json_path = png_path.with_suffix(".json")
    w, h = frames[0].size
    cols, _, sw, sh = grid_layout(len(frames), w, h, padding)
    sheet = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    durations = frame_durations_ms(len(frames), fps)
    table = {}
    names = []
    for i, im in enumerate(frames):
        col, row = i % cols, i // cols
        x, y = col * (w + 2 * padding) + padding, row * (h + 2 * padding) + padding
        paste_extruded(sheet, im, (x, y), extrude)
        key = f"{name}_{i:03d}"
        names.append(key)
        table[key] = {
            "frame": {"x": x, "y": y, "w": w, "h": h},
            "rotated": False,
            "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": w, "h": h},
            "sourceSize": {"w": w, "h": h},
            "pivot": {"x": round(anchor[0] / w, 6), "y": round(anchor[1] / h, 6)},
            "duration": durations[i],
        }
    sheet.save(png_path)
    atlas = {
        "frames": table,
        "animations": {name: names},
        "meta": {
            "app": "ggs",
            "version": __version__,
            "image": png_path.name,
            "format": "RGBA8888",
            "size": {"w": sw, "h": sh},
            "scale": "1",
            "ggs": {
                "format": "ggs.atlas",
                "version": 1,
                "padding": padding,
                "extrude": extrude,
                "anchor_px": {"x": anchor[0], "y": anchor[1]},
                "animations": {name: {"fps": fps, "loop": loop, "frames": len(frames)}},
            },
        },
    }
    json_path.write_text(json.dumps(atlas, indent=1), encoding="utf-8")
    return png_path, json_path
