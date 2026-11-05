"""Character = a rig spec + a part sheet image + the character's own bone lengths.

``character.yaml`` sits next to its part sheet PNG:

    format: ggs.character
    version: 1
    name: programmer_art
    rig: humanoid_side_v1
    sheet: parts.png          # part sheet laid out by the rig's part_sheet grid
    px_per_height: 600        # the parts are drawn at this many pixels per standing height
    bone_lengths: {pelvis: 0.06, torso: 0.30, ...}   # standing heights; drawn length = this * px_per_height
    pivots: {torso: [0.5, 0.7], ...}   # optional: per-part joint position in its cell (default: the rig's)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from ggs.rig.pose import bone_defs, bone_vector
from ggs.rig.puppet import PartImage, Puppet
from ggs.rig.retarget import limb_groups
from ggs.rig.spec import RigSpec


class CharacterSpec(BaseModel):
    format: Literal["ggs.character"] = "ggs.character"
    version: Literal[1] = 1
    name: str
    rig: str
    sheet: str
    px_per_height: float = Field(gt=0)
    bone_lengths: dict[str, float]
    pivots: dict[str, tuple[float, float]] = Field(default_factory=dict)  # part name -> cell fraction
    stature: float = Field(1.0, gt=0, le=10)  # in-game height relative to other characters (export height x this)


class Character:
    def __init__(self, spec: CharacterSpec, base_dir: Path) -> None:
        self.spec = spec
        self.base_dir = base_dir

    @property
    def sheet_path(self) -> Path:
        return self.base_dir / self.spec.sheet

    def check(self, rig: RigSpec) -> None:
        if self.spec.rig != rig.id:
            raise ValueError(f"character {self.spec.name!r} is for rig {self.spec.rig!r}, not {rig.id!r}")
        names = {b.name for b in rig.bones}
        unknown = set(self.spec.pivots) - {p.name for p in rig.part_sheet.parts}
        if unknown:
            raise ValueError(f"pivots name parts the rig does not have: {sorted(unknown)}")
        if set(self.spec.bone_lengths) != names:
            raise ValueError(f"bone_lengths must cover the rig's bones: {sorted(set(self.spec.bone_lengths) ^ names)}")

    def pivot(self, part) -> tuple[float, float]:
        """Where ``part``'s joint sits in its cell (fractions): the character's own value or the rig's."""
        return tuple(self.spec.pivots.get(part.name, part.pivot))

    def load_sheet(self):
        from PIL import Image

        return Image.open(self.sheet_path).convert("RGBA")

    def puppet(self, rig: RigSpec, items: list | None = None) -> Puppet:
        """The character's puppet; ``items`` (``ggs.rig.equipment.Item``) are held at their sockets."""
        self.check(rig)
        sheet = self.load_sheet()
        cell = rig.part_sheet.cell_px
        want = (rig.part_sheet.cols * cell, rig.part_sheet.rows * cell)
        if sheet.size != want:
            raise ValueError(f"{self.sheet_path}: part sheet is {sheet.size}, rig expects {want}")
        parts: dict[str, PartImage] = {}
        for p in rig.part_sheet.parts:
            col, row = p.cell
            img = sheet.crop((col * cell, row * cell, (col + 1) * cell, (row + 1) * cell))
            fx, fy = self.pivot(p)
            pivot = (fx * cell, fy * cell)
            length = self.spec.bone_lengths[p.bone] * self.spec.px_per_height
            parts[p.bone] = PartImage(img, pivot, p.art_angle, length, 1.0)
            if p.bone_far:
                parts[p.bone_far] = PartImage(img, pivot, p.art_angle, length, p.far_tint)
        from ggs.rig.equipment import attachment_for

        return Puppet(bone_defs(rig), dict(self.spec.bone_lengths), list(rig.draw_order),
                      limb_groups(rig), parts, [attachment_for(i, rig) for i in items or []])


def load_character(path: str | Path) -> Character:
    path = Path(path)
    try:
        spec = CharacterSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return Character(spec, path.parent)


# ---------------------------------------------------------------- programmer art

OUTLINE = (35, 32, 40)
SKIN = (238, 194, 160)
SHIRT = (70, 128, 200)
PANTS = (64, 66, 98)
SHOE = (96, 62, 42)
EYE = (30, 30, 40)

# part name -> (fill colour, width in standing heights)
_STYLE = {
    "head": (SKIN, 0.050),  # neck width; the skull is a circle
    "torso": (SHIRT, 0.140),
    "pelvis": (PANTS, 0.135),
    "upper_arm": (SHIRT, 0.058),
    "forearm": (SKIN, 0.048),
    "hand": (SKIN, 0.050),
    "thigh": (PANTS, 0.078),
    "shin": (PANTS, 0.062),
    "foot": (SHOE, 0.052),
}


def _capsule(draw, p0, p1, width, fill, outline_px, start_cap=True):
    def cap(p, r, color):
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)

    for w, color in ((width + 2 * outline_px, OUTLINE), (width, fill)):
        draw.line([p0, p1], fill=color, width=max(1, round(w)))
        if start_cap:
            cap(p0, w / 2, color)
        cap(p1, w / 2, color)


def _head(draw, pivot, d, length, s, variant, outline_px, skull: float = 1.0, neck: float = 1.0):
    """Neck capsule + skull circle + a face on the forward (screen right) side. ``skull``/``neck`` scale the
    skull radius and neck width (body shapes)."""
    neck_end = (pivot[0] + d[0] * length * 0.35, pivot[1] + d[1] * length * 0.35)
    _capsule(draw, pivot, neck_end, 0.05 * s * neck, SKIN, outline_px)
    c = (pivot[0] + d[0] * length * 0.62, pivot[1] + d[1] * length * 0.62)
    r = 0.075 * s * skull
    draw.ellipse([c[0] - r - outline_px, c[1] - r - outline_px, c[0] + r + outline_px, c[1] + r + outline_px],
                 fill=OUTLINE)
    draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=SKIN)
    # hair cap on the back/top
    draw.pieslice([c[0] - r, c[1] - r, c[0] + r, c[1] + r], 150, 300, fill=OUTLINE)
    ex, ey, er = c[0] + 0.45 * r, c[1] - 0.15 * r, 0.12 * r
    if variant == "blink":
        draw.line([(ex - er * 1.5, ey), (ex + er * 1.5, ey)], fill=EYE, width=max(1, round(er * 0.8)))
    elif variant == "hurt":
        for sgn in (-1, 1):
            draw.line([(ex - er * 1.3, ey - sgn * er * 1.3), (ex + er * 1.3, ey + sgn * er * 1.3)],
                      fill=EYE, width=max(1, round(er * 0.7)))
    else:
        draw.ellipse([ex - er, ey - er, ex + er, ey + er], fill=EYE)
    mx, my = c[0] + 0.5 * r, c[1] + 0.45 * r
    if variant == "mouth_open":
        mr = 0.16 * r
        draw.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=EYE)
    else:
        draw.line([(mx - 0.18 * r, my), (mx + 0.15 * r, my)], fill=EYE, width=max(1, round(0.06 * r)))
    # nose
    draw.polygon([(c[0] + r * 0.92, c[1] - r * 0.05), (c[0] + r * 1.12, c[1] + r * 0.18),
                  (c[0] + r * 0.9, c[1] + r * 0.22)], fill=SKIN, outline=OUTLINE)


def make_programmer_art(rig: RigSpec, out_dir: str | Path, px_per_height: float = 600.0,
                        name: str = "programmer_art", supersample: int = 4, body=None) -> Path:
    """Draw a flat-colour placeholder part sheet for ``rig`` and write ``parts.png`` + ``character.yaml``.

    Lengths are the rig's rest lengths, reshaped by ``body`` (a ``ggs.rig.body.BodyShape``) when given.
    The art has native alpha (drawn on a transparent canvas)."""
    from PIL import Image, ImageDraw

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_spec = rig.part_sheet
    cell = sheet_spec.cell_px
    k = supersample
    big = Image.new("RGBA", (sheet_spec.cols * cell * k, sheet_spec.rows * cell * k), (0, 0, 0, 0))
    lengths = {b.name: b.rest_length for b in rig.bones}
    widen = skull = 1.0
    if body is not None:
        from ggs.rig.body import body_lengths, body_scale

        lengths = body_lengths(rig, body)
        f = body_scale(rig, body)
        widen, skull = body.width * f, body.head * f
    s = px_per_height * k
    outline_px = max(1.0, 0.006 * s)

    def cell_draw(cell_xy):
        layer = Image.new("RGBA", (cell * k, cell * k), (0, 0, 0, 0))
        return layer, ImageDraw.Draw(layer)

    head_part = next((p for p in sheet_spec.parts if p.name == "head"), None)
    jobs = [(p, p.cell, None) for p in sheet_spec.parts]
    if head_part is not None:
        jobs += [(head_part, v.cell, v.name) for v in sheet_spec.head_variants]
    for part, cell_xy, variant in jobs:
        layer, draw = cell_draw(cell_xy)
        pivot = (part.pivot[0] * cell * k, part.pivot[1] * cell * k)
        length = lengths[part.bone] * s
        dx, dy = bone_vector(part.art_angle, 1.0)
        end = (pivot[0] + dx * length, pivot[1] + dy * length)
        fill, width = _STYLE.get(part.name, (SHIRT, 0.06))
        width *= widen
        if part.name == "head":
            _head(draw, pivot, (dx, dy), length, s, variant, outline_px, skull=skull, neck=widen)
        elif part.name == "torso":  # flat hem at the hips (the pelvis covers it); keeps the cell edge clear
            _capsule(draw, pivot, end, width * s, fill, outline_px, start_cap=False)
        elif part.name == "foot":
            _capsule(draw, (pivot[0] - 0.02 * s, pivot[1]), end, width * s, fill, outline_px)
        else:
            _capsule(draw, pivot, end, width * s, fill, outline_px)
        big.alpha_composite(layer, (cell_xy[0] * cell * k, cell_xy[1] * cell * k))
        if not (0 <= end[0] <= cell * k and 0 <= end[1] <= cell * k) or not math.isfinite(end[0]):
            raise ValueError(f"part {part.name!r} does not fit its cell at {px_per_height} px per height")

    sheet = big.convert("RGBa").resize((big.width // k, big.height // k), Image.LANCZOS).convert("RGBA")
    sheet.save(out_dir / "parts.png")
    spec = CharacterSpec(name=name, rig=rig.id, sheet="parts.png", px_per_height=px_per_height,
                         bone_lengths=lengths)
    path = out_dir / "character.yaml"
    path.write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False), encoding="utf-8")
    return path
