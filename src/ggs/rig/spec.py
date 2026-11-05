"""Rig spec models and validation. The YAML files in rigs/ are the source of truth."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class Bone(BaseModel):
    name: str
    parent: str | None = None
    rest_length: float = Field(gt=0)
    # Where this bone starts on its parent: t along the parent bone (0 = parent start, 1 = parent end).
    attach: float = Field(default=1.0, ge=0, le=1)
    # ... and sideways, in standing heights along the parent's local x axis (the bone axis is local +y; for a
    # parent pointing up, +x is screen left). Front-view rigs use it for shoulders and hips.
    attach_side: float = 0.0


class Part(BaseModel):
    name: str
    bone: str
    bone_far: str | None = None
    cell: tuple[int, int]
    pivot: tuple[float, float]
    # Direction the bone points in the art, same convention as bone angles: 0 = drawn hanging down,
    # 180 = drawn pointing up, -90 = drawn pointing right.
    art_angle: float = 0.0
    far_tint: float = Field(default=0.75, gt=0, le=1)


class HeadVariant(BaseModel):
    name: str
    cell: tuple[int, int]


class PartSheet(BaseModel):
    cell_px: int = Field(gt=0)
    cols: int = Field(gt=0)
    rows: int = Field(gt=0)
    parts: list[Part]
    head_variants: list[HeadVariant] = []


class Socket(BaseModel):
    name: str
    bone: str
    t: float = Field(ge=0, le=1)
    offset: float = 0.0


class RigSpec(BaseModel):
    id: str
    version: int
    view: str
    facing: str
    near_side: str
    bones: list[Bone]
    draw_order: list[str]
    part_sheet: PartSheet
    sockets: list[Socket] = []
    mocap_profiles: dict[str, dict[str, list[str]]] = {}

    @model_validator(mode="after")
    def _check(self) -> RigSpec:
        names = [b.name for b in self.bones]
        if len(set(names)) != len(names):
            raise ValueError("duplicate bone names")
        nameset = set(names)

        roots = [b.name for b in self.bones if b.parent is None]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root bone, found {roots}")
        parents = {b.name: b.parent for b in self.bones}
        for b, p in parents.items():
            if p is not None and p not in nameset:
                raise ValueError(f"bone {b!r} has unknown parent {p!r}")
        for b in names:  # cycle check
            seen: set[str] = set()
            cur: str | None = b
            while cur is not None:
                if cur in seen:
                    raise ValueError(f"cycle in bone hierarchy at {b!r}")
                seen.add(cur)
                cur = parents[cur]

        if sorted(self.draw_order) != sorted(names):
            raise ValueError("draw_order must list every bone exactly once")

        sheet = self.part_sheet
        used_cells: dict[tuple[int, int], str] = {}
        covered: set[str] = set()
        entries = [(p.name, p.cell) for p in sheet.parts] + [
            (f"head:{v.name}", v.cell) for v in sheet.head_variants
        ]
        for label, cell in entries:
            col, row = cell
            if not (0 <= col < sheet.cols and 0 <= row < sheet.rows):
                raise ValueError(f"{label!r} cell {cell} is outside the {sheet.cols}x{sheet.rows} grid")
            if cell in used_cells:
                raise ValueError(f"cell {cell} used by both {used_cells[cell]!r} and {label!r}")
            used_cells[cell] = label
        for p in sheet.parts:
            for bone in (p.bone, p.bone_far):
                if bone is None:
                    continue
                if bone not in nameset:
                    raise ValueError(f"part {p.name!r} references unknown bone {bone!r}")
                covered.add(bone)
            if not (0 <= p.pivot[0] <= 1 and 0 <= p.pivot[1] <= 1):
                raise ValueError(f"part {p.name!r} pivot must be within 0..1")
        missing = nameset - covered
        if missing:
            raise ValueError(f"bones without art source: {sorted(missing)}")

        for s in self.sockets:
            if s.bone not in nameset:
                raise ValueError(f"socket {s.name!r} references unknown bone {s.bone!r}")

        for profile, mapping in self.mocap_profiles.items():
            if not mapping:  # empty = not mapped yet
                continue
            if set(mapping) != nameset:
                diff = sorted(set(mapping) ^ nameset)
                raise ValueError(f"mocap profile {profile!r} does not cover the bone set exactly: {diff}")
            for bone, pair in mapping.items():
                if len(pair) != 2:
                    raise ValueError(f"mocap profile {profile!r} bone {bone!r} needs [from, to]")
        return self

    def bone(self, name: str) -> Bone:
        return next(b for b in self.bones if b.name == name)


def load_rig_spec(path: str | Path) -> RigSpec:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return RigSpec.model_validate(data)
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc
