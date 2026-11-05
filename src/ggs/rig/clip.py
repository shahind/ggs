"""Clip format v1: a 2D animation for a rig, as produced by the retargeter. See docs/CLIP_FORMAT_V1.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RootTrack(BaseModel):
    """Root (pelvis start) position per frame, in standing heights. +x forward/right, +y down, y=0 ground."""

    x: list[float]
    y: list[float]


class Clip(BaseModel):
    format: Literal["ggs.clip"] = "ggs.clip"
    version: Literal[1] = 1
    rig: str
    source: str
    profile: str
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    in_place: bool = False
    loop: bool = False  # True: frame 0 follows the last frame seamlessly
    loop_travel: float = 0.0  # root x distance covered per loop (standing heights), for root motion
    bones: list[str]
    bone_lengths: dict[str, float]
    angles: dict[str, list[float]]  # absolute, degrees clockwise from +y, unwrapped over time
    local_angles: dict[str, list[float]]  # relative to parent, wrapped to (-180, 180]
    root: RootTrack
    swaps: dict[str, list[bool]] = {}  # limb group -> True when the far limb is nearer the camera
    warnings: list[str] = []

    @model_validator(mode="after")
    def _check(self) -> Clip:
        n = self.frame_count
        bones = set(self.bones)
        for label, table in (("angles", self.angles), ("local_angles", self.local_angles)):
            if set(table) != bones:
                raise ValueError(f"{label} keys must match bones: {sorted(set(table) ^ bones)}")
            for b, track in table.items():
                if len(track) != n:
                    raise ValueError(f"{label}[{b!r}] has {len(track)} values, expected {n}")
        if set(self.bone_lengths) != bones:
            raise ValueError("bone_lengths keys must match bones")
        if len(self.root.x) != n or len(self.root.y) != n:
            raise ValueError(f"root track must have {n} values")
        for g, track in self.swaps.items():
            if len(track) != n:
                raise ValueError(f"swaps[{g!r}] has {len(track)} values, expected {n}")
        return self

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(), separators=(",", ":")), encoding="utf-8")


def load_clip(path: str | Path) -> Clip:
    path = Path(path)
    try:
        return Clip.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc
