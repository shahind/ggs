"""Recipes = asset-class presets (recipes/*.yaml). The CLI always names the class explicitly."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from ggs.paths import repo_root
from ggs.qa.sprite import SpriteQaRules


class PostSettings(BaseModel):
    alpha_low: float = Field(0.04, ge=0, lt=0.5)
    alpha_high: float = Field(0.96, gt=0.5, le=1)
    padding: int = Field(4, ge=0)
    power_of_two: bool = False


class LayerSpec(BaseModel):
    """One background layer. The first layer is the far one (opaque); later layers are drawn on white and matted,
    using the first layer as a style reference."""
    name: str
    prompt_template: str
    opaque: bool = False
    parallax: float = Field(1.0, ge=0, le=2)  # scroll speed relative to the camera (far < 1)


class TileSettings(BaseModel):
    seamless: Literal["x", "xy"] = "xy"
    band: float = Field(0.14, gt=0, lt=0.5)  # repainted seam band, share of the tile
    max_seam: float = Field(0.995, gt=0, le=1)  # percentile rank of the seam among the image's lines
    export_px: int = Field(512, ge=16, le=4096)


class PanelSettings(BaseModel):
    min_border: int = Field(6, ge=1)
    max_border: float = Field(0.42, gt=0, lt=0.5)  # of the panel size
    centre_tolerance: float = Field(28.0, gt=0)  # colour distance that still counts as the plain centre
    max_centre_spread: float = Field(22.0, gt=0)  # the centre must be flat enough to stretch
    states: bool = True


class VfxSettings(BaseModel):
    max_edge: float = Field(0.08, ge=0, le=1)  # brightest border pixel (the effect must fade out inside the frame)
    export_px: int = Field(256, ge=16, le=4096)


class Recipe(BaseModel):
    id: str
    kind: Literal["sprite", "background", "tile", "ui_panel", "vfx"] = "sprite"
    description: str = ""
    base_model: str
    matting: str
    size: tuple[int, int]
    background: str
    prompt_template: str
    # per base model (pipeline id) overrides, e.g. SDXL's CLIP encoders read only 77 tokens and take
    # "game asset" to mean a sheet of many objects
    prompt_templates: dict[str, str] = Field(default_factory=dict)
    post: PostSettings = PostSettings()
    qa: SpriteQaRules = SpriteQaRules()
    export: Literal["png"] = "png"
    min_profile: Literal["lite", "standard"] = "lite"
    layers: list[LayerSpec] = Field(default_factory=list)  # background
    seamless_x: bool = False  # background: every layer wraps horizontally (side-scrollers)
    tile: TileSettings | None = None
    panel: PanelSettings | None = None
    vfx: VfxSettings | None = None

    @field_validator("size")
    @classmethod
    def _size(cls, v: tuple[int, int]) -> tuple[int, int]:
        if any(s % 16 or s < 256 or s > 2048 for s in v):
            raise ValueError("size must be multiples of 16 between 256 and 2048")
        return v

    @field_validator("background")
    @classmethod
    def _bg(cls, v: str) -> str:
        t = v.lstrip("#")
        if len(t) != 6:
            raise ValueError("background must be #RRGGBB")
        int(t, 16)
        return "#" + t.upper()

    @field_validator("prompt_template")
    @classmethod
    def _tpl(cls, v: str) -> str:
        if "{description}" not in v:
            raise ValueError("prompt_template needs {description}")
        return v

    @field_validator("prompt_templates")
    @classmethod
    def _tpls(cls, v: dict[str, str]) -> dict[str, str]:
        for model, t in v.items():
            if "{description}" not in t:
                raise ValueError(f"prompt_templates.{model} needs {{description}}")
        return v

    @property
    def background_rgb(self) -> tuple[int, int, int]:
        t = self.background.lstrip("#")
        return int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)

    def prompt(self, description: str, style: str = "", model: str | None = None) -> str:
        style = f"{style.strip()}, " if style.strip() else ""
        template = self.prompt_templates.get(model or "", self.prompt_template)
        return " ".join(template.format(style=style, description=description.strip()).split())


def recipes_dir() -> Path:
    return repo_root() / "recipes"


def available_recipes() -> list[str]:
    return sorted(p.stem for p in recipes_dir().glob("*.yaml"))


def load_recipe(name_or_path: str | Path) -> Recipe:
    p = Path(name_or_path)
    if not p.suffix:
        p = recipes_dir() / f"{name_or_path}.yaml"
        if not p.exists():
            raise ValueError(f"unknown asset class {name_or_path!r}; available: {available_recipes()}")
    try:
        return Recipe.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError(f"{p}: {exc}") from exc


def layer_prompt(recipe: Recipe, layer: LayerSpec, description: str, style: str = "") -> str:
    style = f"{style.strip()}, " if style.strip() else ""
    return " ".join(layer.prompt_template.format(style=style, description=description.strip()).split())
