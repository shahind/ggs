"""What the user can create, in their words: icon, name, one-line hint, and the size choices that apply."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Kind:
    id: str  # recipe id (asset class), or "character"
    icon: str  # name in ggs.ui.icons
    label: str
    hint: str
    sizes: tuple[str, ...] = ("small", "medium", "large", "full")
    ready: bool = True


KINDS = [
    Kind("sprite", "sprite", "Object", "Props, pickups, enemies, anything on its own"),
    Kind("held_item", "held_item", "Weapon or tool", "Something a character holds (sword, staff, gun)"),
    Kind("ui_icon", "ui_icon", "Icon", "Inventory items, skills, buttons"),
    Kind("background", "background", "Background", "A scene behind the gameplay, with a far layer", ("full",)),
    Kind("background_wrap", "background_wrap", "Scrolling background", "Repeats left to right forever", ("full",)),
    Kind("tile", "tile", "Tile", "A texture that repeats: floor, wall, ground", ("small", "medium", "large")),
    Kind("ui_panel", "ui_panel", "UI panel", "Window or button frame that stretches", ("medium", "large", "full")),
    Kind("vfx", "vfx", "Effect", "Glows, sparks, magic for particles", ("small", "medium", "large")),
    Kind("character", "character", "Character", "A person or creature to animate"),
]

SIZE_LABELS = {"small": ("S", "128 px"), "medium": ("M", "256 px"), "large": ("L", "512 px"),
               "full": ("Full", "as drawn")}


def kind(kind_id: str) -> Kind:
    return next((k for k in KINDS if k.id == kind_id), Kind(kind_id, "assets", kind_id, ""))
