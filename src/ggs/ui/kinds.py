"""What the user can create, in their words: icon, name, one-line hint, and the size choices that apply."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Kind:
    id: str  # recipe id (asset class), or "character"
    icon: str
    label: str
    hint: str
    sizes: tuple[str, ...] = ("small", "medium", "large", "full")
    ready: bool = True


KINDS = [
    Kind("sprite", "\U0001F381", "Object", "Props, pickups, enemies, anything on its own"),
    Kind("held_item", "\U0001F5E1", "Weapon or tool", "Something a character holds (sword, staff, gun)"),
    Kind("ui_icon", "\U0001F4A0", "Icon", "Inventory items, skills, buttons"),
    Kind("background", "\U0001F3DE", "Background", "A scene behind the gameplay, with a far layer", ("full",)),
    Kind("background_wrap", "\U0001F501", "Scrolling background", "Repeats left to right forever", ("full",)),
    Kind("tile", "\U0001F9F1", "Tile", "A texture that repeats: floor, wall, ground", ("small", "medium", "large")),
    Kind("ui_panel", "\U0001FA9F", "UI panel", "Window or button frame that stretches", ("medium", "large", "full")),
    Kind("vfx", "✨", "Effect", "Glows, sparks, magic for particles", ("small", "medium", "large")),
    Kind("character", "\U0001F9CD", "Character", "A person or creature to animate"),
]

SIZE_LABELS = {"small": ("S", "128 px"), "medium": ("M", "256 px"), "large": ("L", "512 px"),
               "full": ("Full", "as drawn")}


def kind(kind_id: str) -> Kind:
    return next((k for k in KINDS if k.id == kind_id), Kind(kind_id, "\U0001F4E6", kind_id, ""))
