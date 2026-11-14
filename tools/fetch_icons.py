"""Fetch the app's icon set: Lucide line icons (ISC licence, https://lucide.dev), pinned to one version so the set
stays consistent. Writes src/ggs/ui/icons/<name>.svg and LICENSE-lucide.txt. Run: python tools/fetch_icons.py

The icons are tinted at run time (ggs.ui.icons), so one SVG serves light text, dark text on accent buttons and
the disabled state. The app logo (app.svg) is drawn by hand and is not fetched.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

VERSION = "1.69.0"
BASE = f"https://unpkg.com/lucide-static@{VERSION}"
OUT = Path(__file__).resolve().parents[1] / "src" / "ggs" / "ui" / "icons"

# GGS name -> Lucide name
ICONS = {
    # asset kinds
    "sprite": "box",
    "held_item": "sword",
    "ui_icon": "gem",
    "background": "mountain-snow",
    "background_wrap": "gallery-horizontal",
    "tile": "brick-wall",
    "ui_panel": "app-window",
    "vfx": "sparkles",
    "character": "user-round",
    # project tree
    "characters": "users-round",
    "assets": "boxes",
    "animations": "clapperboard",
    "styles": "palette",
    "projects": "folders",
    # actions
    "new": "plus",
    "settings": "settings",
    "generate": "sparkles",
    "regenerate": "refresh-cw",
    "refine": "wand-sparkles",
    "variations": "copy",
    "export": "file-output",
    "advanced": "sliders-horizontal",
    "save": "save",
    "delete": "trash-2",
    "revert": "undo-2",
    "undo": "undo-2",
    "redo": "redo-2",
    "fit": "scan",
    "open": "folder-open",
    "open_with": "external-link",
    "play": "play",
    "pause": "pause",
    "skeleton": "bone",
    "confirm": "check",
    "accept": "circle-check",
    "previous": "chevron-left",
    "next": "chevron-right",
    "download": "download",
    "layout": "layout-template",
    "rig": "spline",
    "brush": "brush",
    "eraser": "eraser",
    "picker": "pipette",
    "fill": "paint-bucket",
    "front": "person-standing",
    "side": "footprints",
    "diagnostics": "activity",
    "cloud": "cloud",
    "key": "key-round",
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def main() -> None:
    for name, lucide in ICONS.items():
        svg = fetch(f"{BASE}/icons/{lucide}.svg").decode("utf-8")
        (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
        print(f"{name:16} <- {lucide}")
    (OUT / "LICENSE-lucide.txt").write_bytes(fetch(f"{BASE}/LICENSE"))


if __name__ == "__main__":
    main()
