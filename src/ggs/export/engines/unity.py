"""Unity exporter: ``GgsRigPlayer.cs`` (runtime component) + ``<name>.ggsrig.json`` (JsonUtility-friendly).

A runtime component instead of a generated prefab: Unity prefab/meta YAML carries GUIDs and file IDs that are
fragile to write by hand, while this needs no editor tooling and works in any Unity 2021.3+ project.
The component builds bone GameObjects with SpriteRenderers cut from ``parts.png`` and plays the clips.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from ggs.export.engines.common import draw_index_tracks, ident, load_package, local_unwrapped

UNITY_MIN = "2021.3"


def unity_rig_json(package_dir: str | Path) -> dict:
    pkg = load_package(package_dir)
    clips = []
    for cname, c in pkg.clips.items():
        local = local_unwrapped(pkg, c)
        clips.append({
            "name": cname,
            "fps": c["fps"],
            "frameCount": c["frame_count"],
            "loop": c["loop"],
            "loopTravelPx": c["loop_travel_px"],
            "rootX": c["root"]["x"],
            "rootY": c["root"]["y"],
            "local": [{"bone": b, "values": local[b]} for b in pkg.ordered_bones()],
            "drawIndex": [{"bone": b, "values": v} for b, v in draw_index_tracks(pkg, c).items()],
        })
    d = pkg.doc
    return {
        "format": "ggs.unity_rig",
        "version": 1,
        "name": d["name"],
        "heightPx": d["height_px"],
        "atlasW": d["image_size"]["w"],
        "atlasH": d["image_size"]["h"],
        "bones": [{"name": b, "parent": pkg.bone(b)["parent"] or "", "length": pkg.bone(b)["length"],
                   "attach": pkg.bone(b)["attach"]} for b in pkg.ordered_bones()],
        "parts": [{"name": p["name"], "bones": p["bones"], "rect": p["rect"], "pivotX": p["pivot"]["x"],
                   "pivotY": p["pivot"]["y"], "artAngle": p["art_angle"], "lengthPx": p["length_px"],
                   "farTint": p["far_tint"]} for p in d["parts"]],
        "items": [{"name": i["name"], "bone": i["bone"], "rect": i["rect"], "pivotX": i["pivot"]["x"],
                   "pivotY": i["pivot"]["y"], "artAngle": i["art_angle"], "holdAngle": i["hold_angle"],
                   "socketT": i["socket_t"], "socketOffset": i["socket_offset"]} for i in d.get("items", [])],
        "headVariants": [{"name": v["name"], "part": v["part"], "rect": v["rect"], "pivotX": v["pivot"]["x"],
                          "pivotY": v["pivot"]["y"]} for v in d.get("head_variants", [])],
        "drawOrder": d["draw_order"],
        "clips": clips,
    }


def player_source() -> str:
    return resources.files("ggs.export.engines").joinpath("unity_player.cs.txt").read_text(encoding="utf-8")


def write_unity(package_dir: str | Path) -> Path:
    package_dir = Path(package_dir)
    data = unity_rig_json(package_dir)
    name = ident(data["name"])
    path = package_dir / f"{name}.ggsrig.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8", newline="\n")
    (package_dir / "GgsRigPlayer.cs").write_text(player_source(), encoding="utf-8", newline="\n")
    first = data["clips"][0]["name"] if data["clips"] else ""
    (package_dir / "README_UNITY.txt").write_text(
        f"Unity {UNITY_MIN}+:\n"
        f"1. Copy parts.png, {path.name} and GgsRigPlayer.cs into Assets/ (any folder).\n"
        "2. Select parts.png: Texture Type = Sprite (2D and UI) or Default, Alpha Is Transparency = on,\n"
        "   Compression = None (or high quality), Filter Mode = Bilinear (Point for pixel art). Apply.\n"
        "3. Create an empty GameObject, add the GgsRigPlayer component, assign Rig Json = "
        f"{path.name} and Atlas = parts.png.\n"
        f"4. Press Play: it plays '{first}'. Call Play(\"name\") to switch clips, SetHeadVariant(\"blink\") for\n"
        "   faces. The GameObject origin is the ground point under the character; move it by LoopTravelUnits\n"
        "   per loop for root motion. Pixels Per Unit defaults to 100.\n",
        encoding="utf-8", newline="\n")
    return path
