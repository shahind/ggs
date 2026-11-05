"""Shared helpers for engine exporters. They all start from a written rig package (``ggs.rig_package``).

Engine conventions differ from the package (screen pixels, +y down, angles clockwise from down):
- Godot 2D is +y down and its rotation is clockwise, so values map 1:1.
- Unity and Defold are +y up and rotate counter-clockwise: negate y and angles.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ggs.rig.retarget import LimbGroup, swap_draw_order


@dataclass
class Package:
    path: Path
    doc: dict
    clips: dict[str, dict]

    @property
    def name(self) -> str:
        return self.doc["name"]

    @property
    def bones(self) -> list[dict]:
        return self.doc["bones"]

    def bone(self, name: str) -> dict:
        return next(b for b in self.bones if b["name"] == name)

    def children(self, name: str | None) -> list[str]:
        return [b["name"] for b in self.bones if b["parent"] == name]

    @property
    def root(self) -> str:
        return self.children(None)[0]

    def part_for(self, bone: str) -> tuple[dict, bool]:
        """(part, is_far) for a bone."""
        for p in self.doc["parts"]:
            if bone in p["bones"]:
                return p, p["bones"].index(bone) > 0
        raise KeyError(bone)

    def ordered_bones(self) -> list[str]:
        """Parents before children."""
        out: list[str] = []
        stack = [self.root]
        while stack:
            cur = stack.pop(0)
            out.append(cur)
            stack[:0] = self.children(cur)
        return out

    def bone_path(self, name: str) -> list[str]:
        chain = []
        cur: str | None = name
        while cur is not None:
            chain.append(cur)
            cur = self.bone(cur)["parent"]
        return list(reversed(chain))


def load_package(path: str | Path) -> Package:
    path = Path(path)
    doc = json.loads((path / "rig.json").read_text(encoding="utf-8"))
    if doc.get("format") != "ggs.rig_package":
        raise ValueError(f"{path}: not a ggs rig package")
    clips = {c: json.loads((path / "clips" / f"{c}.json").read_text(encoding="utf-8")) for c in doc["clips"]}
    return Package(path, doc, clips)


def local_unwrapped(pkg: Package, clip: dict) -> dict[str, list[float]]:
    """Per-bone angle relative to its parent, continuous over time (from the unwrapped absolute angles)."""
    out = {}
    for b in pkg.bones:
        a = clip["angles"][b["name"]]
        if b["parent"] is None:
            out[b["name"]] = list(a)
        else:
            p = clip["angles"][b["parent"]]
            out[b["name"]] = [round(x - y, 3) for x, y in zip(a, p)]
    return out


def draw_index_tracks(pkg: Package, clip: dict) -> dict[str, list[int]]:
    """Per-frame back-to-front index for every bone whose index changes during the clip ({} if none)."""
    groups = [LimbGroup(g["name"], tuple(g["near"]), tuple(g["far"])) for g in pkg.doc["limb_groups"]]
    order = pkg.doc["draw_order"]
    n = clip["frame_count"]
    swaps = clip.get("swaps", {})
    tracks: dict[str, list[int]] = {b: [] for b in order}
    for i in range(n):
        o = swap_draw_order(order, groups, {g: v[i] for g, v in swaps.items()})
        for idx, b in enumerate(o):
            tracks[b].append(idx)
    return {b: v for b, v in tracks.items() if len(set(v)) > 1}


def base_draw_index(pkg: Package) -> dict[str, int]:
    return {b: i for i, b in enumerate(pkg.doc["draw_order"])}


_IDENT = re.compile(r"[^A-Za-z0-9_]")


def ident(name: str) -> str:
    """Safe identifier for file, node and class names."""
    s = _IDENT.sub("_", name).strip("_") or "character"
    return s if not s[0].isdigit() else f"c_{s}"
