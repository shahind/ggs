---
name: export-formats
description: Checklist for implementing or changing asset exports (APNG/WebP/GIF, sprite sheet + JSON atlas, 9-slice, rig package, Godot importer). Use whenever output files that a game engine reads are created or changed.
---

# Export formats checklist

**Animated images**
- APNG/WebP keep full alpha; GIF only 1-bit alpha and 256 colours, so offer GIF only with a matte colour option.
- Store fps/frame durations explicitly and test the round trip (frame count, size, timing, alpha).

**Sprite sheet + atlas**
- Power-of-two sheet, uniform or packed frames, 1-2px padding/extrusion against bleeding.
- JSON atlas: frame rects, pivot, per-animation frame lists with fps and loop flag.
- A test loads the JSON and slices the PNG to verify every frame rectangle.

**9-slice UI**
- One PNG per state (normal, hover, pressed, disabled) with identical slice borders recorded in a metadata file.

**Rig package**
- `parts.png` atlas + `rig.json` (bones, rest pose, pivots, draw order, sockets) + `clips/*.json` (per-frame bone angles, root offsets, draw-order swaps).
- Pure data; the engine importer builds the scene (Godot: Skeleton2D + Bone2D + Polygon2D/Sprite2D + AnimationPlayer).
- Round-trip test: render a clip from the package and compare with the in-app render.

**All**
- Keep build size in mind; prefer rig packages and particle textures to large sheets.
- Document each format in `docs/` with one worked example.
