# Export formats (M2)

All exports come from one renderer (`src/ggs/rig/puppet.py`): part images + bones + a clip -> RGBA frames.
Resampling runs in premultiplied alpha, so edges stay clean. Far-side limbs reuse the near-side art, darkened
by the part's `far_tint`. Draw order follows the rig's `draw_order`, with near/far groups swapped on frames
where the clip's `swaps` flag is set.

Frame exports are rendered **in place**: the character stays put and its travel per loop is kept in the clip
(`loop_travel`) so the game can move the sprite. Every frame shares one crop box, so the pivot is the same in
every frame.

**Pivot:** x sits under the hips. y is the lowest visible pixel over the whole clip, which is where the feet touch
the ground. Place the sprite with its pivot on the ground line.

## Worked example

```powershell
python -m ggs mocap retarget data\mocap\cmu\02_01_walk.bvh --profile cmu_bvh --fps 24 --out take.json
python -m ggs clip loop take.json --out walk.json          # one seamless cycle (26 frames, 1.08 s)
python -m ggs character programmer-art --out chars\pa      # placeholder art: parts.png + character.yaml
python -m ggs export walk.json --character chars\pa\character.yaml --format apng  --out walk.png
python -m ggs export walk.json --character chars\pa\character.yaml --format webp  --out walk.webp
python -m ggs export walk.json --character chars\pa\character.yaml --format gif   --matte "#88aa66" --out walk.gif
python -m ggs export walk.json --character chars\pa\character.yaml --format sheet --out walk_sheet.png
python -m ggs export walk.json --character chars\pa\character.yaml --format rig   --out walk_rig
```

`--height` sets the character's standing height in pixels (default 256). At 256 px the CMU walk gives
154x282 frames. The sizes from this example:

| Output | Size |
|---|---|
| APNG | 540 KB |
| WebP (lossless) | 390 KB |
| GIF | 225 KB |
| Sprite sheet (1024x2048) | 560 KB |
| Rig package: parts atlas (256x256) + JSON | 15 KB + 14 KB |

Prefer the rig package when the engine can animate bones.

## Animated images
- **APNG** (`.png`) and **WebP** (lossless) keep full 8-bit alpha. They are the default choices.
- **GIF** has only 1-bit alpha and 256 colours, so it needs `--matte #RRGGBB`. Soft edges are flattened onto
  that colour, and fully transparent pixels stay transparent. Pick the colour of the game background.
- Frame durations are whole milliseconds that add up to the exact clip length (24 fps -> 42, 41, 42 ...).
  Looping clips loop forever; other clips play once.

## Sprite sheet + JSON atlas (`--format sheet`)
`walk_sheet.png` plus `walk_sheet.json`:
- The sheet is a uniform grid on a power-of-two canvas, with 2 px padding and 1 px edge extrusion against bleeding.
- The JSON uses the TexturePacker "JSON hash" layout, so loaders such as Phaser read it directly:
  - `frames.<name>_NNN`: `frame` rect, `pivot` (fraction of the frame), `duration` in ms.
  - `animations.<name>`: the frame names in order.
  - `meta.ggs`: `fps`, `loop`, frame count, padding, and `anchor_px` (the pivot in pixels).

## Rig package (`--format rig`)
A folder for engines that animate bones. It is small: one part atlas plus JSON.

```
walk_rig/
  parts.png        trimmed parts, shelf-packed into a power-of-two atlas
  rig.json         bones, parts, draw order, limb groups, sockets, rest pose
  clips/walk.json  per-frame angles, root track, swaps
```

`rig.json` (format `ggs.rig_package`, version 1). All values are in output pixels: +x right, +y down, angles in
degrees clockwise from +y.
- `bones[]`: `name`, `parent`, `length` (px), `attach` (where the bone starts on its parent: 0 = parent start,
  1 = parent end).
- `parts[]`: `name`, `bones` (the near bone first; the far bone uses the same art), `rect` in the atlas, `pivot`
  (px inside the rect, at the bone start), `art_angle` (direction the bone points in the art), `length_px`
  (drawn bone length), `far_tint`.
- `head_variants[]`: blink / mouth_open / hurt, each with its own rect and pivot. They swap in for the head part.
- `items[]` (held equipment, `ggs export --item`): `name`, `bone` and `socket` (e.g. `hand_near` / `grip_near`),
  `socket_t` / `socket_offset` (where on the bone: t along it, offset across it, in bone lengths), `rect`,
  `pivot` (the grip), `art_angle`, `hold_angle` (item direction = bone direction + hold_angle) and `hold`.
  In bone space (bone along +y, y down) the item sits at (-offset x L, t x L), rotated by hold_angle - art_angle.
  Godot, Unity and Defold place it as a fixed child of the bone, drawn above every part.
- `draw_order` (back to front), `limb_groups` (near/far chains that swap slots), `sockets`, and `rest_pose`
  (the pose the parts were drawn in).

`clips/<name>.json` (format `ggs.rig_clip`, version 1):
- `fps`, `frame_count`, `loop`, `loop_travel_px`.
- `root.x` / `root.y` (px; y is negative above the ground). For a loop, `root.x` has the steady travel
  removed so the loop plays in place; move the character by `loop_travel_px` per loop.
- `angles` (absolute) and `local_angles` (relative to the parent).
- `swaps`.

Placing a part (the same maths an importer needs): rotate the part image about its pivot by
`angle - art_angle` (clockwise on screen), scale it by `bone length / length_px`, and put the pivot at the
bone's start.

`ggs.export.rig_package.load_rig_package` rebuilds the puppet from the package. The tests render a clip from
the package and check it matches the in-app render.

## Engine exports (`--format godot|unity|defold`)
Each one writes the rig package into the output folder and adds that engine's files next to it, along with a
`README_<ENGINE>.txt` covering setup. None of them need an editor plugin.

| Engine | Files added | How it works |
|---|---|---|
| Godot 4.2+ | `<name>.tscn` | A scene: Node2D per bone, Sprite2D per part (a region of `parts.png`), and an AnimationPlayer with one Animation per clip. Rotations use linear-angle interpolation; draw-order swaps animate `z_index`. Head variants are hidden sprites. The texture path is relative, so the folder can sit anywhere under `res://`. |
| Unity 2021.3+ | `GgsRigPlayer.cs`, `<name>.ggsrig.json` | A runtime component builds the bone GameObjects and SpriteRenderers from the JSON (TextAsset) and the atlas, then plays clips. Draw-order swaps change `sortingOrder`. API: `Play(name)`, `Stop()`, `SetHeadVariant("blink")`, `LoopTravelUnits`. No prefab or meta files are generated, because Unity's GUID-based YAML is fragile to write by hand. |
| Defold 1.9+ | `<name>.atlas`, `<name>.collection`, `<name>_rig.lua`, `<name>_rig.script`, `parts/*.png` | One game object per bone plus a sprite object per part. The script plays clips from the Lua data module. Messages: `play {clip=...}`, `stop`, `set_head {variant=...}`. Defold paths are absolute, so `--defold-path /chars/hero` (default `/ggs/<name>`) must match where the folder is put. |

Coordinate handling: Godot is +y down and rotates clockwise, so package values carry over unchanged (in
radians). Unity and Defold are +y up and rotate counter-clockwise, so y and angles are negated.

`tests/test_engines.py` parses each engine's generated files, applies that engine's transform rules, and checks
every bone against the in-app pose. Deliberately flipped signs or offsets make those tests fail.

**Not yet checked inside the engines themselves.** Before relying on an export, open it once in each engine:
the Godot scene, the Unity component on an empty GameObject, and the Defold collection in a project.
