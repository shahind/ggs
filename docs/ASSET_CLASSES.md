# Asset classes (recipes)

Collections are user labels (Characters, Weapons, Trees...). Each binds to one recipe below.

| Recipe | Covers | Output | Post-processing |
|---|---|---|---|
| sprite | items, icons, props, enemies, projectiles, static characters | RGBA PNG, power-of-two canvas, tight trim + padding | matting, edge defringe, optional palette lock |
| rig_character | animated humanoids and creatures | part sheet -> rig package, GIF/APNG/WebP, sprite sheet + JSON | normalize to bone lengths, reassembly QA |
| background | full scenes, parallax layers | opaque PNG/WebP (e.g. 2560x1440); parallax as 3-5 RGBA layers | layer split, edge-safe crops |
| tile | seamless terrain, tilesets | square seamless PNG, tileset atlas | wrap + edge-match check |
| ui_kit | panels, buttons (normal/hover/pressed/disabled), bars, frames, icons | 9-slice PNGs + border metadata | slice borders, state variants, flat fills |
| vfx | explosions, sparkles, smoke, glow | atlas on black for additive blend, or RGBA | luminance-to-alpha, loop trim; prefer particle textures over sheets for size |
| key_art | posters, logos, store capsules, menu art | large PNG sized to store specs | upscale; text via a text-capable model |
| font | not generated | use OFL fonts; optional bitmap-font atlas | |

## Built (v1)
`ggs generate --class <id>`; the recipe files are in `recipes/`.
| Class | What you get | QA |
|---|---|---|
| sprite, held_item, ui_icon | trimmed RGBA PNG (held_item: upright, handle at the bottom, for `ggs item make`) | coverage, cut-off, halo, leftover background, specks |
| background | `<out>.png` full scene + `<out>_far.png` (the scene with foreground and middle ground edited away; JSON lists parallax factors) | layer coverage |
| background_wrap | the same, both wrapping horizontally (rolled half a frame, seam repainted) | seam rank |
| tile | seamless x+y texture, `tile.export_px` (512) | seam rank: the seam must not be the strongest line in the image (approximate) |
| ui_panel | trimmed panel + `_hover` + `_pressed` + `_slices.json` (Unity border L,B,R,T; Godot patch margins; Defold slice9) | all four borders found, centre flat enough to stretch |
| vfx | RGBA particle texture for additive blending (alpha = brightness on black) | fades out before the edges, coverage |

Not yet: matted mid/near parallax layers (drawn layers get painted haze, which no matting separates), tilesets
and autotiles, UI bars/states beyond brightness, VFX flipbooks, key art, pixel-art palette lock.

Environment motion (trees, grass, flags) is a single sprite swayed by an engine shader, not an animation sheet.
Pixel-art output: generate large, downscale with k-centroid, snap to a fixed palette.
