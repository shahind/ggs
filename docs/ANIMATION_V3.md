# Animation v3: find the joints, bend or cut the drawing, pose-driven frames (owner review 2026-11-05)

## What the owner saw
- **Slicer:** body parts came out deformed, with cuts and pieces stuck together, so even good animations looked bad.
- **Drawn frames:** the frames did not keep the character's design, and did not follow the poses.
- The pose guide itself works well. A good starting pose is what makes animation work.

## What was wrong with the slicer
The drawing model follows the guide closely, but not exactly. The cutter gave every pixel to the *guide part*
nearest to it, so these went to the wrong part:
- a forearm drawn 20 px lower
- a skirt over the thighs
- the far arm drawn at another angle

Cutting by the guide's shapes instead of the drawing's is what produced the deformed, merged parts.

## What changed (done)
1. **Joint finder** (`ggs.rig.joints`). ViTPose-base (Apache-2.0, 0.33 GB on disk, about 0.2 GB of VRAM, about 1 s
   per drawing) finds shoulders, elbows, wrists, hips, knees and ankles in each slicing try.
   - A detection replaces the guide's joint only when it is confident (score of at least 0.45) and close to where
     the guide had that joint (within 0.15 of the standing height). The drawing was made over the guide, so a
     far-off detection is a mistake.
   - Near and far are matched by position, because detectors mix up left and right in side views.
   - On the clown the joints moved by 0.01–0.09 of the standing height.
2. **Pixel ownership from the fitted skeleton.** Each pixel belongs to the bone it is closest to, measured
   relative to that bone's width. On top of that:
   - Where parts overlap, the one drawn on top wins inside its own width.
   - Outlines (dark *and* thin) and colour changes split the drawing into regions. A region that a limb mostly
     owns (a boot, a sleeve with its own outline) goes wholly to that limb.
   - **Cloth:** a region on the thighs that mostly reaches past the thighs' width, and sits only on the trunk and
     thighs, hangs from the waist (a skirt or coat tail).
   - The ownership map is smoothed.
   - Checked on the six programmer-art bodies: 80–90 % agreement with their own parts, and none is mistaken for a
     skirt.
3. **Skirts:** when cloth hangs over the legs, the legs are drawn behind the body.
   - This is a per-character `draw_order` in `character.yaml`, so the Godot, Unity and Defold exports follow it.
   - The cutter fills the hidden tops of the legs under the skirt.
   - Hidden areas are never filled from outline colour.
4. **Warp animation (new, "Bend the drawing")** (`ggs.rig.skin`). The whole drawing is bent by its skeleton,
   with nothing cut apart.
   - Layers: far arm, far leg, body, near leg, near arm.
   - Each layer is a triangle mesh with linear blend skinning: a vertex blends between its bone and that bone's
     parent and children, weighted by distance relative to width.
   - Hidden parts of a layer get the limb's own typical colour.
   - A skirt sways with the legs at 40 %.
   - Rendered with OpenCV and exported as sprite sheet + JSON, APNG, WebP or GIF.
   - Made together with the sliced version (`sliced/warp/skin.json`).
   - `ggs export clip.json --character <item>/sliced/warp/skin.json --format sheet`.
   - On the clown: walk, run, punch and jump hold together and the skirt stays whole. The cut-out cannot do that
     for this character.
5. **The cut-out improves too**, because it cuts the fitted ownership map: the skirt stays whole and the limbs are
   separate pieces. Reassembly overlap on the clown went from 0.80–0.84 to 0.81–0.87. The 0.90 bar is not
   reached because far limbs reuse the near limb's art, a rig design limit. The rough-cut fallback keeps the best
   try.

## Owner's idea 2 (front + side views): not built
The joint finder already works well on the template poses, which keep the limbs apart. Fusing two views is worth
it only if characters with loose robes, capes or big props still cut badly. Revisit with real failures.

## Pose-driven frame model (next, needs a decision on downloads)
The klein "drawn frames" path draws each frame as a new picture, so the design drifts. Options found:

| Model | Base | Driven by | VRAM | Code / weights licence | diffusers |
|---|---|---|---|---|---|
| AnimateDiff + ControlNet OpenPose (+ IP-Adapter for the design) | SD 1.5 | OpenPose images (we render them from clips) | about 8–10 GB for 16 frames at 512 px; 8 GB possible with fp16 and offload | Apache-2.0 (motion adapter); CreativeML OpenRAIL-M (SD 1.5) | yes: `AnimateDiffControlNetPipeline` is in the installed diffusers 0.42 |
| MimicMotion | SVD-xt 1.1 | DWPose images | 8 GB for the 16-frame U-Net with the VAE decoder on the CPU (16 GB on GPU) | Apache-2.0 code; SVD weights under the Stability community licence | no (own scripts) |
| Champ | SD 1.5 | SMPL depth, normal and semantic maps + DWPose | about 20 GB for its default run | MIT code | no |
| MagicAnimate | SD 1.5 | DensePose | not stated | BSD-3 code | no |

Recommendation: spike AnimateDiff + ControlNet OpenPose + IP-Adapter in diffusers first.
- It is native diffusers (hard rule 4).
- It takes the OpenPose stick images GGS already renders from clips.
- It is likely to fit the 8 GB profile at 512 px and 16 frames.

MimicMotion is the fallback if identity drifts too much: it is stronger, but slower and not diffusers. The extra
downloads are about 6–8 GB (SD 1.5, motion adapter, OpenPose ControlNet, IP-Adapter + image encoder).
