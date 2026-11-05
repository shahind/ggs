# Master image -> part sheet (M5: fallback cutter, normalizer, reassembly validator)

Code: `src/ggs/rig/master.py` (pose guide), `src/ggs/rig/cutter.py` (labels, cut, fill, normalize),
`src/ggs/rig/validate.py` (reassembly QA). CLI: `ggs character master | cut | validate`.

## Idea: control the pose instead of detecting it
Finding joints and part boundaries in an arbitrary drawing needs pose and segmentation models that do badly on
stylised game art. GGS draws the master on top of a **pose guide** instead: the programmer-art puppet rendered in
a fixed *master pose* (side view, facing right, near arm a little forward and bent, legs in a small stride so near
and far limbs separate). Because GGS renders the guide, it knows every joint position and every part's mask in it.

1. `ggs character master`: render the guide, then FLUX.2 klein redraws it as the described character with the guide
   as reference image ("same pose, same size and position"). BiRefNet gives real alpha. Writes `master.png`,
   `guide.png`, `guide_labels.png` and `master.json` (rig, pose angles, joints in pixels, pixels per height).
2. `ggs character cut`: master + guide -> part sheet + `character.yaml` + `validation.json`.
3. Validation rejects the result (exit 2) when the master did not follow the guide or a cut is bad; the next seed
   is tried.

The roadmap planned SAM for step 2. With a known pose the guide masks already give the part boundaries, so no
segmentation model is needed (deterministic, nothing to download). SAM stays an option for masters that come from
elsewhere (a user's own drawing), which also needs joint picking in the UI (M9).

## Cutting
- **Labels:** every opaque master pixel gets the part of the nearest guide pixel (the top-most part where parts
  overlap). Hair, capes and wide sleeves therefore join the nearest part.
- **Parts:** only the near side, torso, pelvis and head are cut; far limbs reuse the near art (rig spec).
- **Hidden areas:** torso and pelvis are partly covered by the near arm and thigh. The covered area inside the
  part's convex hull is filled.
- **Joint caps:** a limb gets a disc at its start joint (radius = half its narrowest width there). A parent gets
  a capsule continuing its bone under the part drawn on top, up to and round the joint, limited to where the
  master is opaque. Bent joints show no gap, and nothing bulges past the master silhouette. Head and torso get no
  start disc (they turn little; the parent's capsule covers them).
- **Fill:** numpy onion-peel fill from the part's own pixels (good enough for flat game art). A diffusion
  inpainter can replace it later without changing the file formats.

## Normalizing
Each part is rotated about its start joint so the bone points along the rig's `art_angle`, then all parts are
scaled by one shared factor (the largest that fits every part in its cell) and placed at the cell pivot. Bone
lengths come from the master's joints, in standing heights. Head variants (blink, mouth open, hurt) are copies
of the head until an edit model draws them (M6).

## Validation (`validation.json`)
| Check | What it catches | Pass |
|---|---|---|
| Reassembly IoU | the puppet in the master pose vs the master silhouette: crop, fill and scale errors | >= 0.90 |
| Reassembly colour | mean colour difference on near/core parts | <= 0.10 |
| Guide agreement | per part, IoU of the guide mask and the master pixels labelled with it: master ignored the pose | >= 0.40 |
| Joint interface | each joint lies on the border between its two parts: wrong joint positions | <= 0.35 x part width |
| Connected parts | a part split into pieces: mislabelled pixels | largest piece >= 85% |
| Cell fit | a part clipped at its cell edge | no clipping |
| Test poses | arms and legs bent far from the master pose: a limb comes loose (joint gap) | one silhouette |

## Body shapes and stature
Characters do not share one body. `--body` picks the guide's proportions, and with them the master's, the cut's
and the animation's: a preset (`standard`, `heroic`, `chibi`, `stocky`, `lanky`, `child`) or ratios such as
`head=1.6,legs=0.7,arms=1.1,torso=1,width=1.2`. The shape scales the rig's rest lengths and part widths and is
then rescaled to one standing height, so only proportions change. Its words ("chibi proportions: a very large
head...") are added to the master prompt so the generator follows the guide. The shape is always an explicit
flag, never read from the prompt (deterministic backend).
- The cutter measures bone lengths from the guide joints, so `character.yaml` gets the character's own lengths.
  The animator already plays any clip on any lengths: angles come from the clip, and the root track is scaled by
  the leg-length ratio so the feet stay on the ground.
- `--stature` (character.yaml `stature`) is how tall the character is in the game relative to others: exports
  render it at `--height x stature` pixels.

## Refining (M6)
`ggs character refine <character folders>` (run by `character master` on every valid sheet unless `--no-refine`):
klein edits the cut parts zero-shot, in one model load. Code: `src/ggs/rig/refine.py`.
- **Fills:** a part whose filled-in share (`parts_filled.png`, written by the cutter) is at least 8% is repainted
  ("repaint the blurry smudged areas ... keep everything else the same"). Only the filled-in pixels take the
  repaint and the part's alpha is kept, so everything the master shows stays pixel-exact. A repaint that turns
  more than 30% of the filled area white is refused.
- **Expressions:** each head variant is an edit of the neutral head (blink, mouth open, hurt), matted with
  BiRefNet and kept only if it has the head's shape and place (IoU >= 0.85) and colours (error <= 0.15);
  otherwise the cell keeps the copy.
- `refine.json` lists every edit and whether it was applied; `parts_cut.png` keeps the cutter's sheet, and
  refining again starts from it.

The trained rig-sheet adapter planned for M6 is built but not the default: `ggs adapter pairs | train | eval`
(pair factory, edit-LoRA trainer, comparison against the cutter). Its zero-shot training sheets were mostly
restyled mannequins, so it waits for a better data source and a bigger GPU.

## Master poses
`--pose walk` (default: arms down, small stride) or `--pose spread` (near arm forward, far arm back, wide
stride). Spread hides less (the belt no longer meets the hand), but long robes become a triangle between the
legs and accessories in front of the arm (a beard) still get split. Neither pose makes the cutter content-aware.

## Known limits of the cut-out route (owner review, 2026-11-04)
- Labels are geometric (nearest guide part), not content-aware: a belt next to the hand, a beard over the arm or
  a robe over both legs get split by position, not by what they are. A clothing/part segmentation model (human
  parsing) or garment parts in the rig (a skirt/robe piece on the pelvis) are the candidate fixes.
- klein cannot draw a character's parts separately from a master zero-shot (probe: it redraws the mannequin
  sheet with stray bits of the character).

## Frame-by-frame sprites (experimental)
`ggs character sprites master.json --clip walk.json --out dir --count 3`: the clip's poses are rendered as a
pose sheet (the guide puppet in the character's body shape, 8 poses in a 4x2 sheet) and klein draws the
character from the master in every cell, all frames in one image. Cloth, hair and capes move as the model draws
them; the result is a fixed animation (APNG, WebP, sprite sheet + atlas), not editable parts.
Each frame is checked (follows its pose, master's colours, same height) and the best seed is exported.
Status: parked by the owner ("not good, not bad"). First runs: the dwarf and knight stay consistent across
frames, the wizard's robe comes and goes; the colour check (palette histogram distance <= 0.35) refused most
frames and is not calibrated yet.
