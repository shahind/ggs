# Body layouts: what a character is made of, how it is cut, posed and animated (proposal, 2026-11-05)

Status: **built** (2026-11-05): layouts, capture poses, suggestion, layout-aware cutting and warp, follow-through
motion, the Sliced Version tab's layout picker. Body parsing (SegFormer-B2 clothes, `ggs.rig.parsing`) was added
after the owner's third review: trusted classes (face, hair, hat, shoes, arm and leg skin) decide their parts. It replaces the three "slice templates" (side
view, side view detailed, front view). Those differ only by view and an extra repaint pass; they say nothing
about what the character is made of, and that is what decides how it can be cut and animated.

## 1. The kinds of characters to cover
All side view (the motion library is side view). Non-humanoids (quadrupeds, birds, slimes) need their own rigs
and are out of scope here.

| # | Kind | Typical look | What makes it different |
|---|---|---|---|
| A | Chibi / big head | head 1/3 of the height, no visible neck, short limbs | the head sits on the body; hands often mitten-like |
| B | Standard cartoon | head about 1/6, short neck under the chin | the neck goes with the head |
| C | Tall / realistic | head about 1/7–1/8, visible neck, long torso | separate neck; the torso bends (chest + waist) |
| D | Skirt / dress / robe / long coat | any of A–C | cloth over the thighs; the legs are partly or fully hidden |
| E | Long hair / cape / scarf / tail | any of A–C | a loose part behind the body that should follow through |
| F | Bulky / armoured | wide torso, thick limbs | elbows and knees hard to see; the warp suits it better than cutting |

D and E combine with A–C (a tall robed wizard with a cape = C + D + E). So a layout is **a few switches**, not
one fixed template per kind.

## 2. What the detector can find (and what it cannot)
| Body feature | How GGS gets it | Reliability |
|---|---|---|
| Shoulders, elbows, wrists, hips, knees, ankles | ViTPose keypoints | good when limbs are apart; weak when an arm lies over the body |
| Head position / facing | ViTPose nose, eyes, ears | good |
| Neck | no keypoint: the skin region between collar and chin (region cutting) + ear-to-shoulder gap | medium; fine for "neck with head" vs "separate", poor on tiny necks |
| Chest / waist split | no keypoint: the belt line (a dark horizontal band) or halfway down the torso | estimate |
| Hands / feet as separate pieces | wrist / ankle keypoints + the hand's or boot's outlined region | good |
| Fingers, toes | none | not cut (the hand and foot are one piece each) |
| Hair beyond the head | outlined region owned by the head, reaching past the head's width or below the jaw | good on flat art with outlines |
| Skirt / dress / coat | cloth rule: a region on the thighs reaching past their width, hanging from the waist | good (working now) |
| Cape / scarf | region behind the back, past the torso's width, not a sleeve | medium |
| Held props | separate items (M7), not part of the body | n/a |

Conclusion: these are realistic to cut:
- head (with or without neck), neck, one- or two-piece trunk, three-piece arms and legs
- back hair, skirt, cape

Fingers, toes and face features are not; expressions stay a later, separate step.

## 3. A layout = switches (stored with the character)
```yaml
layout:
  neck: head        # none (head sits on the body) | head (neck drawn with the head) | separate
  trunk: one        # one | two (chest + waist)
  hands: separate   # separate | with_forearm (mitten hands, small chibis)
  extras: [skirt]   # any of: hair_back, skirt, cape
```
Presets offered first: **Chibi** (none / one / with_forearm), **Standard** (head / one / separate), **Tall**
(separate / two / separate). The extras can be ticked on any preset.

The rig becomes **one superset skeleton** (`humanoid_side_v2`):
- `pelvis`, `waist`, `chest`, `neck`, `head`, three-bone arms and legs (near and far), and `hair_back`,
  `skirt`, `cape`.
- A layout switches bones off or merges them into a neighbour:
  - `neck: head`: the neck's art belongs to the head piece, whose pivot is the base of the neck.
  - `neck: none`: no neck bone; the head pivots at the chin line.
  - `trunk: one`: waist and chest are one piece on the chest bone.
- Everything downstream reads the layout, so the software always knows "this one has no neck". That covers the
  cutter, the warp, the validator, the editor and the exporters.
- Existing characters (rig v1) load as Standard with no extras.

## 4. The software-chosen pose (the "animation-ready drawing")
The design pose stays the user's choice. Animation is made from **one extra drawing in a pose GGS picks for
the layout**: the pose guide redraws the design (it does this well already). Both the cut-out and "Bend the
drawing" use this drawing. Using the design pose directly stays available as "advanced".

| Layout | Capture pose (side view, facing right) | Why |
|---|---|---|
| all | near arm 40° forward of vertical, elbow bent 20°, hand clear of the chest; far arm 30° back | the near arm does not cover the chest; the far arm shows behind |
| all | legs in a stride ±22°, knees slightly bent, feet flat | the thighs separate, the knees show |
| skirt | stride ±14°, near arm 50° forward | the feet still show under the skirt; the arm clears the skirt |
| hair_back / cape | head level; arms stay off the back | the hair and cape stay a separate, readable region |
| chibi | arms 45° out from the body | short arms otherwise touch the big head and body |
| neck: separate | head slightly raised (chin up 5°) | the neck shows between chin and collar |

The user's body settings (head size, limb lengths, build) still shape this drawing, so custom proportions keep
working.

## 5. Where each part sits, and its drawing order
- **Pivots** are the fitted joints of the animation-ready drawing:
  - neck base = top of the chest
  - shoulders = top of the chest, front edge (both arms attach there)
  - hips = pelvis start
  - extras pivot where they hang from: back hair at the back of the skull, skirt at the waist line, cape at
    the shoulders' back edge.
- **Pieces overlap** their neighbour across each joint with real art (as now), so bends show no gap.
- **Drawing order** (back to front), switched by the clip's near/far swaps as now:
  1. cape
  2. back hair
  3. far arm
  4. far leg
  5. (both legs, when there is a skirt)
  6. waist / pelvis
  7. skirt
  8. chest
  9. neck
  10. head
  11. near leg (no skirt)
  12. near arm

  Each piece's position in this order is stored in `character.yaml`, so Godot, Unity and Defold follow it.
- The **warp** uses the same pieces as layers (cape, hair, far limbs, body, near limbs), so a cape or hair can
  bend on its own instead of being dragged by the torso.

## 6. How each layout animates
Library clips move the 15 v1 bones. The extra bones get their motion **computed and stored in the clip**
("baked"), so every engine just plays bone tracks and nothing needs physics at run time:
- **chest / waist**: the clip's torso bend is shared 60 % chest, 40 % waist (a curved back instead of a
  plank).
- **neck**: halfway between the torso's and the head's angle (the head turns, the neck follows).
- **skirt**: hangs from the pelvis, swings with 40 % of the legs' mean swing; the hem follows the forward leg.
  This is what the warp does now.
- **back hair, cape**: follow-through. Each is a damped spring driven by its parent's turning and the
  character's movement (running pushes the cape back; stopping lets it swing forward).
- The live editor shows the computed motion and lets the user touch it up like any other bone.
- AnimateDiff "Drawn frames" are not affected: they follow the OpenPose skeleton and draw the clothes
  themselves.

## 7. The software suggests the layout
After the joint finder runs on the animation-ready drawing, GGS measures and proposes:

| Measure | Proposes |
|---|---|
| head height ≥ 0.22 of the body | Chibi (`neck: none`) |
| skin gap between jaw and collar ≥ 0.04 of the height | `neck: separate`, else `neck: head` |
| body ≥ 6.5 heads tall | `trunk: two` |
| cloth share ≥ 0.15 (cloth rule) | `skirt` |
| head-owned region past the head's width or below the jaw | `hair_back` |
| region behind the back past the torso's width | `cape` |
| limbs much wider than the guide's (bulky) | recommends "Bend the drawing" over the cut-out |

The user sees the suggested layout with its pieces, can switch presets and tick extras, and the choice is
saved with the character.

## 8. Build order
1. **Layouts and rig v2.** Superset skeleton, layout switches in `character.yaml`, clip mapping for
   chest/waist/neck, the capture poses and the animation-ready drawing step. Old characters still load.
2. **Cutting and warp follow the layout.** Neck merge rules, chest/waist split at the belt line, hair, skirt
   and cape as separate pieces/layers, the layout's drawing order.
3. **Follow-through motion.** Skirt, hair and cape tracks baked into clips; all three engine exporters carry
   the extra bones; the editor shows them.
4. **Suggestion + UI.** The measurements, the suggested layout with a pieces preview, preset buttons and extras
   ticks on the character's Sliced version tab.

Risks:
- Hair and cape detection depends on clean outlines; painterly art falls back to the warp.
- A separate neck is only offered when the neck is visible, otherwise it would be a sliver.
