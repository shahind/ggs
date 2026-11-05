# Rig Spec v1: side-view humanoid cut-out puppet

Source of truth: `rigs/humanoid_side_v1.yaml` (validated by `ggs rig validate`). This document explains the why.

## Conventions
- Screen space: +x right, +y down. Angles in degrees, clockwise from +y (straight down = 0).
- Character faces right. The near side is the character's right side. Far limbs reuse near-side art, darkened.

## Bones (15) and what was dropped
pelvis (root), torso, head; per side: upper_arm, forearm, hand, thigh, shin, foot.
Dropped from mocap skeletons: fingers, toes, collarbones, twist joints; extra spine joints merge into the torso.
A later version may split the torso into chest and belly so it can bend.

## Part sheet
4x3 grid of 256px cells = 9 unique parts + 3 head variants (blink, mouth_open, hurt).
Limbs are drawn straight with the joint at the cell's pivot and overlapping caps so joints hide seams.
The normalizer crops each part, scales it to its bone length and snaps the pivot, so the generator does not need
pixel-exact placement. Validation reassembles the master pose and compares it with the master image.
The rig's pivots are defaults: a character.yaml may store its own `pivots` per part (the cutter centres each part
in its cell). See `docs/PART_CUTTER.md`.

## Part orientation
Each part has an `art_angle`: the direction its bone points in the drawing, using the same convention as bone
angles. Limbs hang down (0), the head, torso and pelvis point up (180) and the foot points forward (-90).
The renderer rotates a part by `bone angle - art_angle` about its pivot.

## Bone attachment
Each bone starts at `attach` along its parent (0 = parent start, 1 = parent end, default 1). The pelvis points up
from the hip joint, so the thighs use `attach: 0`. This is a cut-out simplification: the real hip joints sit a little
below and beside the hips pivot.

## Retargeting (3D mocap -> 2D bones)
Code: `src/ggs/rig/bvh.py` (parser + forward kinematics), `src/ggs/rig/retarget.py`. Output: `docs/CLIP_FORMAT_V1.md`.
For each frame: take the 3D position of each bone's (from, to) joints from the mocap profile, drop the depth axis
(the camera looks along it), and compute the 2D angle of that vector. Root translation drives the pelvis position.
- Depth axis = horizontal far-hip -> near-hip (and far-shoulder -> near-shoulder) direction, averaged over the clip.
  Screen right = up x depth, which is the character's forward direction. No per-frame facing: turns stay visible.
- Scale = rest-pose height of the file's skeleton; positions are in standing heights, ground = lowest joint.
- Angles are unwrapped over time (no 359 -> 0 jumps). Local angles are wrapped to (-180, 180].
- Depth also decides which limb is nearer the camera. Per near/far group the clip stores a `swaps` flag (with a small
  hysteresis so it does not flicker); the renderer swaps the group's draw-order slots.
- A T-pose first frame (both arms straight out sideways) is dropped by default; the CMU BVH release adds one.
- Mocap profiles map each bone to two joints. Never pick a joint and its zero-offset child (zero-length bone).
  BVH End Sites are addressable as `<Parent>_End`.
Known limits: flips, rolls and turns lose information when projected; the retargeter warns on turns over 45 deg,
on bones projected below 35% of their length (pointing at the camera) and on feet that point backwards
(left-handed or mirrored files). Expect a curated clip set.

## Equipment
Held items are plain sprites in a standard orientation (handle at the bottom) with a grip marker, attached to the
`grip_near` / `grip_far` sockets. Hold poses (one-hand, two-hand, pistol, rifle) plus two-bone IK place the hands.
Aiming and swings use short keyframed clips, not mocap.
Built in M7 (`src/ggs/rig/equipment.py`, `layers.py`):
- `ggs generate --class held_item` draws items upright with the handle at the bottom; `ggs item make` turns one
  into `item.yaml` (grip on the centre line 12% up, second grip 20% higher for two-hand items, size from
  `--length` in standing heights). The item points along hand angle + `hold_angle` (-90: forward when the hand
  hangs down) and is drawn right after its hand.
- Keyframe clips (`clips/*.yaml`: `hold_one_hand`, `hold_two_hand`, `swing_one_hand`) become clips with
  `ggs clip keys`; angles are absolute for an upright body and ease in and out between keys.
- Layers: `ggs clip layer walk.json --over swing.json --bones near_arm|arms|upper` plays the overlay's angles,
  relative to each parent bone, over the base clip, so a swing stays on a torso that walks.
- Two-hand items: export solves the far arm with two-bone IK every frame so the far grip socket lands on the
  item's second grip (the far hand copies the near hand's angle).

## Mounts and vehicles
- Horse-like mounts get their own quadruped side-view rig spec (~14 parts) with built-in walk, trot and gallop.
- Cars and bikes are body + wheels with a bounce; no skeleton needed.
- The rider uses a seated pose locked to the mount's `seat` socket. Animation layers: lower body = pose,
  upper body = any clip (shoot, swing). The mount's gait phase drives a small rider bob.

## Exports
GIF / APNG / WebP (prefer APNG or WebP for soft alpha), sprite sheet + JSON atlas, or a rig package
(parts atlas + skeleton + clips) with an engine importer.
