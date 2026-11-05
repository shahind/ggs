# Clip format v1 (`ggs.clip`)

A clip is a 2D animation for one rig. It is what `ggs mocap retarget` writes and what the renderer and exporters
(M2) read. Model + validation: `src/ggs/rig/clip.py`. JSON, one object:

| Field | Meaning |
|---|---|
| `format`, `version` | `"ggs.clip"`, `1` |
| `rig` | rig spec id the clip was made for, e.g. `humanoid_side_v1` |
| `source`, `profile` | source file and the `mocap_profiles` entry used |
| `fps`, `frame_count` | playback rate and length |
| `in_place` | `true` when forward travel was removed from the root |
| `loop`, `loop_travel` | `true` for a seamless cycle (`ggs clip loop`); travel per loop in standing heights, for root motion |
| `bones` | bone names in rig order |
| `bone_lengths` | median 3D length of each mocap bone, in standing heights (the character's own lengths come from its part sheet) |
| `angles` | per bone, per frame: absolute angle in degrees, clockwise from +y (down = 0, screen left = 90, screen right = -90). Unwrapped over time, so values can leave -180..180 but never jump |
| `local_angles` | per bone, per frame: angle relative to the parent bone, wrapped to (-180, 180]. Root = absolute |
| `root` | `x`, `y` per frame: position of the root bone's start (the hips), in standing heights. x = travel from frame 0 (+x = forward = screen right), y = -(height above ground), so +y is down and the ground is y = 0 |
| `swaps` | per near/far limb group (`thigh`, `upper_arm`), per frame: `true` when the far limb is closer to the camera, so the renderer must draw that pair in swapped order (`ggs.rig.retarget.draw_order_for`) |
| `warnings` | projection problems found while retargeting (turns, bones pointing at the camera, mirrored files) |

Rebuilding a pose (`ggs.rig.pose.pose_frame`): place the root at `root[x, y]`; each other bone starts at
`attach` (from the rig spec) along its parent and points along its absolute angle,
`(dx, dy) = length * (-sin a, cos a)`.

Units are standing heights so a clip fits any character size: multiply by the character's height in pixels.
