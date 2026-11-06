# Open-source options for slicing, warp and pose-driven frames (research, 2026-11-05)

Owner's request: before building more home-made code, find what the open-source world already uses for accurate
2D slicing, warping and pose-driven animation, and pick the best motion model (better than MimicMotion).
Licences and VRAM below are as stated on the official pages on 2026-11-05; nothing here is cleared for release
(see LICENSE_REVIEW.md).

## Why our slicer and warp keep failing
Commercial and open 2D rigs (Spine, Live2D, Rigby2D, DragonBones) all start from **complete layers**: an arm
layer that also contains the part of the arm hidden behind the body, a torso layer that also contains the chest
under the arm. With complete layers, cutting is trivial and the warp never needs fills. GGS cuts one flat picture,
so it must guess ownership and invent hidden pixels; every artefact so far (triangles, flaps, holes, shoulder
copies) comes from that guess. The fix is to **generate** complete layers, not to cut better.

## Slicing: layer decomposition
| Project | What it does | Licence | VRAM | Fit for GGS |
|---|---|---|---|---|
| **See-through** (SIGGRAPH 2026, shitagaki-lab/see-through) | One anime picture -> up to 23 semantic layers (hair, face, eyes, clothing, limbs...), **hidden regions inpainted**, draw order from a fine-tuned Marigold depth; writes a PSD | code Apache-2.0; LayerDiff 3D and Marigold weights openrail++ ("Commercial use is permitted", Open RAIL use restrictions must be passed on); body-part SAM-HQ Apache-2.0 | 12-16 GB bf16 at 1280 px; **~8 GB** with the NF4 or block-swap scripts | Best match. SDXL + LayerDiffuse based (diffusers pipeline types). Trained on anime / Live2D (front-facing): must be tested on our side-view, full-body, non-anime drawings |
| anime-part-split (HF tsunehimatoi) | similar, repaints each part whole | research / personal only | - | not usable (licence) |

Downstream editors that already consume See-through PSDs: Rigby2D (MIT/Apache, auto-rig from a layered PSD:
joints where parts meet, weighted meshes, IK; early alpha), StretchyStudio, Anime2.5DRig.

## Warp: deformation
- **As-Rigid-As-Possible (ARAP)** is what Photoshop Puppet Warp and Meta's Animated Drawings (MIT, archived
  2025-09) use: pins (joints) move, the mesh keeps each triangle as rigid as possible. Our connected warp uses
  linear blend skinning; switching the solve to ARAP removes the stretched outlines at bends.
- With complete layers (above), each layer is its own connected ARAP mesh drawn in depth order: no tearing (the
  layer is whole), no fills (hidden parts exist), arms pass over the chest correctly.

## Pose-driven frames (instead of AnimateDiff / MimicMotion)
| Model | Base / size | Licence | Input | VRAM (stated) | Notes |
|---|---|---|---|---|---|
| **Wan-Animate-2** (Wan-AI, 2026-08-07) | Wan2.2, 14B (16B stored); base 40 steps, distilled 10 steps | Apache-2.0 | reference image + **raw driving video** (no pose extractor) + Chinese appearance prompt | tuned for 8x A800 (720p), 480p tested on 2x A800; quantised community files ~17 GB int8 | diffusers `WanAnimate2Pipeline` (from source). Best quality claimed; too big for 8 GB without heavy GGUF quantisation + offload |
| **One-to-All Animation** (CVPR 2026) | Wan2.1, **1.3B** and 14B | Apache-2.0 (code + weights) | reference + pose sequence; **alignment-free**: reference and pose may differ in layout and proportions; trained with cartoon data | none stated; 1.3B runs on a 16 GB T4 (10 s 832x480 in 11 min) | Best fit for the 8 GB profile; own inference code on a Wan2.1-1.3B diffusers base. Beats MimicMotion, StableAnimator, UniAnimate and Wan-Animate on identity in its paper (self-reported) |
| SCAIL-2 (zai-org, 2026) | Wan2.1, 14B | code Apache-2.0, weights tagged MIT | reference + mask + driving video or pose | none stated; offload only | needs masks; 14B only |
| Wan2.2-Animate-14B | Wan2.2, 14B | Apache-2.0 | reference + preprocessed pose and face videos | offload + dtype convert | diffusers `WanAnimatePipeline` |
| MimicMotion | SVD | - | DWPose | - | outclassed by all of the above |

For GGS the driving video can be rendered from the clip itself (the mannequin or the warp), so the models that take
a raw driving video (Wan-Animate-2) get a clean, exact motion source.

## The 3D route (what many asset websites do)
Image -> 3D (TRELLIS.2, Hunyuan3D, Pixal3D) -> auto-rig (UniRig code MIT, Puppeteer Apache-2.0) -> mocap ->
render side-view sprites. AssetFurnace (Apache-2.0) packages this locally. Perfectly consistent frames, but the
look is a rendered 3D model, not the drawing; weights licences vary (Hunyuan3D excludes EU/UK/KR). Worth a spike
only if the 2D routes fail.

## Recommendation (order)
1. **Spike See-through on Jones** (8 GB NF4 mode): if side-view layers come out whole, the slicer becomes
   "decompose -> map layers to bones", and the warp becomes per-layer connected meshes.
2. **ARAP solve** for the warp mesh (small, no model).
3. **Generated frames: One-to-All 1.3B** on the 8 GB laptop, **Wan-Animate-2 distilled** for 16 GB+ users,
   both driven by a video rendered from the clip.

Sources: github.com/shitagaki-lab/see-through, huggingface.co/layerdifforg, huggingface.co/24yearsold/l2d_sam_iter2,
github.com/facebookresearch/AnimatedDrawings, github.com/maruccimauro/rigby2d, huggingface.co/Wan-AI/Wan2.2-Animate-2-14B,
huggingface.co/Wan-AI/Wan2.2-Animate-14B, github.com/ssj9596/One-to-All-Animation, huggingface.co/MochunniaN1,
github.com/zai-org/SCAIL-2, github.com/Bingeljell/image-to-3dlab, github.com/VAST-AI-Research/UniRig, github.com/Seed3D/Puppeteer.
