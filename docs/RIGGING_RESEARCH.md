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

## Follow-up (owner, same day): rigging models, hidden parts, MimicMotion first
**2D rigging models / LoRAs:** none found in open source. What exists is prompt workflows ("parts sheet" prompts
for any image model) and paid services (Layer.ai's Spine/Live2D parts-sheet tool). Character-sheet LoRAs
(turnarounds) do not output separated limbs.

**Image to layers / hidden parts:**
- Qwen-Image-Layered (Apache-2.0, Dec 2025): splits a picture into a chosen number of RGBA layers; not by body
  part, and 16 GB+ in practice.
- Amodal completion (complete what is hidden): Open-World Amodal Appearance Completion (CVPR 2025,
  saraao/amodal, Apache-2.0) inpaints the occluded part of an object with SD2 inpainting and returns RGBA.
- GGS already has an editing model (FLUX.2 klein) that can inpaint a masked area from a text instruction.

**Planned combination (owner's direction):**
1. Body-part segmentation: See-through's body-part parser (SAM-HQ, Apache-2.0, 19 parts) or our SegFormer
   parse, decides arm / torso / leg / head / hair / clothing and which side (near/far).
2. The fitted joints (ViTPose) split a limb region along the bone: See-through merges thigh and shin, but the
   knee joint says exactly where to cut.
3. For each part, the pixels covered by parts in front (the chest under the arm) are masked and completed by an
   inpainting model (klein edit, or the amodal-completion model), so every layer is whole.
4. Whole layers -> rig (cut-out) or per-layer connected meshes (warp), drawn in depth order.

**MimicMotion facts (for the spike):**
- Code Apache-2.0; weights "other", fine-tuned from Stable Video Diffusion.
- Needs `stabilityai/stable-video-diffusion-img2vid-xt-1-1`, which is **gated**: a Hugging Face login that has
  accepted the Stability AI Community License (free under US$1M annual revenue, registration for commercial use,
  "Powered by Stability AI" attribution, no use to train foundation models).
- 16-frame U-Net: 8 GB minimum, VAE decoder on the CPU (16 GB on the GPU). 576 px short side; DWPose for poses.
- Size versus One-to-All 1.3B: SVD-xt U-Net about 1.5B parameters, download about 8 GB in all. Wan2.1 1.3B is a
  smaller network, but its umT5-xxl text encoder adds 6-11 GB of download; VRAM is similar if the text encoder
  is streamed as GGS does for klein.

## Built (2026-11-05, evening)
- **Slicer, hidden parts:** each piece now gets its full shape where a part in front covered it (`hidden_shape` in
  `ggs.rig.cutter`): limbs along the bone at their width, torso and hip inside their own outline. The klein
  refiner paints those areas on every slicing, keeping visible pixels exact. On the earlier boy test copy the
  chest and hip under the near arm and thigh came out complete; 6 pieces took 7 min 48 s on the 8 GB laptop.
  Still to do: a better body-part segmentation (See-through's SAM-HQ parser) for the regions themselves.
- **Generated Frames: One-to-All 1.3B** (`ggs.pipeline.one_to_all`, code vendored in `ggs.third_party.one_to_all`).
  The design on a 384x576 canvas, the clip's skeleton in the design's proportions drawn in One-to-All's pose
  style, 4k+1 frames, 30 steps with reference and pose guidance, matted with BiRefNet. umT5 is streamed and its
  embeddings cached per prompt. Default method when its models are downloaded (`frames_o2a` pack, 18 GB).
- **One-to-All on the owner's Jones (8 GB laptop):**
  - Clips of 13 frames grey out after frame 3: a grey background and low contrast, which matting then makes
    see-through. 17, 25 and 33 frames hold full colour to the end.
  - Guidance strength and float32 timestep layers made no difference.
  - GGS now always draws at least 17 frames and keeps the number asked for.
  - Walk and punch, 12 frames each: about 245 s per clip, 6.8 GB peak. The design is kept, and the walk and the
    jab follow the clip; fists blur a little at the punch's peak.

## Layered output instead of a sheet (owner, 2026-11-05 night)
The owner's idea: get the character from the model already as layers (body parts separate, hidden parts
complete), usable by both the warp (one connected mesh per layer) and the cut-out. Hand-made template sheets
are not practical to prepare.

| Model | What it outputs | Released? | Licence | Fit |
|---|---|---|---|---|
| Bunraku (arXiv 2607.27348, Jul 2026) | from one illustration: ordered RGBA layers (8 classes / 32 subclasses), hidden parts completed, a mesh per layer and keypose deformations, i.e. a Live2D-style rig | **no**: the repo holds only a README (no code, weights or licence); dataset by request | paper CC BY-NC-SA | the ideal target, not usable yet |
| See-through (SIGGRAPH 2026) | from one image: up to 23 semantic layers (hair, face, clothing, limbs...), hidden parts inpainted, draw order from depth, PSD | **yes** | code Apache-2.0; weights openrail++ (commercial allowed with use restrictions) | the usable one today. Anime-trained and mostly front-facing; limbs come as whole layers (thigh+shin), which GGS can split at the known knee joint because the layer is complete |
| Qwen-Image-Layered (Dec 2025) | an image into N RGBA layers (by depth/objects, not body parts) | yes | Apache-2.0 | 20B, generic layers; Bunraku fine-tuned it on ~10K Live2D models we do not have |
| OmniPSD, PSDiffusion, ART+ | posters: foreground / midground / background layers | partly | OmniPSD on FLUX.1-dev (non-commercial) | not characters |

No model generates a layered character from a prompt directly. The practical pipeline:
1. The prompt draws the design (klein, in the capture pose).
2. See-through splits it into complete layers.
3. GGS maps the layers to bones and splits limb layers at the fitted joints.
4. The rig is built from the layers: per-layer meshes for the warp, pieces for the cut-out.

Risk to test first: side-view, full-body, non-anime drawings (See-through's training data is anime Live2D, mostly
front-facing).

## See-through in the slicer (built, 2026-11-06)
- **Setup:** layer model vendored (`ggs.third_party.see_through`, Apache-2.0 code), wrapped in
  `ggs.pipeline.layers`. NF4 weights in the optional `layers` pack (3.6 GB, openrail++); needs bitsandbytes.
  Jones: 412 s at 768 px, 4.6 GB peak.
- **What it gives:** complete layers. On Jones: both arms as one layer (sleeve to hand), the whole shirt, the
  whole trousers, face, front and back hair, ears, eyes. In one run the hat and boots got no layer.
- **Slicing:** after the tries are cut, the best try gets layers (`<try>/layers`, reused on re-cut) and is cut
  again from them (`ggs.rig.cutter._layered_parts`):
  - Each layer's pixels go to the nearest of its allowed bones (`LAYER_BONES`): arms are split at shoulder,
    elbow and wrist; trousers at hip and knee. Far-side bones are penalised (their art is never shown).
  - A piece is its layers composited back to front, so hidden parts are already drawn.
  - Across a joint, only the piece drawn underneath continues into its neighbour (same layer's art; strip
    width from the narrower bone, so a hat-wide head borrows nothing).
  - Pixels the layers do not reproduce (by colour) keep the drawing's colour and go with the nearest explained
    pixel: the hat with the hair, a collar with the shirt. A bone no layer drew (boots) takes what lies nearest
    to it.
  - The layered cut is kept only when it validates at least as well as the plain cut. The klein hidden-area
    painting is skipped when it is kept.
- **Jones:** passes, 0.904–0.910. The head piece is clean (hat, face, neck, no shirt) and the shirt is whole.
  Still rough: pointed overlap ends on the leg pieces, a notch at the back of the shoulder, a sliver above the
  upper arm.
