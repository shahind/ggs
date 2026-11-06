# A trained, rig-aware slicer (plan, 2026-11-05)

Owner's request: a slicer that knows *how a rig can be cut*. The cuff goes with the forearm, the hat with the
head, the boot with the foot, the sleeve with the upper arm. It also knows what each part looks like where it is
hidden. General segmenters (SegFormer ATR, See-through's 19 classes) cut by clothing semantics, not by bones, and
See-through merges thigh and shin. So we train our own.

## What exists (none of it does this end to end)
| Work | What it gives | Usable for |
|---|---|---|
| See-through (SIGGRAPH 2026; Apache-2.0 code, SAM-HQ body-part parser Apache-2.0) | 19 semantic parts plus hidden regions, trained on commercial Live2D models (data not released) | a starting checkpoint (SAM-HQ) |
| "How to Train Your Dragon" / AniDiffusion (Eurographics 2025) | procedural 2D shapes, one closed shape per bone, random layer order; evaluation set renders each body part separately to expose hidden areas | the data idea: build images part by part, so ownership and hidden areas are known exactly |
| ChildlikeSHAPES (2025, Meta; 16,075 drawings, 25 part classes, CC-BY 4.0 per Animated Drawings) | hierarchical part masks of figure drawings | pre-training on drawn bodies |
| Synthetic humans (SURREAL-style Blender renders) | exact per-part masks from posed 3D bodies | the method for our data |

## The model
1. **Rig-aware amodal part segmenter.**
   - Input: the drawing plus our skeleton (we know the joints from the pose guide and ViTPose), as bone-coded
     channels, so the model cuts *for this rig*.
   - Output, per part of the layout: the visible mask and the full mask, including what is hidden.
   - Parts: head (with hat and hair), neck, chest/torso, pelvis, upper arm (with sleeve), forearm (with cuff),
     hand (with glove), thigh, shin, foot (with shoe), plus skirt, cape and back hair.
   - Labels follow the layouts (neck with head for Standard, hand with forearm for Chibi...).
   - Base: SegFormer-B2 (MIT, already in GGS) or See-through's SAM-HQ, with extra input channels. It trains on
     8–16 GB.
2. **Hidden-part painter.** klein zero-shot, as now (`ggs.rig.refine`). If that is not good enough, a klein LoRA
   trained on pairs (a part with its hidden area masked -> the whole part), using GGS's style trainer.

## Training data (the real work)
- **A. 3D toon renders, exact labels.**
  - Source: CC0 rigged characters (Quaternius, Kenney, KayKit: 50+ characters with clothes and props), posed
    with our mocap clips in side view.
  - Rendered headless in Blender with a cel shader and outlines.
  - Each vertex goes to the bone with its largest skinning weight, mapped to our parts. A cuff skinned to the
    forearm is labelled forearm by construction.
  - Every part is also rendered alone, which gives the full (hidden) masks and complete part images.
  - Variety: random colours, outline widths, proportions and poses.
  - About 1–2 s per image, so 10k images in a few hours.
- **B. Close the style gap.** Low-poly renders do not look like GGS drawings.
  - Redraw a share of the renders with klein ("as a 2D cartoon illustration") at low strength, so the shapes stay.
  - Keep a redraw only if its silhouette still matches the labels.
- **C. Our own characters, checked by a person.**
  - GGS draws characters over a known pose guide.
  - The current slicer gives first labels; the owner corrects a few hundred in a simple label editor.
  - These are the fine-tuning and test sets that matter most.
- Not used: the See-through and Live2D data (not released or commercial); anime-part-split (non-commercial).

## Cost (8 GB laptop / 16 GB desktop)
| Step | Time |
|---|---|
| Blender generator + 10k renders | 1–2 days to build, hours to run |
| klein restyle of 2k renders | ~15–20 h laptop, ~6 h desktop |
| Segmenter fine-tune | 2–6 h |
| Optional hidden-part LoRA | ~2 h at 512 px (laptop) |
| Label editor + 200–500 owner corrections | 1 day to build, owner's time |

## Licences
- CC0 assets: training use is fine (no redistribution of the assets themselves).
- Blender: GPL, a build-time tool, never shipped.
- Every dataset gets a registry entry and a LICENSE_REVIEW row (hard rule 3).

## Order
1. Blender data generator on CC0 characters; 5k side-view renders with visible and full part masks.
2. Fine-tune the segmenter with skeleton channels. Measure on 30 hand-checked GGS characters against today's
   slicer.
3. If the score gap is large, restyle (B) and add owner corrections (C).
4. Hidden-part LoRA only if klein zero-shot is not good enough.
