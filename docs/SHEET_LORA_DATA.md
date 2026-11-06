# Training data for the "prompt -> part sheet" LoRA

What to prepare so GGS can train a klein LoRA that draws a character directly as a rig part sheet.
Templates: `python tools/sheet_template.py --body standard` (also chibi, heroic, stocky, lanky, child) writes
`data/sheet_lora/template_<body>.png`: the grid, each cell's name, its joint point (red circle) and a mannequin
part at the right size.

## Folder (git-ignored, never committed)
```
data/sheet_lora/sheets/
  <id>.png        the part sheet (required)
  <id>.txt        one-line description of the character (required)
  <id>_full.png   the whole character assembled, side view (optional; helps checking and the image->sheet adapter)
  <id>.yaml       optional: body (a preset name or ratios), source, licence
```
`<id>` is any short name without spaces (`explorer_boy_01`).

## The sheet (`humanoid_side_v1` layout)
- **Size:** 1024 x 768 px, a 4 x 3 grid of 256 px cells. 2048 x 1536 (exactly 2x) is fine too.
- **Background:** transparent PNG; plain white also works.
- **Cells (column, row):**

  | Row | Col 0 | Col 1 | Col 2 | Col 3 |
  |---|---|---|---|---|
  | 0 | head | torso | pelvis | upper arm |
  | 1 | forearm | hand | thigh | shin |
  | 2 | foot | head, eyes closed | head, mouth open | head, hurt |

  The three expression cells are optional; leave them empty if you have none.
- **View:** side view, the character faces right. The limb parts are the near side (the character's right arm
  and leg); GGS reuses them, darker, for the far side.
- **Where each part sits:** the joint point goes on the red circle of the template.
  - Arm and leg parts hang straight down from the joint at the top: upper arm from the shoulder, forearm from the
    elbow, hand from the wrist, thigh from the hip, shin from the knee.
  - The head points up from the base of the neck. The torso points up from the waist to the neck. The pelvis is
    centred on its point.
  - The foot lies flat from the ankle, toes to the right.
- **Scale:** every part of one sheet at the same scale, about 540 px per standing height for a standard body
  (as in the template). Leave about 6 px of empty margin inside each cell.
- **Whole parts:**
  - Draw each part complete, including what is normally hidden: the chest behind the arm, the top of the thigh
    under the shirt.
  - Each limb part continues a little past its joint with a rounded end, so the pieces overlap when they bend.
- **What goes with what:**
  - head: hat, hair, ears, neck
  - torso: collar, shirt body, chest straps
  - pelvis: belt, hips, the top of the trousers or skirt
  - upper arm: sleeve
  - forearm: cuff, bracer
  - hand: glove
  - shin: trouser hem, sock
  - foot: shoe or boot
- Skirts, capes, long back hair and held props need the layout cells for those extras; leave them out of this
  first set.

## The description (`<id>.txt`)
- Appearance only, one line. For example: "a boy explorer in a khaki short-sleeved shirt, khaki trousers rolled
  at the ankles, brown lace-up boots and a safari hat".
- No layout words ("sheet", "parts") and no style trigger; GGS adds the LoRA's own trigger word.

## How many
- **First test:** about 150 sheets.
- **Good:** 300–1,000.
- **Variety matters more than count:**
  - body types (standard, chibi, stocky, lanky, child), noting the body in `<id>.yaml`
  - clothes and armour
  - colours
  - drawing styles (outlined cartoon, flat, painterly)

## Rights
Only images you have the rights to train on (hard rule 7). Record the source and licence in `<id>.yaml` for
bought or downloaded art.

## What GGS does with it
1. Checks each sheet: grid size, a part in every required cell, nothing touching a cell edge. It reassembles the
   sheet in a standing pose so you can review it.
2. Pairs each sheet with the layout guide for its body (the reference image) and its description, and trains the
   klein LoRA with `ggs.style.train_edit` extended to one caption per sheet. Settings: rank 32–64, 2,000–3,000
   steps, 512 px on the laptop or 768–1024 px on the desktop.
3. Scores the LoRA on new prompts with the reassembly validator against today's slicer.
