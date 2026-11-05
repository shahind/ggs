---
name: add-rig-spec
description: Create a new rig spec (e.g. quadruped side-view, front-view humanoid, top-down character) or change the bone/part set of an existing one. Use when a creature, mount or view needs animation.
---

# Add or change a rig spec

1. Start from the motion side: list the animations needed and which joints they move. Collapse mocap joints to the
   minimum bone set; drop anything the view cannot show (fingers, toes, twist joints).
2. Copy `rigs/humanoid_side_v1.yaml` as the template. Keep the conventions (screen space, +y down, angles clockwise
   from +y, facing right, near side = right) unless the new view requires otherwise; if so, document it at the top of the file.
3. Define: bones (+ parent, rest_length), `draw_order` (back to front, every bone once), `part_sheet` grid with one
   cell per unique art part (reuse far-side art through `bone_far` where symmetric), `sockets`, and `mocap_profiles`
   (leave a profile `{}` if not mapped yet; non-empty profiles must cover every bone).
4. Run `python -m ggs rig validate rigs/<file>.yaml`. Fix errors in the YAML, not by loosening the validator.
5. Add a test in `tests/test_rig_spec.py` style: valid file passes; at least two realistic mistakes are rejected.
6. For animals/vehicles with no mocap source, add procedural or keyframed clips instead of a mocap profile.
7. Document the rig in `docs/` (short) and mention it in `docs/ROADMAP.md` if it changes a milestone.
