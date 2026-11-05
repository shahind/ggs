"""Make the starter animation library (motions/library.yaml) with NVIDIA Kimodo.

    python tools/make_motion_library.py            # every motion that has no clip yet
    python tools/make_motion_library.py walk run --force

Needs the motion and motion_text packs (``ggs models fetch motion``, ``... motion_text``). Writes
motions/clips/<name>.json + .gif + .motion.json, and copies each prompt's text embedding to motions/embeds so the
app can make new takes of these motions with the motion model alone. BVH files go to data/motions_build
(git-ignored).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from ggs.pipeline import motion as M
    from ggs.pipeline.vram import Stages
    from ggs.rig import load_rig_spec
    from ggs.rig.library import library_clip, load_library, to_clip, write_sidecar
    from ggs.rig.preview import render_preview

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="only these motions (default: all)")
    ap.add_argument("--force", action="store_true", help="remake motions that already have a clip")
    a = ap.parse_args()
    lib = load_library()
    spec = load_rig_spec(ROOT / "rigs" / "humanoid_side_v1.yaml")
    todo = [m for m in lib.motions if (not a.names or m.name in a.names)
            and (a.force or not library_clip(m.name).is_file())]
    if not todo:
        print("nothing to do")
        return 0
    stages = Stages(profile="lite")
    # read every prompt in one text-encoder load, then one Kimodo load per motion
    M.encode_texts([t for m in todo for t in M.split_prompt(m.prompt)], stages)
    embeds = ROOT / "motions" / "embeds"
    embeds.mkdir(parents=True, exist_ok=True)
    build = ROOT / "data" / "motions_build"
    for m in todo:
        for t in M.split_prompt(m.prompt):
            src = M.find_embed(t)
            if src and src.parent != embeds:
                shutil.copy2(src, embeds / src.name)
        take = M.generate(m.prompt, m.seconds, [m.seed], stages)[0]
        bvh = M.save_bvh(take, build / f"{m.name}.bvh")
        clip = to_clip(bvh, spec, loop=m.loop, in_place=m.in_place, loop_seconds=m.loop_seconds)
        out = library_clip(m.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        clip.save(out)
        render_preview(spec, clip, out.with_suffix(".gif"), cell=200)
        write_sidecar(out, {"name": m.label, "prompt": m.prompt, "seconds": m.seconds, "seed": m.seed,
                            "model": M.MODEL, "credit": "Made with NVIDIA Kimodo"})
        warn = f"  ({len(clip.warnings)} warnings: {clip.warnings[0]})" if clip.warnings else ""
        print(f"{m.name:<14} {clip.frame_count:4d} frames {clip.duration:5.2f} s{warn}", flush=True)
    for r in stages.reports:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
