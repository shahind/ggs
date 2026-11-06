"""Template for hand-made part sheets (the prompt -> part sheet LoRA's training data, docs/SHEET_LORA_DATA.md).

Draws the rig's part-sheet grid with every cell's name, its joint point (pivot) and the mannequin part at the size
the factory layout uses, for a body preset. Run:
    python tools/sheet_template.py --body standard --out data/sheet_lora/template_standard.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_template(body: str, out: Path) -> Path:
    from PIL import Image, ImageDraw

    from ggs.paths import rigs_dir
    from ggs.rig import load_rig_spec
    from ggs.rig.body import PRESETS
    from ggs.rig.factory import sheet_guide

    rig = load_rig_spec(rigs_dir() / "humanoid_side_v1.yaml")
    ps = rig.part_sheet
    c = ps.cell_px
    guide = sheet_guide(rig, PRESETS[body]).image.convert("RGB")
    faint = Image.blend(Image.new("RGB", guide.size, "white"), guide, 0.35)
    d = ImageDraw.Draw(faint)
    for i in range(ps.cols + 1):
        d.line([(i * c, 0), (i * c, ps.rows * c)], fill=(170, 170, 190), width=1)
    for j in range(ps.rows + 1):
        d.line([(0, j * c), (ps.cols * c, j * c)], fill=(170, 170, 190), width=1)
    cells = [(p.name, p.cell, p.pivot) for p in ps.parts] + [(v.name + " (head)", v.cell, None)
                                                           for v in ps.head_variants]
    for name, (col, row), pivot in cells:
        d.text((col * c + 6, row * c + 4), name, fill=(60, 60, 90))
        if pivot:
            x, y = col * c + pivot[0] * c, row * c + pivot[1] * c
            d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(220, 40, 40), width=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    faint.save(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--body", default="standard")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sheet_lora" / "template_standard.png")
    a = ap.parse_args()
    print(make_template(a.body, a.out))
