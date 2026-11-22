"""Pack the motion library (motions/catalogs, clips, embeds) into one zip for a GitHub release.

These files are generated (tools/make_motion_catalogs.py from CMU / 100STYLE, tools/make_motion_library.py with
Kimodo), so they are not in git (hard rule 8). Setup downloads the zip instead (``ggs motion fetch``).

    python tools/pack_motions.py --out dist/ggs-motions-v1.zip

Prints the size and SHA-256 to put under ``library:`` in models/downloads.yaml. The zip is deterministic (sorted
names, fixed timestamps), so packing the same files twice gives the same checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ("catalogs", "clips", "embeds")


def pack(out: Path) -> tuple[int, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for part in PARTS for p in (ROOT / "motions" / part).rglob("*") if p.is_file())
    if not files:
        raise SystemExit("motions/ has no catalogs, clips or embeds to pack")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())
    data = out.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "dist" / "ggs-motions-v1.zip"))
    a = ap.parse_args()
    size, sha = pack(Path(a.out))
    print(f"{a.out}\nsize: {size}\nsha256: {sha}")


if __name__ == "__main__":
    main()
