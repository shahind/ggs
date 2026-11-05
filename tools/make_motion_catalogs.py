"""Build the bundled motion catalogs (motions/catalogs/<source>.zip) from free motion capture.

    python tools/make_motion_catalogs.py cmu      --src data/mocap/src/cmu-mocap.zip --index data/mocap/src/cmu-index.txt
    python tools/make_motion_catalogs.py 100style --src data/mocap/src/100STYLE.zip

Sources (licences in docs/LICENSE_REVIEW.md, credits in motions/CREDITS.md):
  cmu       Carnegie Mellon University Graphics Lab Motion Capture Database, BVH conversion by Bruce Hahne
            (cgspeed, GitHub mirror meshula/cmu-mocap). "Free for use in research and commercial projects."
            Every take, full length, retargeted with the rig's ``cmu_bvh`` profile at 24 fps.
  100style  The 100STYLE Dataset - Ian Mason (CC BY 4.0). One actor, ~100 styles x (walk, run forwards,
            backwards, sideways, idle). Each take is trimmed by Frame_Cuts.csv and cut into a seamless loop.
The raw data stays in data/ (git-ignored); only the converted 2D clips are written.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FPS = 24.0
OUT = ROOT / "motions" / "catalogs"

CMU_CATEGORIES = [  # first match wins
    ("Fighting", r"punch|kick|box|sword|fight|strike|karate|martial|attack|block|combat|wrestl|stab|duel|kung"),
    ("Sports", (r"basketball|soccer|football|golf|baseball|tennis|bowl|frisbee|skat|ski\b|dribbl|hockey|"
               r"volleyball|sport|swim|throw|catch|pitch|bat\b|racquet|yoga|stretch|exercise|jumping jack")),
    ("Dance", r"danc|salsa|ballet|waltz|charleston|lambada|tap\b|hip ?hop|breakdanc|indian|jive|cha"),
    ("Moving", (r"walk|run|jog|jump|hop|climb|crawl|skip|sneak|stair|step|march|leap|roll|cartwheel|flip|"
               r"swing|hang|ladder|balance|turn|navigat|tiptoe|limp|stride|dash")),
    ("Acting", (r"animal|monkey|chicken|zombie|drunk|old|pantomim|express|gestur|wave|sit|stand|wash|drink|eat|"
               r"phone|laugh|cry|salute|bow|clap|point|cheer|sad|happy|angry|scared|tired|story|act|mime|"
               r"pick|lift|carry|push|pull|open|reach|bend|kneel|lie|fall|sweep|mop|wipe|play")),
]


def _round(clip) -> dict:
    from ggs.rig.catalog import compact

    return compact(clip)


def _write(source: str, index: dict, clips: dict[str, dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{source}.zip"
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("index.json", json.dumps(index, indent=1))
        for mid, d in clips.items():
            z.writestr(f"clips/{mid}.json", json.dumps(d, separators=(",", ":")))
    tmp.replace(out)
    raw = sum(len(json.dumps(d, separators=(",", ":"))) for d in clips.values())
    print(f"wrote {out}: {len(clips)} motions, {out.stat().st_size / 2**20:.1f} MB "
          f"({raw / 2**20:.0f} MB of clip JSON)")
    return out


# ---------------------------------------------------------------- CMU


def _cmu_index(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    takes, subjects = {}, {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"Subject #(\d+) \((.*)\)", line.strip())
        if m:
            subjects[f"{int(m.group(1)):02d}"] = m.group(2)
            continue
        m = re.match(r"(\d+_\d+)\s+(.*)", line.strip())
        if m:
            takes[m.group(1)] = m.group(2).strip()
    return takes, subjects


def _category(text: str) -> str:
    t = text.lower()
    for name, pat in CMU_CATEGORIES:
        if re.search(pat, t):
            return name
    return "Other"


def _cmu_one(job: tuple[str, str]) -> tuple[str, dict | None, str]:
    mid, path = job
    from ggs.rig import load_rig_spec
    from ggs.rig.bvh import load_bvh
    from ggs.rig.retarget import retarget

    try:
        spec = load_rig_spec(ROOT / "rigs" / "humanoid_side_v1.yaml")
        clip = retarget(load_bvh(path), spec, "cmu_bvh", fps=FPS)
        return mid, _round(clip), ""
    except Exception as exc:  # noqa: BLE001 - a bad take is skipped and reported
        return mid, None, f"{type(exc).__name__}: {exc}"


def build_cmu(src: Path, index: Path, limit: int | None = None) -> Path:
    takes, subjects = _cmu_index(index)
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(src) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".bvh")]
        jobs = []
        for n in sorted(names):
            mid = Path(n).stem
            if not re.fullmatch(r"\d+_\d+", mid):
                continue
            p = Path(tmp) / f"{mid}.bvh"
            p.write_bytes(z.read(n))
            jobs.append((mid, str(p)))
        jobs = jobs[:limit] if limit else jobs
        print(f"cmu: converting {len(jobs)} takes")
        clips, motions, failed = {}, [], []
        with ProcessPoolExecutor() as pool:
            for i, (mid, d, err) in enumerate(pool.map(_cmu_one, jobs, chunksize=8)):
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(jobs)}", flush=True)
                if d is None:
                    failed.append((mid, err))
                    continue
                desc = takes.get(mid, "Unknown")
                subj = subjects.get(mid.split("_")[0], "")
                if desc.lower() == "unknown" and subj:
                    desc = subj
                text = f"{desc} {subj}"
                clips[mid] = d
                motions.append({"id": mid, "name": desc[:1].upper() + desc[1:], "category": _category(text),
                                "tags": sorted(set(re.findall(r"[a-z]+", text.lower())) - {"and", "the", "of",
                                                                                         "a", "on", "with"}),
                                "seconds": round(d["frame_count"] / FPS, 2), "frames": d["frame_count"],
                                "note": f"subject {mid.split('_')[0]}: {subj}" if subj else ""})
    for mid, err in failed:
        print(f"  skipped {mid}: {err}")
    idx = {"format": "ggs.catalog", "version": 1, "source": "cmu",
           "title": "CMU Graphics Lab Motion Capture Database",
           "credit": "The data used in this project was obtained from mocap.cs.cmu.edu. The database was created "
                     "with funding from NSF EIA-0196217. BVH conversion by Bruce Hahne (cgspeed).",
           "license": "Free for use in research and commercial projects (CMU)", "url": "http://mocap.cs.cmu.edu",
           "rig": "humanoid_side_v1", "motions": motions}
    return _write("cmu", idx, clips)


# ---------------------------------------------------------------- 100STYLE

STYLE_MOVES = {"FW": ("Walk forward", True), "FR": ("Run forward", True), "BW": ("Walk backward", True),
               "BR": ("Run backward", True), "SW": ("Side-step walk", True), "SR": ("Side-step run", True),
               "ID": ("Idle", True)}


def _straightest(motion, start: int, end: int, seconds: float) -> tuple[int, int]:
    """Frame window of ``seconds`` inside start:end where the body faces the most constant direction (the
    actor walks long laps; a straight stretch projects cleanly onto the side view)."""
    import numpy as np

    P = motion.world_positions()[start:end]
    lat = P[:, motion.index("RightHip")] - P[:, motion.index("LeftHip")]
    ang = np.unwrap(np.arctan2(lat[:, 2], lat[:, 0]))
    w = max(2, round(seconds * motion.fps))
    if len(ang) <= w:
        return start, end
    step = max(1, round(motion.fps / 2))
    best, best_i = float("inf"), 0
    for i in range(0, len(ang) - w, step):
        seg = ang[i:i + w]
        spread = float(seg.max() - seg.min())
        if spread < best:
            best, best_i = spread, i
    return start + best_i, start + best_i + w


def _style_one(job: tuple[str, str, str, int, int]) -> tuple[str, dict | None, str]:
    mid, path, move, start, end = job
    from ggs.rig import load_rig_spec
    from ggs.rig.bvh import load_bvh
    from ggs.rig.cliptools import cut_loop, find_loop
    from ggs.rig.retarget import retarget

    try:
        spec = load_rig_spec(ROOT / "rigs" / "humanoid_side_v1.yaml")
        motion = load_bvh(path)
        s0, e0 = _straightest(motion, start, min(end, motion.n_frames), seconds=4.0)
        clip = retarget(motion, spec, "xsens_100style", fps=FPS, start=s0, end=e0,
                        in_place=move in ("ID",), drop_tpose_frame=False)
        lo, hi = (1.0, 3.0) if move == "ID" else (0.6, 1.6)
        hi = min(hi, clip.duration * 0.9)
        s, e, _ = find_loop(clip, min(lo, hi), hi)
        return mid, _round(cut_loop(clip, s, e)), ""
    except Exception as exc:  # noqa: BLE001
        return mid, None, f"{type(exc).__name__}: {exc}"


def _split_style(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


def build_100style(src: Path, limit: int | None = None) -> Path:
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(src) as z:
        names = z.namelist()
        cuts_name = next(n for n in names if n.endswith("Frame_Cuts.csv"))
        cuts = {}
        for row in csv.DictReader(io.StringIO(z.read(cuts_name).decode("utf-8-sig"))):
            cuts[row["STYLE_NAME"]] = row
        jobs = []
        for n in sorted(names):
            m = re.fullmatch(r"(?:.*/)?([A-Za-z]+)_([A-Z]{2})\.bvh", n)
            if not m or m.group(2) not in STYLE_MOVES:
                continue
            style, move = m.groups()
            row = cuts.get(style)
            if row is None:
                continue
            try:
                start, end = int(row[f"{move}_START"]), int(row[f"{move}_STOP"])
            except (KeyError, ValueError):
                continue
            p = Path(tmp) / Path(n).name
            p.write_bytes(z.read(n))
            jobs.append((f"{style}_{move}", str(p), move, start // 1, end))
        jobs = jobs[:limit] if limit else jobs
        print(f"100style: converting {len(jobs)} takes")
        clips, motions, failed = {}, [], []
        with ProcessPoolExecutor() as pool:
            for mid, d, err in pool.map(_style_one, jobs, chunksize=4):
                if d is None:
                    failed.append((mid, err))
                    continue
                style, move = mid.rsplit("_", 1)
                label, loop = STYLE_MOVES[move]
                clips[mid] = d
                motions.append({"id": mid, "name": f"{label} ({_split_style(style).lower()})",
                                "category": "Styled moves", "tags": [style.lower(), *label.lower().split()],
                                "seconds": round(d["frame_count"] / FPS, 2), "frames": d["frame_count"],
                                "loop": loop, "in_place": move == "ID"})
    for mid, err in failed:
        print(f"  skipped {mid}: {err}")
    idx = {"format": "ggs.catalog", "version": 1, "source": "100style", "title": "The 100STYLE Dataset",
           "credit": "The 100STYLE Dataset - Ian Mason (Mason, Starke & Komura, ACM TOG 2022)",
           "license": "CC BY 4.0", "url": "https://www.ianxmason.com/100style", "rig": "humanoid_side_v1",
           "motions": motions}
    return _write("100style", idx, clips)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", choices=["cmu", "100style"])
    ap.add_argument("--src", type=Path, required=True, help="the source zip")
    ap.add_argument("--index", type=Path, help="CMU: cmu-mocap-index-text.txt")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N takes (testing)")
    a = ap.parse_args()
    if a.source == "cmu":
        if not a.index:
            ap.error("cmu needs --index")
        build_cmu(a.src, a.index, a.limit)
    else:
        build_100style(a.src, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
