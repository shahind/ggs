"""Frame-by-frame sprites (experimental) and the spread master pose, on CPU."""

import numpy as np
import pytest

pytest.importorskip("PIL")

from ggs.cli import main
from ggs.paths import repo_root
from ggs.rig.make import cut_master, guide_only_master, guide_puppet
from ggs.rig.master import MASTER_POSE, pose_clip
from ggs.rig.spec import load_rig_spec
from ggs.rig.sprites import (
    SpriteSettings,
    check_frames,
    crop_frames,
    describe_master,
    palette_distance,
    pose_sheet,
)


@pytest.fixture(scope="module")
def rig():
    return load_rig_spec(repo_root() / "rigs" / "humanoid_side_v1.yaml")


def _walkish_clip(rig, lengths, n=8):
    from ggs.rig.clip import Clip, RootTrack

    base = pose_clip(rig, MASTER_POSE, lengths)
    angles = {b: [base.angles[b][0] + (10 * np.sin(i / n * 2 * np.pi) if "thigh" in b else 0.0) for i in range(n)]
              for b in base.bones}
    return Clip(rig=rig.id, source="t", profile="pose", fps=8, frame_count=n, in_place=True, loop=True,
                bones=base.bones, bone_lengths=base.bone_lengths, angles=angles,
                local_angles={b: [0.0] * n for b in base.bones},
                root=RootTrack(x=[0.0] * n, y=[-(lengths["thigh_near"] + lengths["shin_near"])] * n))


def test_describe_master():
    p = "Redraw the figure in the image as a dwarf with a red beard, with a stocky build. Keep exactly the same pose"
    assert describe_master(p) == "a dwarf with a red beard, with a stocky build"
    assert describe_master("something else") == "the character"


def test_pose_sheet_and_checks(rig):
    puppet, lengths = guide_puppet(rig)
    s = SpriteSettings(frames=8, cols=4, rows=2, width=512, height=384)
    sheet, masks, idx, anchor = pose_sheet(rig, puppet, _walkish_clip(rig, lengths), s)
    assert sheet.size == (512, 384) and len(masks) == 8 and len(idx) == 8
    rgba = np.asarray(sheet.convert("RGBA")).copy()
    cw, ch = 128, 192
    cells = []
    for k, m in enumerate(masks):
        c, r = k % 4, k // 4
        cell = rgba[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].copy()
        cell[..., 3] = np.where(m, 255, 0)
        cells.append(cell)
    master = cells[0]
    checks = check_frames(cells, masks, master, s)
    assert all(c.passed for c in checks), [c.why for c in checks]
    cells[3] = np.zeros_like(cells[3])  # a missing frame
    assert not check_frames(cells, masks, master, s)[3].passed
    frames, a = crop_frames(cells, anchor)
    assert len({f.size for f in frames}) == 1 and 0 <= a[1] <= frames[0].size[1]
    assert palette_distance(master, master) == pytest.approx(0.0)


def test_spread_pose_cuts_cleanly(rig, tmp_path):
    out = cut_master(guide_only_master(rig, tmp_path / "m", 512, pose="spread"), rig, tmp_path / "c", "s")
    assert out.report.passed, out.report.failures()


def test_cli_sprites_with_stub(tmp_path, rig):
    from ggs.rig.make import guide_only_master

    master = guide_only_master(rig, tmp_path / "m", 512)
    _, lengths = guide_puppet(rig)
    clip = tmp_path / "walk.json"
    _walkish_clip(rig, lengths).save(clip)
    code = main(["character", "sprites", str(master), "--clip", str(clip), "--out", str(tmp_path / "s"),
                 "--model", "stub", "--count", "1", "--no-vram-check"])
    assert code in (0, 2)  # the stub hands back the mannequins, so the colour check may refuse them
    assert (tmp_path / "s" / "seed_000" / "report.json").exists()
    assert list((tmp_path / "s").glob("sprites*.png"))
