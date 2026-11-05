"""M1 acceptance on real CMU clips. Skipped unless BVH files are in data/mocap/cmu/ (never committed).

Put a CMU walk clip there with "walk" in its name (e.g. 02_01_walk.bvh) to run the walk checks.
"""

import numpy as np
import pytest

from ggs.paths import repo_root, rigs_dir
from ggs.rig import load_bvh, load_rig_spec, retarget

CMU = repo_root() / "data" / "mocap" / "cmu"
ALL = sorted(CMU.glob("*.bvh")) if CMU.is_dir() else []
WALKS = [p for p in ALL if "walk" in p.name.lower()]

pytestmark = pytest.mark.mocap_data


@pytest.fixture(scope="module")
def spec():
    return load_rig_spec(rigs_dir() / "humanoid_side_v1.yaml")


@pytest.mark.skipif(not ALL, reason="no CMU BVH files in data/mocap/cmu/")
@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_cmu_clip_retargets(path, spec):
    clip = retarget(load_bvh(path), spec, "cmu_bvh")
    for bone in clip.bones:
        assert np.isfinite(clip.angles[bone]).all(), bone


@pytest.mark.skipif(not WALKS, reason="no CMU walk clip (*walk*.bvh) in data/mocap/cmu/")
@pytest.mark.parametrize("path", WALKS, ids=lambda p: p.name)
def test_cmu_walk_is_plausible(path, spec):
    clip = retarget(load_bvh(path), spec, "cmu_bvh", fps=30)
    assert not any("backwards" in w for w in clip.warnings), clip.warnings
    for side in ("near", "far"):
        thigh = np.array(clip.angles[f"thigh_{side}"])
        assert np.abs(np.diff(thigh)).max() < 25.0, f"thigh_{side} flips"  # 30 fps
        swing = np.percentile(thigh, 95) - np.percentile(thigh, 5)
        assert 15.0 < swing < 100.0, f"thigh_{side} swing {swing:.1f} deg"
        centered = thigh - np.median(thigh)
        assert np.count_nonzero(np.diff(np.sign(centered)) != 0) >= 2, "thigh does not cycle"
    near = np.array(clip.angles["thigh_near"])
    far = np.array(clip.angles["thigh_far"])
    assert np.corrcoef(near, far)[0, 1] < 0, "legs should swing in anti-phase"
    assert abs(clip.root.x[-1] - clip.root.x[0]) > 0.2  # actually travels
