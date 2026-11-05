import numpy as np
import pytest
from bvh_synth import HEIGHT, JOINTS, walk_bvh

from ggs.rig.bvh import BvhError, parse_bvh

CHAIN = """HIERARCHY
ROOT A
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT B
  {
    OFFSET 0 -1 0
    CHANNELS 3 {order}
    End Site
    {
      OFFSET 0 -1 0
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.5
0 0 0 0 0 0 0 0 0
1 2 3 90 0 0 0 0 0
0 0 0 0 0 0 90 90 0
"""


def chain(order: str = "Zrotation Xrotation Yrotation") -> str:
    return CHAIN.replace("{order}", order)


def test_chain_structure():
    m = parse_bvh(chain())
    assert m.joint_names == ["A", "B", "B_End"]
    assert m.n_frames == 3
    assert m.fps == pytest.approx(2.0)
    assert m.joints[2].is_end_site
    assert m.joints[1].channel_start == 6


def test_chain_known_positions():
    p = parse_bvh(chain()).world_positions()
    np.testing.assert_allclose(p[0], [[0, 0, 0], [0, -1, 0], [0, -2, 0]], atol=1e-12)
    # root moved to (1,2,3) and rotated +90 about Z: the -Y chain now points along +X
    np.testing.assert_allclose(p[1], [[1, 2, 3], [2, 2, 3], [3, 2, 3]], atol=1e-12)


def test_rotation_channel_order_matters():
    # frame 2: joint B has Z=90, X=90 in channel order Z,X (R = Rz @ Rx) -> B_End = B + (0, 0, -1)
    p = parse_bvh(chain("Zrotation Xrotation Yrotation")).world_positions()
    np.testing.assert_allclose(p[2, 2] - p[2, 1], [0, 0, -1], atol=1e-12)
    # same numbers, but now column 1 is X and column 2 is Z (R = Rx @ Rz) -> (1, 0, 0)
    p = parse_bvh(chain("Xrotation Zrotation Yrotation")).world_positions()
    np.testing.assert_allclose(p[2, 2] - p[2, 1], [1, 0, 0], atol=1e-12)


def test_rest_pose_ignores_channels():
    rest = parse_bvh(chain()).world_positions(rest=True)
    assert rest.shape == (1, 3, 3)
    np.testing.assert_allclose(rest[0, 2], [0, -2, 0])


def test_wrong_value_count_rejected():
    bad = chain().replace("0 0 0 0 0 0 90 90 0\n", "0 0 0 0 0 0 90 90\n")
    with pytest.raises(BvhError, match="expected 3 frames"):
        parse_bvh(bad)


def test_unknown_channel_rejected():
    with pytest.raises(BvhError, match="unknown channels"):
        parse_bvh(chain("Zrotation Xrotation Wrotation"))


def test_truncated_hierarchy_rejected():
    with pytest.raises(BvhError):
        parse_bvh("HIERARCHY\nROOT A\n{\n OFFSET 0 0")


def test_synthetic_humanoid_parses():
    m = parse_bvh(walk_bvh(frames=10))
    names = m.joint_names
    for j in JOINTS:
        assert j in names
    assert "Head_End" in names
    rest = m.world_positions(rest=True)[0]
    assert rest[:, 1].max() - rest[:, 1].min() == pytest.approx(HEIGHT)
