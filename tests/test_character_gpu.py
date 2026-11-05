"""M5 on the real models (python -m pytest -m gpu): klein draws a master over the pose guide, the cutter turns it
into a part sheet and the validator scores it. Checks the plumbing, VRAM and that the master follows the guide."""

import json

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from ggs.hardware import detect_vram_gb, profile_for
from ggs.paths import repo_root
from ggs.pipeline.models import MissingModelError, resolve_matting, resolve_pipeline
from ggs.pipeline.vram import BUDGET_GB
from ggs.rig.make import cut_master, make_masters
from ggs.rig.spec import load_rig_spec


@pytest.fixture(scope="module")
def profile():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    p = profile_for(detect_vram_gb())
    if p not in BUDGET_GB:
        pytest.skip(f"profile {p} cannot generate")
    try:
        resolve_pipeline("flux2-klein-4b", p)
        resolve_matting("birefnet")
    except MissingModelError as exc:
        pytest.skip(str(exc))
    return p


def test_master_follows_guide_and_cuts(tmp_path, profile):
    rig = load_rig_spec(repo_root() / "rigs" / "humanoid_side_v1.yaml")
    (master,) = make_masters(rig, "a knight in silver armour with a red cape", tmp_path, seeds=[1],
                             model="flux2-klein-4b", profile=profile)
    for s in json.loads((tmp_path / "stages.json").read_text(encoding="utf-8")):
        assert s["peak_gb"] < BUDGET_GB[profile], s
    out = cut_master(master, rig, tmp_path / "character", "knight")
    agreement = [c for c in out.report.checks if c.name.startswith("guide_agreement:")]
    assert agreement and all(c.passed for c in agreement), out.report.failures()
    assert out.report.score > 0.85
