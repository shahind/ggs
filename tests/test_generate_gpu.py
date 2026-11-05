"""M3 acceptance on the real models (skipped by default; run with: python -m pytest -m gpu).
Outputs go to outputs/test/ (git-ignored)."""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from ggs.generate import run_recipe
from ggs.hardware import detect_vram_gb, profile_for
from ggs.paths import repo_root
from ggs.pipeline.models import MissingModelError, resolve_matting, resolve_pipeline
from ggs.pipeline.vram import BUDGET_GB
from ggs.recipes import load_recipe

OUT = repo_root() / "outputs" / "test"


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


@pytest.mark.parametrize("subject", ["a red health potion in a round glass bottle", "a golden coin with a star"])
def test_sprite_end_to_end(profile, subject):
    OUT.mkdir(parents=True, exist_ok=True)
    name = subject.split(" ")[2]
    results, meta = run_recipe(load_recipe("sprite"), subject, OUT / f"gpu_{name}.png", seeds=[0],
                               profile=profile, style="flat cartoon game art, bold outlines")
    r = results[0]
    assert r.passed, r.failures
    assert r.qa["checks"]["halo"]["value"] <= 0.10
    for s in meta["stages"]:  # one model at a time, inside the profile budget, unloaded afterwards
        assert s["peak_gb"] < BUDGET_GB[profile], s
        assert s["left_gb"] < 0.3, s
