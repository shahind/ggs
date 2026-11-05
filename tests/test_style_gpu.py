"""M4 on the real models (python -m pytest -m gpu): a few-step training run -> .ggsstyle -> install into a
second project -> generate with it. Checks the plumbing and the VRAM budget, not the quality of the style."""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("peft")

from PIL import Image

from ggs.generate import run_recipe
from ggs.hardware import detect_vram_gb, profile_for
from ggs.pipeline.models import MissingModelError, resolve_matting, resolve_pipeline
from ggs.pipeline.vram import BUDGET_GB, Stages
from ggs.recipes import load_recipe
from ggs.style.package import StyleMeta, init_project, install, write_package
from ggs.style.train import TrainSettings, train_lora


@pytest.fixture(scope="module")
def profile():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    p = profile_for(detect_vram_gb())
    if p not in BUDGET_GB:
        pytest.skip(f"profile {p} cannot train")
    try:
        resolve_pipeline("flux2-klein-4b", p)
        resolve_matting("birefnet")
    except MissingModelError as exc:
        pytest.skip(str(exc))
    return p


def test_train_package_install_generate(tmp_path, profile):
    files = resolve_pipeline("flux2-klein-4b", profile)
    imgs, caps = [], []
    for i in range(4):
        im = Image.new("RGB", (512, 512), "white")
        im.paste((40 * i, 120, 200 - 30 * i), (128, 128, 384, 384))
        imgs.append(im)
        caps.append(f"GGSTESTSTYLE, a coloured square number {i} on a plain white background")
    stages = Stages(profile=profile)
    settings = TrainSettings(steps=6, rank=4, resolution=512, checkpoint_every=100,
                             preview_prompts=["a square"])
    res = train_lora(imgs, caps, files, settings, tmp_path / "work", stages, "GGSTESTSTYLE")
    assert res.lora.exists() and len(res.losses) == 6 and len(res.previews) == 1
    for s in stages.summary():
        assert s["peak_gb"] < BUDGET_GB[profile] and s["left_gb"] < 0.3, s

    meta = StyleMeta(name="teststyle", trigger="GGSTESTSTYLE", base_model="flux2-klein-4b", rank=4, alpha=4,
                     steps=6, resolution=512, learning_rate=1e-4, images=4)
    pkg = write_package(tmp_path / "teststyle", meta, res.lora, res.previews[0])
    proj = tmp_path / "second_project"
    init_project(proj)
    install(pkg, proj / ".ggs" / "styles")
    results, info = run_recipe(load_recipe("sprite"), "a red apple", tmp_path / "apple.png", seeds=[0],
                               profile=profile, style_ref="teststyle", project=proj, keep_failed=True)
    assert info["style_package"]["name"] == "teststyle"
    assert results[0].path is not None
    for s in info["stages"]:
        assert s["peak_gb"] < BUDGET_GB[profile], s
