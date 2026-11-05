"""SDXL on the real models (python -m pytest -m gpu): a few-step style training run at 1024 px -> .ggsstyle ->
a project whose base model is SDXL -> generate a sprite with the style. Checks the plumbing and the VRAM budget,
not the quality of the style."""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("peft")

from PIL import Image

from ggs.generate import run_recipe
from ggs.hardware import detect_vram_gb, profile_for
from ggs.pipeline.models import MissingModelError, load_pipelines, resolve_matting, resolve_pipeline
from ggs.pipeline.vram import BUDGET_GB, Stages
from ggs.recipes import load_recipe
from ggs.style.package import StyleMeta, init_project, install, write_package
from ggs.style.train import TrainSettings, train_style


@pytest.fixture(scope="module")
def profile():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    p = profile_for(detect_vram_gb())
    if p not in BUDGET_GB:
        pytest.skip(f"profile {p} cannot train")
    try:
        resolve_pipeline("sdxl", p)
        resolve_matting("birefnet")
    except MissingModelError as exc:
        pytest.skip(str(exc))
    return p


def test_sdxl_train_and_generate_in_project(tmp_path, profile):
    spec = load_pipelines().pipelines["sdxl"]
    files = resolve_pipeline("sdxl", profile)
    imgs, caps = [], []
    for i in range(4):
        im = Image.new("RGB", (1024, 1024), "white")
        im.paste((40 * i, 120, 200 - 30 * i), (256, 256, 768, 768))
        imgs.append(im)
        caps.append(f"GGSTESTSTYLE, a coloured square number {i} on a plain white background")
    stages = Stages(profile=profile)
    settings = TrainSettings(steps=6, rank=4, resolution=1024, checkpoint_every=100, preview_prompts=["a square"])
    res = train_style("sdxl", imgs, caps, files, settings, tmp_path / "work", stages, "GGSTESTSTYLE",
                      profile=profile, defaults=spec.defaults)
    assert res.lora.exists() and len(res.losses) == 6 and len(res.previews) == 1
    for s in stages.summary():
        assert s["peak_gb"] < BUDGET_GB[profile] and s["left_gb"] < 0.3, s

    meta = StyleMeta(name="sdxltest", trigger="GGSTESTSTYLE", base_model="sdxl", rank=4, alpha=4, steps=6,
                     resolution=1024, learning_rate=1e-4, images=4)
    pkg = write_package(tmp_path / "sdxltest", meta, res.lora, res.previews[0])
    proj = tmp_path / "sdxl_project"
    init_project(proj, base_model="sdxl")
    install(pkg, proj / ".ggs" / "styles")
    results, info = run_recipe(load_recipe("sprite"), "a red apple", tmp_path / "apple.png", seeds=[0],
                               profile=profile, style_ref="sdxltest", project=proj, keep_failed=True)
    assert info["model"] == "sdxl" and info["style_package"]["name"] == "sdxltest"
    assert results[0].path is not None
    for s in info["stages"]:
        assert s["peak_gb"] < BUDGET_GB[profile], s
