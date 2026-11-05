"""Base-model selection, SDXL wiring and training speed helpers (CPU only)."""

import pytest

import ggs.pipeline.models as pm
from ggs.cli import main
from ggs.generate import choose_model
from ggs.pipeline.backends import make_generator
from ggs.pipeline.flux2 import text_token_budget
from ggs.recipes import load_recipe
from ggs.style.package import StyleError, init_project, load_project


def _touch(root, *files):
    for f in files:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def _spec(**extra):
    return pm.PipelinesFile.model_validate({"pipelines": {"m": {
        "registry": "r", "loader": "sdxl", "configs": ["cfg"], "text_encoder": ["te"],
        "transformer": {"lite": ["unet"]}, "vae": ["vae"], **extra}}})


def test_shipped_sdxl_pipeline():
    spec = pm.load_pipelines().pipelines
    assert spec["sdxl"].loader == "sdxl" and spec["sdxl"].registry == "sdxl-base-1.0"
    assert spec["sdxl"].training.resolution == 1024
    assert spec["flux2-klein-4b"].training.resolution == 512
    assert "speed_lora" in spec["sdxl"].extras


def test_unknown_loader_is_rejected():
    with pytest.raises(ValueError, match="unknown loader"):
        pm.PipelinesFile.model_validate({"pipelines": {"m": {
            "registry": "r", "loader": "comfy", "configs": [], "text_encoder": [], "transformer": {}, "vae": []}}})


def test_extras_are_optional(tmp_path):
    _touch(tmp_path, "cfg/model_index.json", "te/text_encoder/w.safetensors", "unet/w.safetensors",
           "vae/w.safetensors")
    spec = _spec(extras={"speed_lora": ["fast.safetensors"]})
    files = pm.resolve_pipeline("m", "lite", [tmp_path], spec)
    assert "speed_lora" not in files
    _touch(tmp_path, "fast.safetensors")
    assert pm.resolve_pipeline("m", "lite", [tmp_path], spec)["speed_lora"].name == "fast.safetensors"


def test_sdxl_generator_settings(tmp_path):
    spec = pm.load_pipelines().pipelines["sdxl"]
    plain = make_generator(spec, {"text_encoder": tmp_path})
    assert (plain.steps, plain.guidance) == (30, 6.0) and plain.negative
    fast = make_generator(spec, {"text_encoder": tmp_path, "speed_lora": tmp_path / "l.safetensors"})
    assert (fast.steps, fast.guidance) == (8, 0.0)  # Lightning: few steps, CFG off


def test_project_base_model_choice(tmp_path):
    recipe = load_recipe("sprite")
    init_project(tmp_path / "a", base_model="sdxl")
    init_project(tmp_path / "b")
    assert load_project(tmp_path / "a").base_model == "sdxl"
    assert choose_model(recipe, None, tmp_path / "a") == "sdxl"
    assert choose_model(recipe, "flux2-klein-4b", tmp_path / "a") == "flux2-klein-4b"  # explicit wins
    assert choose_model(recipe, None, tmp_path / "b") == recipe.base_model
    assert choose_model(recipe, None, None) == recipe.base_model
    with pytest.raises(StyleError):
        choose_model(recipe, None, tmp_path / "missing")


def test_cli_project_model_and_training_limits(tmp_path, capsys):
    assert main(["project", "init", str(tmp_path / "p"), "--model", "nope"]) == 1
    assert "unknown base model" in capsys.readouterr().err
    assert main(["project", "init", str(tmp_path / "p"), "--model", "sdxl"]) == 0
    assert main(["style", "train", str(tmp_path / "ds"), "--name", "x", "--out", str(tmp_path / "x.ggsstyle"),
                 "--project", str(tmp_path / "p"), "--profile", "lite", "--resolution", "2048"]) == 1
    assert "sdxl" in capsys.readouterr().err  # the project's base model was used


def test_text_token_budget():
    assert text_token_budget([43, 113]) == 128
    assert text_token_budget([10]) == 64
    assert text_token_budget([600]) == 512
    assert text_token_budget([128]) == 128


def test_fp8_linear_matches_and_skips_lora():
    torch = pytest.importorskip("torch")
    from ggs.pipeline.fp8 import Fp8Linear, linears_to_fp8

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ModuleDict({"lora_A": torch.nn.Linear(32, 4)}))
    x = torch.randn(3, 16)
    ref = model[0](x)
    n = linears_to_fp8(model)
    assert n == 16 * 32
    assert isinstance(model[0], Fp8Linear) and model[0].weight.dtype == torch.float8_e4m3fn
    assert type(model[1]["lora_A"]) is torch.nn.Linear  # adapters keep full precision
    assert torch.allclose(model[0](x), ref, atol=0.1, rtol=0.1)


def test_recipe_prompt_per_base_model():
    r = load_recipe("sprite")
    sdxl = r.prompt("a lantern", "GGSINK", "sdxl")
    assert sdxl.startswith("GGSINK, a lantern") and "game asset" not in sdxl  # SDXL reads it as a sheet
    assert len(sdxl.split()) < 60  # CLIP reads 77 tokens
    assert r.prompt("a lantern", "", "flux2-klein-4b") == r.prompt("a lantern")  # no override: default template
