"""M3 generation core on the CPU: model resolution, VRAM stages, post-processing, QA, recipes, the stub chain
and the CLI. Real-model runs are in test_generate_gpu.py (marked gpu)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image

import ggs.pipeline.models as pm
from ggs.cli import main
from ggs.generate import process, run_recipe
from ggs.pipeline.post import (
    clean_alpha,
    decontaminate,
    estimate_background,
    halo_fraction,
    pad_to_pot,
    refine_alpha_known_background,
    to_rgba,
    trim,
)
from ggs.pipeline.stub import StubGenerator
from ggs.pipeline.vram import Stages, VramBudgetError
from ggs.qa.sprite import SpriteQaRules, check_sprite
from ggs.recipes import Recipe, available_recipes, load_recipe

WHITE = np.array([255.0, 255.0, 255.0])


# ------------------------------------------------------------------ helpers


def disc_scene(size=200, r=50, outline=(20, 20, 30), fill=(200, 40, 40), bg=(255, 255, 255), ss=8):
    """Antialiased outlined disc over a flat background, plus its exact coverage alpha."""
    big = size * ss
    yy, xx = np.mgrid[0:big, 0:big] + 0.5
    d = np.hypot(xx - big / 2, yy - big / 2) / ss
    inside = (d <= r).astype(np.float32)
    ring = ((d > r - 6) & (d <= r)).astype(np.float32)
    rgb = np.empty((big, big, 3), np.float32)
    for c in range(3):
        rgb[..., c] = bg[c] * (1 - inside) + inside * (outline[c] * ring + fill[c] * (1 - ring))
    rgb = rgb.reshape(size, ss, size, ss, 3).mean(axis=(1, 3))
    alpha = inside.reshape(size, ss, size, ss).mean(axis=(1, 3))
    return rgb, alpha


# ------------------------------------------------------------------ model resolution


def test_resolution_prefers_shared_dir_and_skips_config_only_folders(tmp_path):
    home, comfy = tmp_path / "home", tmp_path / "comfy"
    (home / "K" / "text_encoder").mkdir(parents=True)
    (home / "K" / "text_encoder" / "config.json").write_text("{}")  # configs only: must be skipped
    (comfy / "text_encoders").mkdir(parents=True)
    (comfy / "text_encoders" / "te.safetensors").write_bytes(b"x")
    got = pm.resolve(["K/text_encoder", "text_encoders/te.safetensors"], [home, comfy], "te")
    assert got == comfy / "text_encoders" / "te.safetensors"
    assert pm.resolve(["K"], [home, comfy], "cfg", need_weights=False) == home / "K"
    (home / "K" / "text_encoder" / "model.safetensors").write_bytes(b"x")
    assert pm.resolve(["K/text_encoder", "text_encoders/te.safetensors"], [home, comfy], "te") == \
        home / "K" / "text_encoder"
    with pytest.raises(pm.MissingModelError, match="add-dir"):
        pm.resolve(["nope.gguf"], [home, comfy], "transformer")


def test_pipeline_spec_resolves_per_profile(tmp_path):
    for f in ("cfg/model_index.json", "te.safetensors", "lite.gguf", "std.gguf", "vae/w.safetensors"):
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    spec = pm.PipelinesFile.model_validate({"pipelines": {"m": {
        "registry": "r", "loader": "flux2_klein", "configs": ["cfg"], "text_encoder": ["te.safetensors"],
        "transformer": {"standard": ["std.gguf"], "lite": ["lite.gguf"]}, "vae": ["vae"]}}})
    assert pm.resolve_pipeline("m", "lite", [tmp_path], spec)["transformer"].name == "lite.gguf"
    assert pm.resolve_pipeline("m", "standard", [tmp_path], spec)["transformer"].name == "std.gguf"
    with pytest.raises(ValueError, match="cannot run"):
        pm.resolve_pipeline("m", "minimal", [tmp_path], spec)


def test_shipped_pipelines_file_is_valid():
    spec = pm.load_pipelines()
    assert "flux2-klein-4b" in spec.pipelines and "birefnet" in spec.matting
    from ggs.models import load_registry

    reg = load_registry()
    for p in spec.pipelines.values():
        assert p.registry in reg
    for m in spec.matting.values():
        assert m.registry in reg


def test_user_config_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "shared_dir", lambda: tmp_path)
    assert pm.load_user_config().model_dirs == []
    pm.save_user_config(pm.UserConfig(model_dirs=[str(tmp_path / "comfy")]))
    dirs = pm.model_dirs()
    assert dirs[0] == tmp_path / "models" and dirs[1] == tmp_path / "comfy"


# ------------------------------------------------------------------ VRAM stages


class FakeCuda:
    def __init__(self, peak, left):
        self.peak, self.left, self.calls = peak, left, 0
        self.cuda = SimpleNamespace(
            empty_cache=lambda: None, reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
            max_memory_allocated=lambda: self.peak * 2**30, memory_allocated=self._alloc)

    def _alloc(self):
        self.calls += 1
        return 0 if self.calls == 1 else self.left * 2**30


def test_stages_record_and_enforce_budget(monkeypatch):
    s = Stages(profile="lite")
    monkeypatch.setattr(Stages, "_cuda", staticmethod(lambda: FakeCuda(5.0, 0.0)))
    with s.stage("denoise"):
        pass
    assert s.reports[0].peak_gb == pytest.approx(5.0) and s.summary()[0]["name"] == "denoise"
    monkeypatch.setattr(Stages, "_cuda", staticmethod(lambda: FakeCuda(7.5, 0.0)))
    with pytest.raises(VramBudgetError, match="budget"), s.stage("too_big"):
        pass
    monkeypatch.setattr(Stages, "_cuda", staticmethod(lambda: FakeCuda(3.0, 1.0)))
    with pytest.raises(VramBudgetError, match="left"), s.stage("leaky"):
        pass
    monkeypatch.setattr(Stages, "_cuda", staticmethod(lambda: FakeCuda(7.5, 1.0)))
    with Stages(profile="lite", enforce=False).stage("lenient"):
        pass


def test_stages_without_gpu(monkeypatch):
    monkeypatch.setattr(Stages, "_cuda", staticmethod(lambda: None))
    s = Stages()
    with s.stage("cpu"):
        pass
    assert s.reports[0].peak_gb is None and s.reports[0].seconds >= 0


# ------------------------------------------------------------------ post-processing


def test_background_estimate():
    rgb, _ = disc_scene()
    bg, spread = estimate_background(rgb.astype(np.uint8))
    assert np.allclose(bg, 255) and spread == 0
    gradient = rgb.copy()
    gradient[...] = np.linspace(80, 255, rgb.shape[1])[None, :, None]  # not a flat background
    assert estimate_background(gradient.astype(np.uint8))[1] > 12


def test_decontaminate_recovers_foreground_exactly():
    fg = np.array([30.0, 60.0, 200.0])
    a = np.linspace(0.1, 1.0, 10)[None, :]
    c = a[..., None] * fg + (1 - a[..., None]) * WHITE
    out = decontaminate(c, a, WHITE)
    assert np.allclose(out, fg, atol=1e-3)


def test_refine_alpha_removes_generous_matte_halo():
    rgb, true_a = disc_scene()
    # a matte that is too generous: grows 2 px with soft values where the pixels are really background
    grown = np.clip(true_a + 0.6 * (np.roll(true_a, 2, 0) + np.roll(true_a, -2, 0) + np.roll(true_a, 2, 1)
                                    + np.roll(true_a, -2, 1)), 0, 1)
    naive = to_rgba(decontaminate(rgb, grown, WHITE), grown)
    assert halo_fraction(naive, WHITE) > 0.3  # visible pale rim
    refined = refine_alpha_known_background(rgb, grown, WHITE)
    fixed = to_rgba(decontaminate(rgb, refined, WHITE), refined)
    assert halo_fraction(fixed, WHITE) < 0.05
    edge = (true_a > 0.05) & (true_a < 0.95)
    assert np.abs(refined[edge] - true_a[edge]).mean() < 0.08  # close to the true coverage


def test_clean_trim_and_pot():
    a = clean_alpha(np.array([[0.01, 0.5, 0.99]]))
    assert a.tolist() == [[0.0, 0.5, 1.0]]
    rgba = np.zeros((50, 60, 4), np.uint8)
    rgba[10:20, 30:45] = 255
    out, box = trim(rgba, padding=3)
    assert box == (30, 10, 45, 20) and out.shape == (16, 21, 4)
    assert out[:3].max() == 0 and out[3:13, 3:18, 3].min() == 255
    assert pad_to_pot(out).shape == (16, 32, 4)
    with pytest.raises(ValueError):
        trim(np.zeros((4, 4, 4), np.uint8))


def test_to_rgba_clears_hidden_colour():
    out = to_rgba(np.full((2, 2, 3), 200.0), np.array([[0.0, 1.0], [0.5, 0.0]]))
    assert out[0, 0].tolist() == [0, 0, 0, 0] and out[0, 1, 3] == 255


# ------------------------------------------------------------------ QA


def qa_for(rgb, alpha, rules=None):
    rgba = to_rgba(decontaminate(rgb, alpha, WHITE), alpha)
    return check_sprite(rgba, WHITE, 0.0, rules or SpriteQaRules())


def test_qa_passes_clean_sprite():
    rgb, a = disc_scene()
    res = qa_for(rgb, a)
    assert res.passed, res.failures


def test_qa_fails_cut_off_and_empty():
    rgb, a = disc_scene(r=110)  # bigger than the frame
    res = qa_for(rgb, a)
    assert not res.passed and any(f.startswith("cut_off") for f in res.failures)
    empty = qa_for(np.full((100, 100, 3), 255.0), np.zeros((100, 100)))
    assert any(f.startswith("coverage") for f in empty.failures)


def test_qa_fails_halo_and_leftover_background():
    rgb, a = disc_scene()
    grown = np.clip(a + 0.6 * (np.roll(a, 2, 0) + np.roll(a, -2, 0) + np.roll(a, 2, 1) + np.roll(a, -2, 1)), 0, 1)
    res = qa_for(rgb, grown)
    assert any(f.startswith("halo") for f in res.failures)
    thick = np.clip(a + np.roll(a, 3, 0) + np.roll(a, -3, 0) + np.roll(a, 3, 1) + np.roll(a, -3, 1), 0, 1)
    res = qa_for(rgb, thick)  # opaque matte reaches 3 px into the white background
    assert any(f.startswith("leftover_background") for f in res.failures)


def test_qa_fails_islands():
    rgb, a = disc_scene()
    a = a.copy()
    rgb = rgb.copy()
    a[5:8, 5:8] = 1.0
    rgb[5:8, 5:8] = 0
    a[190:194, 150:154] = 1.0
    rgb[190:194, 150:154] = 0
    res = qa_for(rgb, a, SpriteQaRules(edge_margin_px=1))
    assert any(f.startswith("islands") for f in res.failures)


# ------------------------------------------------------------------ recipes


def test_sprite_recipe():
    assert "sprite" in available_recipes()
    r = load_recipe("sprite")
    assert r.background_rgb == (255, 255, 255) and r.base_model == "flux2-klein-4b"
    p = r.prompt("a red potion", "pixel art")
    assert p.startswith("pixel art, a red potion,") and "white background" in p and "  " not in p
    assert r.prompt("a red potion").startswith("a red potion,")


def test_recipe_validation():
    base = load_recipe("sprite").model_dump()
    for bad in ({"size": (1000, 1000)}, {"background": "white"}, {"prompt_template": "no slot"}):
        with pytest.raises(ValueError):
            Recipe.model_validate(base | bad)
    with pytest.raises(ValueError, match="unknown asset class"):
        load_recipe("spaceship")


# ------------------------------------------------------------------ chain + CLI (stub backend)


def test_stub_chain_writes_trimmed_rgba(tmp_path):
    recipe = load_recipe("sprite")
    results, _ = run_recipe(recipe, "a gem", tmp_path / "gem.png", model="stub", seeds=[1, 2],
                               size=(256, 256), debug=True)
    assert [r.passed for r in results] == [True, True]
    for i, r in enumerate(results, 1):
        assert r.path == tmp_path / f"gem_{i:02d}.png"
        im = Image.open(r.path)
        assert im.mode == "RGBA" and im.size == r.size
        a = np.asarray(im)[..., 3]
        assert a[:4].max() == 0 and a[-4:].max() == 0  # 4 px transparent padding
        assert (tmp_path / f"gem_{i:02d}_raw.png").exists() and (tmp_path / f"gem_{i:02d}_alpha.png").exists()
    side = json.loads((tmp_path / "gem.json").read_text(encoding="utf-8"))
    assert side["class"] == "sprite" and side["model"] == "stub" and len(side["assets"]) == 2
    assert {s["name"] for s in side["stages"]} == {"denoise", "matte"}
    # same seed -> same pixels
    again, _ = run_recipe(recipe, "a gem", tmp_path / "again.png", model="stub", seeds=[1], size=(256, 256))
    assert np.array_equal(np.asarray(Image.open(again[0].path)), np.asarray(Image.open(results[0].path)))


def test_failed_qa_is_not_exported(tmp_path, monkeypatch):
    recipe = load_recipe("sprite")
    from ggs.pipeline import stub

    monkeypatch.setattr(stub, "StubGenerator", lambda bg: StubGenerator(bg, cut_off=True))
    results, _ = run_recipe(recipe, "huge", tmp_path / "huge.png", model="stub", seeds=[0], size=(256, 256))
    assert not results[0].passed and results[0].path is None and not (tmp_path / "huge.png").exists()
    results, _ = run_recipe(recipe, "huge", tmp_path / "huge.png", model="stub", seeds=[0], size=(256, 256),
                            keep_failed=True)
    assert results[0].path.name == "huge_QA_FAILED.png"


def test_process_matches_known_alpha():
    rgb, a = disc_scene()
    img = Image.fromarray(np.clip(rgb + 0.5, 0, 255).astype(np.uint8))
    mask = Image.fromarray((a * 255).round().astype(np.uint8), "L")
    arr, qa, _ = process(img, mask, load_recipe("sprite"))
    assert qa.passed, qa.failures
    assert arr.shape[0] == arr.shape[1] == 100 + 2 * 4  # disc diameter + padding


def test_cli_generate_stub(tmp_path, capsys):
    out = tmp_path / "a.png"
    assert main(["generate", "--class", "sprite", "--prompt", "a coin", "--model", "stub", "--size", "256x256",
                 "--out", str(out)]) == 0
    assert "1/1 passed QA" in capsys.readouterr().out and out.exists()
    assert main(["generate", "--class", "sprite", "--prompt", "x", "--model", "stub", "--size", "big",
                 "--out", str(out)]) == 1
    with pytest.raises(SystemExit):
        main(["generate", "--class", "spaceship", "--prompt", "x", "--out", str(out)])
    with pytest.raises(SystemExit):  # the class is never optional
        main(["generate", "--prompt", "x", "--out", str(out)])
