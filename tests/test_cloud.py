"""Cloud base models (Google Nano Banana) without the network: a fake transport stands in for the Gemini API."""

import base64
import io
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw

import ggs.pipeline.models as pm
from ggs.pipeline import cloud
from ggs.pipeline.vram import Stages


def png_b64(img) -> str:
    b = io.BytesIO()
    img.save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode("ascii")


def drawing(size=(1024, 1024), colour=(200, 40, 40)):
    """What the model "draws": an outlined disc on white, like a sprite."""
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    cx, cy, r = size[0] // 2, size[1] // 2, min(size) // 4
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 20, 24))
    d.ellipse([cx - r + 24, cy - r + 24, cx + r - 24, cy + r - 24], fill=colour)
    return img


def answer(img=None, thought=None, text=None) -> dict:
    parts = []
    if thought is not None:
        parts.append({"thought": True, "inlineData": {"mimeType": "image/png", "data": png_b64(thought)}})
    if text:
        parts.append({"text": text})
    if img is not None:
        parts.append({"inlineData": {"mimeType": "image/png", "data": png_b64(img)}})
    return {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]}


class Fake:
    """Records requests; answers from a list of (status, json) or a function of the body."""

    def __init__(self, replies=None, fn=None):
        self.replies = list(replies or [])
        self.fn = fn
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append(SimpleNamespace(url=url, headers=headers, body=body))
        if self.fn is not None:
            return self.fn(body)
        return self.replies.pop(0)


def spec(name="nano-banana-2.1"):
    return pm.load_pipelines().cloud[name]


def gen(transport, name="nano-banana-2.1"):
    return cloud.GeminiGenerator(spec(name), key="test-key-0123456789", transport=transport, sleep=lambda s: None)


# ------------------------------------------------------------------ catalogue


def test_shipped_cloud_models_have_registry_entries():
    from ggs.models import load_registry

    specs = pm.load_pipelines().cloud
    assert {"nano-banana-2.1", "nano-banana-2", "nano-banana-2-lite", "nano-banana-pro", "nano-banana"} <= set(specs)
    reg = load_registry()
    for s in specs.values():
        e = reg[s.registry]
        assert e.kind == "cloud"
        assert not e.bundle_eligible  # licence not confirmed: never default (hard rule 3)
    assert specs["nano-banana-pro"].api_model == "gemini-3-pro-image"
    assert cloud.is_cloud("nano-banana-pro") and not cloud.is_cloud("flux2-klein-4b") and not cloud.is_cloud(None)


def test_sizes_and_aspects():
    s = spec("nano-banana-2.1")
    assert cloud.nearest_aspect(1024, 1024, s.aspects) == "1:1"
    assert cloud.nearest_aspect(1920, 1080, s.aspects) == "16:9"
    assert cloud.nearest_aspect(512, 2048, s.aspects) == "1:4"
    assert cloud.nearest_aspect(512, 2048, spec("nano-banana-pro").aspects) == "9:16"  # no 1:4 on Pro
    assert cloud.image_size(1024, 1024, s.sizes) == "1K"
    assert cloud.image_size(1536, 768, s.sizes) == "2K"
    assert cloud.image_size(8000, 8000, s.sizes) == "4K"
    assert cloud.image_size(512, 512, spec("nano-banana-2").sizes) == "512"
    assert cloud.image_size(1024, 1024, spec("nano-banana").sizes) is None
    out = cloud.fit(Image.new("RGB", (1376, 768)), 1024, 512)
    assert out.size == (1024, 512)


def test_request_body():
    g = gen(Fake())
    refs = [Image.new("RGB", (3000, 1500), "white")] * 20
    body = g.body("a knight", 7, refs, 1024, 1024)
    parts = body["contents"][0]["parts"]
    assert len(parts) == g.spec.max_refs + 1 and parts[-1] == {"text": "a knight"}
    ref = Image.open(io.BytesIO(base64.b64decode(parts[0]["inline_data"]["data"])))
    assert max(ref.size) == cloud.REF_MAX_PX
    cfg = body["generationConfig"]
    assert cfg["responseModalities"] == ["IMAGE"] and cfg["seed"] == 7
    assert cfg["imageConfig"] == {"aspectRatio": "1:1", "imageSize": "1K"}
    legacy = gen(Fake(), "nano-banana").body("x", 1, None, 1024, 1024)
    assert "imageSize" not in legacy["generationConfig"]["imageConfig"]


# ------------------------------------------------------------------ answers


def test_parse_skips_thought_images_and_explains_refusals():
    final = drawing(colour=(10, 200, 10))
    img = cloud.parse_image(answer(final, thought=drawing(colour=(200, 10, 10))))
    assert img.getpixel((512, 512)) == (10, 200, 10)
    with pytest.raises(cloud.CloudError, match="blocked"):
        cloud.parse_image({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(cloud.CloudError, match="I cannot draw that"):
        cloud.parse_image(answer(text="I cannot draw that"))
    with pytest.raises(cloud.CloudError, match="IMAGE_SAFETY"):
        cloud.parse_image({"candidates": [{"content": {"parts": []}, "finishReason": "IMAGE_SAFETY"}]})


def test_render_fits_size_sends_key_and_runs_jobs_in_parallel():
    fake = Fake(fn=lambda body: (200, answer(drawing((1376, 768)))))
    g = gen(fake)
    out = g.render([("a", 1), ("b", 2, [Image.new("RGB", (64, 64))]), ("c", 3)], 1024, 512, Stages(enforce=False))
    assert [im.size for im in out] == [(1024, 512)] * 3 and g.calls == 3
    assert all(c.headers == {"x-goog-api-key": "test-key-0123456789"} for c in fake.calls)
    assert all("gemini-nano-banana-2.1:generateContent" in c.url for c in fake.calls)
    assert all("test-key" not in c.url for c in fake.calls)  # never in the URL
    assert sorted(c.body["contents"][0]["parts"][-1]["text"] for c in fake.calls) == ["a", "b", "c"]
    assert cloud.cost(g, 1024, 512) == {"model": "gemini-nano-banana-2.1", "size": "1K", "images": 3,
                                        "usd_estimate": round(3 * 0.0336, 4)}


def test_retries_rate_limits_drops_a_rejected_seed_and_reports_bad_keys():
    ok = (200, answer(drawing()))
    fake = Fake([(429, {"error": {"message": "slow down"}}), (503, {}), ok])
    assert gen(fake).render([("a", 1)], 256, 256, Stages(enforce=False))[0].size == (256, 256)
    assert len(fake.calls) == 3

    fake = Fake([(400, {"error": {"message": "Unknown name \"seed\""}}), ok])
    g = gen(fake)
    g.render([("a", 1)], 256, 256, Stages(enforce=False))
    assert "seed" in fake.calls[0].body["generationConfig"] and "seed" not in fake.calls[1].body["generationConfig"]

    fake = Fake([(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})])
    with pytest.raises(cloud.CloudError, match="refused the API key"):
        gen(fake).render([("a", 1)], 256, 256, Stages(enforce=False))
    with pytest.raises(cloud.CloudError, match="cannot load a trained style"):
        gen(Fake()).render([("a", 1)], 256, 256, Stages(enforce=False), lora="ink.safetensors")


def test_check_key_reads_model_info_without_drawing():
    fake = Fake([(200, {"displayName": "Nano Banana 2.1"})])
    assert cloud.check_key(spec(), "k" * 20, fake) == "Nano Banana 2.1"
    assert fake.calls[0].body is None and fake.calls[0].url.endswith("/models/gemini-nano-banana-2.1")


# ------------------------------------------------------------------ the key


def test_key_store_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "key_file", lambda: tmp_path / "google_api_key")
    for env in cloud.KEY_ENV:
        monkeypatch.delenv(env, raising=False)
    assert cloud.key_source() is None
    with pytest.raises(cloud.CloudError, match="no Google API key"):
        cloud.api_key()
    with pytest.raises(cloud.CloudError):
        cloud.save_key("has a space")
    cloud.save_key("  AIzaTestKey1234567890  ")
    raw = (tmp_path / "google_api_key").read_text(encoding="utf-8")
    assert "AIzaTestKey" not in raw if sys.platform == "win32" else raw.startswith("plain:")
    assert cloud.api_key() == "AIzaTestKey1234567890" and cloud.key_source() == "saved"
    monkeypatch.setenv("GEMINI_API_KEY", "from-env-key")
    assert cloud.api_key() == "from-env-key" and cloud.key_source() == "GEMINI_API_KEY"
    assert cloud.mask("AIzaTestKey1234567890") == "AIza...7890"
    assert cloud.clear_key() and not cloud.clear_key()


# ------------------------------------------------------------------ the whole chain


@pytest.fixture
def fake_cloud(tmp_path, monkeypatch):
    """Cloud models answer with a drawn disc; matting is the CPU stub."""
    from ggs.pipeline import matting
    from ggs.pipeline.stub import StubMatter

    fake = Fake(fn=lambda body: (200, answer(drawing())))
    monkeypatch.setattr(cloud, "make_cloud_generator",
                        lambda s: cloud.GeminiGenerator(s, key="k" * 20, transport=fake, sleep=lambda x: None))
    monkeypatch.setattr(cloud, "api_key", lambda: "k" * 20)
    monkeypatch.setattr(pm, "resolve_matting", lambda name, spec_file=None: tmp_path)
    monkeypatch.setattr(matting, "BiRefNetMatter", lambda path: StubMatter())
    return fake


def test_recipe_runs_on_a_cloud_model(tmp_path, fake_cloud):
    from ggs.generate import run_recipe
    from ggs.recipes import load_recipe

    results, meta = run_recipe(load_recipe("sprite"), "a red gem", tmp_path / "gem.png", model="nano-banana-pro",
                               seeds=[1, 2], size=(512, 512), enforce_vram=False)
    assert all(r.passed for r in results) and (tmp_path / "gem_01.png").exists()
    assert Image.open(tmp_path / "gem_01.png").mode == "RGBA"
    assert meta["model"] == "nano-banana-pro" and meta["cloud"]["images"] == 2
    assert "white background" in fake_cloud.calls[0].body["contents"][0]["parts"][-1]["text"]


def test_cli_generate_on_a_cloud_model_needs_no_gpu(tmp_path, fake_cloud, monkeypatch, capsys):
    from ggs import cli

    monkeypatch.setattr(cli, "detect_vram_gb", lambda: None)  # no NVIDIA GPU at all
    code = cli.main(["generate", "--class", "sprite", "--prompt", "a lantern", "--out", str(tmp_path / "l.png"),
                     "--model", "nano-banana-2.1", "--size", "512x512", "--no-vram-check"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "cloud: 1 image(s) from gemini-nano-banana-2.1 at 1K" in out


def test_cli_cloud_key_commands(tmp_path, monkeypatch, capsys):
    from ggs import cli

    monkeypatch.setattr(cloud, "key_file", lambda: tmp_path / "google_api_key")
    for env in cloud.KEY_ENV:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("MY_KEY", "AIzaFromEnv1234567890")
    assert cli.main(["cloud", "key", "set", "--from-env", "MY_KEY"]) == 0
    assert cloud.saved_key() == "AIzaFromEnv1234567890"
    assert cli.main(["cloud", "models"]) == 0
    assert "nano-banana-pro" in capsys.readouterr().out
    assert cli.main(["cloud", "key", "check", "--model", "flux2-klein-4b"]) == 1
    assert cli.main(["cloud", "key", "clear"]) == 0 and cloud.saved_key() is None


def test_project_can_use_a_cloud_base_model(tmp_path):
    from ggs.cli import main
    from ggs.generate import choose_model
    from ggs.recipes import load_recipe

    assert main(["project", "init", str(tmp_path / "p"), "--model", "nano-banana-2"]) == 0
    assert choose_model(load_recipe("sprite"), None, tmp_path / "p") == "nano-banana-2"


def test_a_key_replaces_the_drawing_pack_in_setup():
    pytest.importorskip("PySide6")
    from ggs.ui.setup import needed

    pack = SimpleNamespace(required=True)
    assert needed("drawing", pack, cloud=False) and not needed("drawing", pack, cloud=True)
    assert needed("cutout", pack, cloud=True)  # matting still runs on this computer
    assert not needed("sdxl", SimpleNamespace(required=False), cloud=False)


def test_character_design_sends_the_pose_guide(tmp_path, fake_cloud):
    from ggs.rig.make import draw_character, make_guide
    from ggs.rig.poses import rig_for

    rig = rig_for("side")
    out = draw_character(make_guide(rig, 512), "a knight", view="side", seeds=[3], model="nano-banana-pro",
                         enforce_vram=False, rig=rig)
    assert out[0][0].shape[:2] == (512, 512)
    parts = fake_cloud.calls[0].body["contents"][0]["parts"]
    assert "inline_data" in parts[0] and "stick figure" in parts[-1]["text"]
