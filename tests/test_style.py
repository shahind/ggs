"""M4 style training on the CPU: dataset import (rights gate), captions, packages, projects, style resolution,
the base-model lock and the CLI. The real training run is in test_style_gpu.py (marked gpu)."""

import zipfile

import numpy as np
import pytest
import yaml

pytest.importorskip("PIL")
from PIL import Image

import ggs.style.package as pkgmod
from ggs.cli import main
from ggs.generate import run_recipe
from ggs.recipes import load_recipe
from ggs.style.caption import strip_style_words, with_trigger
from ggs.style.dataset import RightsNotConfirmed, import_dataset, load_dataset
from ggs.style.package import (
    StyleError,
    StyleMeta,
    init_project,
    install,
    list_styles,
    read_meta,
    resolve_style,
    write_package,
)


@pytest.fixture
def shared(tmp_path, monkeypatch):
    """Isolate ~/.ggs for styles."""
    home = tmp_path / "home_ggs"
    monkeypatch.setattr(pkgmod, "shared_dir", lambda: home)
    return home


def make_images(folder, n=6, rgba_first=True):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        if i == 0 and rgba_first:
            im = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
            im.paste((200, 30, 30, 255), (100, 50, 200, 150))
        else:
            im = Image.new("RGB", (200 + 10 * i, 240), (250, 250, 250))
            im.paste((20 * i, 100, 200), (40, 40, 120, 160))
        im.save(folder / f"img{i}.png")
    (folder / "img1.txt").write_text("a blue square", encoding="utf-8")
    return folder


def meta(name="ink", trigger="GGSINK", base="flux2-klein-4b"):
    return StyleMeta(name=name, trigger=trigger, base_model=base, rank=4, alpha=4, steps=10, resolution=512,
                     learning_rate=1e-4, images=6)


def fake_lora(path):
    path.write_bytes(b"not-really-safetensors")
    return path


# ------------------------------------------------------------------ dataset


def test_import_requires_rights(tmp_path):
    src = make_images(tmp_path / "src")
    with pytest.raises(RightsNotConfirmed, match="rights"):
        import_dataset(src, tmp_path / "ds", rights_confirmed=False)
    assert not (tmp_path / "ds").exists()


def test_import_normalises_and_keeps_captions(tmp_path):
    src = make_images(tmp_path / "src")
    (src / "dup.png").write_bytes((src / "img2.png").read_bytes())  # exact duplicate is skipped
    m = import_dataset(src, tmp_path / "ds", rights_confirmed=True, size=256)
    assert len(m.images) == 6 and m.rights_confirmed and m.size == 256
    manifest, items = load_dataset(tmp_path / "ds")
    for p, _ in items:
        im = Image.open(p)
        assert im.mode == "RGB" and im.size == (256, 256)
    by_src = {manifest.images[i].source: (p, c) for i, (p, c) in enumerate(items)}
    first = np.asarray(Image.open(by_src["img0.png"][0]))
    assert first[0, 0].tolist() == [255, 255, 255]  # transparency flattened onto white
    assert by_src["img1.png"][1] == "a blue square" and by_src["img3.png"][1] is None
    assert "img2.png" not in by_src or "dup.png" not in by_src  # one copy of the duplicate


def test_import_rejects_too_few_and_bad_size(tmp_path):
    src = make_images(tmp_path / "src", n=3)
    with pytest.raises(ValueError, match="at least"):
        import_dataset(src, tmp_path / "ds", rights_confirmed=True)
    src = make_images(tmp_path / "src2")
    with pytest.raises(ValueError, match="multiple of 16"):
        import_dataset(src, tmp_path / "ds2", rights_confirmed=True, size=500)


def test_load_dataset_errors(tmp_path):
    with pytest.raises(ValueError, match="not a style dataset"):
        load_dataset(tmp_path)


# ------------------------------------------------------------------ captions


def test_caption_helpers():
    assert with_trigger("GGSINK", "a fox sitting.") == "GGSINK, a fox sitting."
    assert with_trigger("GGSINK", "GGSINK, a fox") == "GGSINK, a fox"
    cap = "A pink rabbit holds a carrot. The illustration uses soft watercolor washes. The background is white."
    assert strip_style_words(cap) == "A pink rabbit holds a carrot. The background is white."
    assert strip_style_words("A watercolor painting.") == "A watercolor painting."  # never empty


# ------------------------------------------------------------------ packages + projects


def test_package_round_trip_and_install(tmp_path, shared):
    lora = fake_lora(tmp_path / "w.safetensors")
    out = write_package(tmp_path / "ink", meta(), lora, Image.new("RGB", (8, 8)), ["GGSINK, a fox"])
    assert out.suffix == ".ggsstyle"
    with zipfile.ZipFile(out) as z:
        assert set(z.namelist()) == {"style.yaml", "lora.safetensors", "preview.png", "captions.txt"}
    assert read_meta(out).trigger == "GGSINK"
    dest = install(out, tmp_path / "styles")
    assert (dest / "lora.safetensors").read_bytes() == lora.read_bytes()
    with pytest.raises(StyleError, match="already installed"):
        install(out, tmp_path / "styles")
    install(out, tmp_path / "styles", overwrite=True)
    bad = tmp_path / "bad.ggsstyle"
    bad.write_bytes(b"zip? no")
    with pytest.raises(StyleError, match="not a GGS style"):
        read_meta(bad)


def test_style_name_rules():
    with pytest.raises(ValueError):
        meta(name="Ink Wash!")


def test_projects_and_resolution_order(tmp_path, shared):
    lora = fake_lora(tmp_path / "w.safetensors")
    pkg_a = write_package(tmp_path / "a", meta(trigger="PROJ"), lora)
    pkg_b = write_package(tmp_path / "b", meta(trigger="SHARED"), lora)
    proj = tmp_path / "game"
    init_project(proj, "My game")
    with pytest.raises(StyleError, match="already"):
        init_project(proj)
    with pytest.raises(StyleError, match="not a GGS project"):
        list_styles(tmp_path / "nope")
    install(pkg_a, proj / ".ggs" / "styles")
    install(pkg_b, shared / "styles")
    assert resolve_style("ink", proj)[0].trigger == "PROJ"  # project copy wins
    assert resolve_style("ink", None)[0].trigger == "SHARED"
    m, path = resolve_style(str(pkg_a), None)  # a .ggsstyle path works without installing
    assert m.trigger == "PROJ" and path.exists()
    with pytest.raises(StyleError, match="not found"):
        resolve_style("missing", proj)
    assert [s.name for s, _ in list_styles(proj)] == ["ink", "ink"]


def test_style_base_model_lock(tmp_path, shared):
    lora = fake_lora(tmp_path / "w.safetensors")
    other = write_package(tmp_path / "x", meta(base="z-image-turbo"), lora)
    with pytest.raises(StyleError, match="trained on 'z-image-turbo'"):
        run_recipe(load_recipe("sprite"), "a fox", tmp_path / "o.png", model="flux2-klein-4b", seeds=[0],
                   style_ref=str(other))


def test_style_trigger_reaches_prompt(tmp_path, shared):
    lora = fake_lora(tmp_path / "w.safetensors")
    pkg = write_package(tmp_path / "ink", meta(), lora)
    _, info = run_recipe(load_recipe("sprite"), "a fox", tmp_path / "fox.png", model="stub", seeds=[0],
                         size=(256, 256), style_ref=str(pkg), style="soft colours")
    assert info["prompt"].startswith("GGSINK, soft colours, a fox")
    assert info["style_package"] == {"name": "ink", "trigger": "GGSINK", "weight": 1.0}


# ------------------------------------------------------------------ CLI


def test_cli_dataset_and_project_flow(tmp_path, shared, capsys):
    src = make_images(tmp_path / "src")
    ds = tmp_path / "ds"
    assert main(["style", "dataset", "import", str(src), "--out", str(ds)]) == 3
    assert "--i-have-the-rights" in capsys.readouterr().err
    assert main(["style", "dataset", "import", str(src), "--out", str(ds), "--i-have-the-rights",
                 "--size", "256"]) == 0
    assert main(["style", "train", str(ds), "--name", "ink", "--out", str(tmp_path / "ink.ggsstyle"),
                 "--profile", "lite"]) == 1
    assert "trigger" in capsys.readouterr().err  # not captioned yet
    assert main(["style", "train", str(ds), "--name", "ink", "--out", str(tmp_path / "i.ggsstyle"),
                 "--profile", "lite", "--resolution", "1024"]) == 1
    assert "lite" in capsys.readouterr().err

    proj = tmp_path / "game"
    assert main(["project", "init", str(proj)]) == 0
    pkg = write_package(tmp_path / "ink", meta(), fake_lora(tmp_path / "w.safetensors"))
    assert main(["style", "install", str(pkg), "--project", str(proj)]) == 0
    assert main(["style", "list", "--project", str(proj)]) == 0
    assert "GGSINK" in capsys.readouterr().out
    assert main(["generate", "--class", "sprite", "--prompt", "a fox", "--model", "stub", "--size", "256x256",
                 "--style", "ink", "--project", str(proj), "--out", str(tmp_path / "fox.png")]) == 0
    side = yaml.safe_load((tmp_path / "fox.json").read_text(encoding="utf-8"))
    assert side["style_package"]["name"] == "ink"
