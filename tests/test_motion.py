"""M10: text to motion (Kimodo), the starter library and the CMU / 100STYLE catalogs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from ggs.pipeline import motion as M
from ggs.rig import load_rig_spec
from ggs.rig.clip import Clip, RootTrack

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / "rigs" / "humanoid_side_v1.yaml"


def _clip(frames: int = 12) -> Clip:
    spec = load_rig_spec(RIG)
    bones = [b.name for b in spec.bones]
    parents = {b.name: b.parent for b in spec.bones}
    ang = {b: [10.0 * i + j for j in range(frames)] for i, b in enumerate(bones)}

    def wrap(x):
        return -((-x + 180.0) % 360.0 - 180.0)

    local = {b: [wrap(v - (ang[parents[b]][k] if parents[b] else 0.0)) for k, v in enumerate(t)]
             for b, t in ang.items()}
    return Clip(rig=spec.id, source="test", profile="soma", fps=24, frame_count=frames, bones=bones,
                bone_lengths={b: 0.1 for b in bones}, angles=ang, local_angles=local,
                root=RootTrack(x=[0.01 * i for i in range(frames)], y=[0.0] * frames),
                swaps={"thigh": [i % 2 == 0 for i in range(frames)]})


def test_prompt_split_and_frames():
    assert M.split_prompt("A person walks forward. The person jumps.") == ["A person walks forward.",
                                                                          "The person jumps."]
    assert M.frames_for(["a.", "b."], 2.0) == [60, 60]
    assert M.frames_for(["a.", "b."], [1.5, 2]) == [45, 60]
    with pytest.raises(M.MotionError):
        M.split_prompt("  . ")
    with pytest.raises(M.MotionError):
        M.frames_for(["a."], 12)
    with pytest.raises(M.MotionError):
        M.frames_for(["a.", "b."], [1, 2, 3])


def test_embed_lookup_prefers_cache_then_shipped(tmp_path):
    a, b = tmp_path / "cache", tmp_path / "shipped"
    b.mkdir()
    (b / M.embed_name("A person waves.")).write_bytes(b"x")
    assert M.find_embed("A person waves.", [a, b]) == b / M.embed_name("A person waves.")
    assert M.find_embed("A person dances.", [a, b]) is None
    assert M.embed_name("x") == M.embed_name("x") != M.embed_name("y")


def test_cached_text_encoder_shapes():
    torch = pytest.importorskip("torch")
    enc = M._CachedText({"A person waves.": torch.ones(4096)}, "cpu")
    feat, lengths = enc(["A person waves.", " "])
    assert tuple(feat.shape) == (2, 1, 4096) and lengths == [1, 1]
    assert float(feat[1].abs().sum()) == 0.0


def test_soma_profile_maps_every_bone():
    spec = load_rig_spec(RIG)
    soma = spec.mocap_profiles["soma"]
    assert set(soma) == {b.name for b in spec.bones}
    assert soma["thigh_near"] == ["RightLeg", "RightShin"]  # near side = the character's right side


def test_starter_library_manifest():
    from ggs.rig.library import load_library

    lib = load_library()
    names = [m.name for m in lib.motions]
    assert {"idle", "walk", "run", "jump", "punch", "death"} <= set(names)
    for m in lib.motions:
        M.frames_for(M.split_prompt(m.prompt), m.seconds)  # every prompt fits Kimodo's limits


def test_catalog_compact_round_trip():
    from ggs.rig.catalog import compact, expand

    c = _clip()
    d = compact(c)
    assert "local_angles" not in d and isinstance(d["swaps"]["thigh"], str)
    back = expand(json.loads(json.dumps(d)))
    for b in c.bones:
        assert max(abs(x - y) for x, y in zip(back.local_angles[b], c.local_angles[b])) <= 0.11
    assert back.swaps == c.swaps


def test_catalog_search_preview_and_add(monkeypatch, tmp_path):
    pytest.importorskip("PIL")
    from ggs.rig import catalog

    cats = tmp_path / "motions" / "catalogs"
    cats.mkdir(parents=True)
    idx = {"source": "demo", "title": "Demo", "credit": "Demo credit", "license": "CC0",
           "motions": [{"id": "01_01", "name": "Walk happily", "category": "Moving", "tags": ["walk"],
                        "seconds": 0.5, "frames": 12},
                       {"id": "02_01", "name": "Punch", "category": "Fighting", "seconds": 0.5, "frames": 12}]}
    with zipfile.ZipFile(cats / "demo.zip", "w") as z:
        z.writestr("index.json", json.dumps(idx))
        for m in idx["motions"]:
            z.writestr(f"clips/{m['id']}.json", json.dumps(catalog.compact(_clip())))
    monkeypatch.setattr(catalog, "catalogs_dir", lambda: cats)
    assert [m.id for _, m in catalog.search("walk")] == ["01_01"]
    assert [m.id for _, m in catalog.search("", category="Fighting")] == ["02_01"]
    assert catalog.read_clip("demo/02_01").frame_count == 12
    out = catalog.add_to(["demo/01_01"], tmp_path / "proj" / "clips")
    assert out[0].name == "walk_happily.json"  # no source names in what the user sees
    assert catalog.add_to(["demo/01_01"], tmp_path / "proj" / "clips")[0] == out[0]  # same motion: replaced
    meta = json.loads(out[0].with_suffix(".motion.json").read_text(encoding="utf-8"))
    assert meta["credit"] == "Demo credit"
    with pytest.raises(KeyError):
        catalog.find("demo/99_99")


def test_bundled_catalogs_are_valid():
    """Whatever catalogs are built must load and point at clips that exist."""
    from ggs.rig import catalog

    for f in catalog.catalog_files():
        idx = catalog.load_index(f)
        with zipfile.ZipFile(f) as z:
            names = set(z.namelist())
        assert idx.credit and idx.license
        assert all(f"clips/{m.id}.json" in names for m in idx.motions)


def test_motion_cli_lists_library(capsys):
    from ggs.cli import main

    assert main(["motion", "library"]) == 0
    assert "walk" in capsys.readouterr().out


@pytest.mark.gpu
def test_motion_make_end_to_end(tmp_path):
    from ggs.cli import main

    out = tmp_path / "wave.json"
    assert main(["motion", "make", "--prompt", "A person waves with the right hand.", "--seconds", "2",
                 "--out", str(out)]) == 0
    clip = json.loads(out.read_text(encoding="utf-8"))
    assert clip["profile"] == "soma" and clip["frame_count"] >= 40
    assert out.with_suffix(".gif").is_file() and out.with_suffix(".bvh").is_file()


def test_export_credits_only_for_catalog_motions(tmp_path):
    from ggs.rig.library import credits_for, write_credits

    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    (tmp_path / "a.motion.json").write_text(json.dumps({"catalog": "100style", "name": "Walk (angry)",
                                                        "credit": "The 100STYLE Dataset - Ian Mason",
                                                        "license": "CC BY 4.0"}), encoding="utf-8")
    (tmp_path / "b.motion.json").write_text(json.dumps({"prompt": "A person waves.", "model": "kimodo-soma"}),
                                            encoding="utf-8")
    lines = credits_for([a, b, c])
    assert len(lines) == 1 and "Ian Mason" in lines[0] and "Walk (angry)" in lines[0]
    assert write_credits([b, c], tmp_path / "none.txt") is None
    out = write_credits([a], tmp_path / "CREDITS.txt")
    assert "CC BY 4.0" in out.read_text(encoding="utf-8")
