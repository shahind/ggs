"""One-to-All frames: pose drawing, model choice and inputs (no GPU)."""

import numpy as np


def test_pose_drawing_matches_one_to_all_style():
    from ggs.pipeline.one_to_all import COLOURS, draw_pose

    pts = [None] * 18
    pts[1], pts[2], pts[3] = (100, 100), (100, 160), (100, 220)  # neck, right shoulder, right elbow
    img = np.asarray(draw_pose(pts, (200, 300)))
    assert img.shape == (300, 200, 3)
    assert img[:5].max() == 0  # black background
    limb = img[120:150, 98:103].reshape(-1, 3).max(axis=0)
    assert limb[0] > 100 and limb[2] < 40  # neck-shoulder limb in its OpenPose colour (red, dimmed to 60 %)
    joint = img[220, 100]
    assert tuple(joint) == tuple(COLOURS[3])  # joints full colour


def test_frames_method_prefers_one_to_all(monkeypatch):
    from ggs.pipeline import models
    from ggs.rig import sprites

    calls = []

    def fake(name, *a, **k):
        calls.append(name)
        if name != sprites.ONE_TO_ALL:
            raise models.MissingModelError("no")
        return {}

    monkeypatch.setattr(models, "resolve_frames", fake)
    assert sprites.frames_method() == "one_to_all"

    def none(name, *a, **k):
        raise models.MissingModelError("no")

    monkeypatch.setattr(models, "resolve_frames", none)
    assert sprites.frames_method() == "klein"


def test_one_to_all_spec_resolves_its_three_parts(tmp_path):
    from ggs.pipeline.models import load_pipelines, resolve_frames

    (tmp_path / "One-to-All-1.3b_2").mkdir()
    (tmp_path / "One-to-All-1.3b_2" / "model.safetensors").write_bytes(b"x")
    (tmp_path / "Wan2.1-T2V-1.3B-Diffusers" / "vae").mkdir(parents=True)
    te = tmp_path / "umt5-xxl" / "split_files" / "text_encoders"
    te.mkdir(parents=True)
    (te / "umt5_xxl_fp16.safetensors").write_bytes(b"x")
    files = resolve_frames("one-to-all-1.3b", [tmp_path], load_pipelines())
    assert set(files) == {"model", "base", "text_encoder"} and files["text_encoder"].is_file()
