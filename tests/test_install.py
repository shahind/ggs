"""Installer support: model packs manifest, the resumable downloader, and finding data in the installed app."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")

from ggs.pipeline import download as D


def test_manifest_loads_and_required_packs_cover_drawing_and_cutout():
    packs = D.load_packs()
    required = {n for n, p in packs.items() if p.required}
    assert required == {"drawing", "cutout"}
    assert packs["drawing"].ready_when == {"pipelines": ["flux2-klein-4b"]}
    for p in packs.values():
        assert p.parts and p.size_gb > 0


def test_manifest_dirs_match_pipelines_yaml():
    """Every folder a pack downloads into must be one pipelines.yaml looks in."""
    text = (Path(__file__).resolve().parents[1] / "models" / "pipelines.yaml").read_text(encoding="utf-8")
    for pack in D.load_packs().values():
        for part in pack.parts:
            assert part.dir in text, part.dir


def test_pack_not_ready_without_files(monkeypatch, tmp_path):
    import ggs.pipeline.models as M

    monkeypatch.setattr(M, "model_dirs", lambda cfg=None: [tmp_path])
    assert not any(D.pack_ready(p) for p in D.load_packs().values())


class _Resp:
    def __init__(self, data: bytes, status: int, fail_after: int | None = None):
        self.data, self.status_code, self.fail_after = data, status, fail_after

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code != 416:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk):
        sent = 0
        for i in range(0, len(self.data), 4):
            if self.fail_after is not None and sent >= self.fail_after:
                raise requests.ConnectionError("dropped")
            yield self.data[i:i + 4]
            sent += 4


def test_fetch_file_resumes_after_a_dropped_connection(monkeypatch, tmp_path):
    blob = bytes(range(40))
    calls = []

    def fake_get(url, headers=None, stream=True, timeout=None):
        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers else 0
        calls.append(start)
        if len(calls) == 1:
            return _Resp(blob, 200, fail_after=16)
        return _Resp(blob[start:], 206)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    got = []
    dest = tmp_path / "a" / "w.bin"
    D.fetch_file("http://x/w.bin", dest, len(blob), on_bytes=got.append)
    assert dest.read_bytes() == blob
    assert calls == [0, 16]
    assert sum(got) == len(blob)
    assert not dest.with_name("w.bin.part").exists()
    D.fetch_file("http://x/w.bin", dest, len(blob))  # complete file: no request
    assert len(calls) == 2


def test_repo_root_env_and_installed_layout(monkeypatch, tmp_path):
    import ggs.paths as P

    monkeypatch.setenv("GGS_ROOT", str(tmp_path))
    assert P.repo_root() == tmp_path
    monkeypatch.delenv("GGS_ROOT")
    assert (P.repo_root() / "recipes").is_dir()  # source checkout


def test_jobs_use_console_python_next_to_pythonw(monkeypatch, tmp_path):
    from ggs.ui import commands as C

    (tmp_path / "python.exe").write_bytes(b"")
    (tmp_path / "pythonw.exe").write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonw.exe"))
    assert C._python() == str(tmp_path / "python.exe")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    assert C._python() == str(tmp_path / "python.exe")


def test_bundled_loras_are_found_in_the_app_models_folder(monkeypatch, tmp_path):
    from ggs.pipeline.models import load_pipelines, model_dirs, resolve

    files = D.bundled_files()
    assert "SDXL-Lightning/sdxl_lightning_8step_lora.safetensors" in files
    # the full installer bundles it; the small installers download it with the SDXL pack
    assert any(p.repo == "ByteDance/SDXL-Lightning" for p in D.load_packs()["sdxl"].parts)
    monkeypatch.setenv("GGS_ROOT", str(tmp_path))
    assert model_dirs()[-1] == tmp_path / "models"
    lora = tmp_path / "models" / files[0]
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"x")
    candidates = load_pipelines(Path(__file__).resolve().parents[1] / "models" / "pipelines.yaml"
                                ).pipelines["sdxl"].extras["speed_lora"]
    assert resolve(candidates, [tmp_path / "models"], "speed lora") == lora
