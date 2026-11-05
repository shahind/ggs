"""Command builders for the desktop app (no Qt here, so they are unit-tested).

The app never calls the engine directly: every action is a ``python -m ggs ...`` command line, run as a
subprocess one at a time. The app therefore behaves exactly like the CLI (hard rule 1), and each job frees the
GPU when it ends (hard rule 2).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


def _python() -> str:
    """The console python next to the running one: the app runs under pythonw.exe (no console), but its jobs
    need a python whose output can be read (Qt starts them without a console window)."""
    exe = Path(sys.executable)
    if exe.stem.lower() == "pythonw" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


GGS = [_python(), "-m", "ggs"]


def slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (s[:limit].rstrip("_") or "asset")


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------- project layout used by the app

def assets_dir(project: Path, kind: str) -> Path:
    return Path(project) / "assets" / kind


def characters_dir(project: Path) -> Path:
    return Path(project) / "characters"


def clips_dir(project: Path) -> Path:
    return Path(project) / "clips"


def exports_dir(project: Path) -> Path:
    return Path(project) / "exports"


def new_asset_path(project: Path, kind: str, prompt: str) -> Path:
    return assets_dir(project, kind) / f"{slug(prompt)}_{stamp()}.png"


# ---------------------------------------------------------------- commands

def _opt(args: list[str], flag: str, value) -> None:
    if value not in (None, "", False):
        args += [flag] if value is True else [flag, str(value)]


def project_init(path: Path, name: str | None = None, model: str | None = None) -> list[str]:
    a = [*GGS, "project", "init", str(path)]
    _opt(a, "--name", name)
    _opt(a, "--model", model)
    return a


def generate(asset_class: str, prompt: str, out: Path, *, project: Path | None = None, style: str | None = None,
             style_weight: float | None = None, count: int = 1, seed: int = 0, model: str | None = None,
             size: str | None = None, keep_failed: bool = True) -> list[str]:
    a = [*GGS, "generate", "--class", asset_class, "--prompt", prompt, "--out", str(out), "--count", str(count),
         "--seed", str(seed)]
    _opt(a, "--project", project)
    _opt(a, "--style", style)
    _opt(a, "--style-weight", style_weight)
    _opt(a, "--model", model)
    _opt(a, "--size", size)
    _opt(a, "--keep-failed", keep_failed)
    return a


def character_master(prompt: str, out: Path, name: str, *, project: Path | None = None, body: str | None = None,
                     stature: float = 1.0, pose: str = "walk", count: int = 2, seed: int = 0,
                     style: str | None = None, refine: bool = True, model: str | None = None) -> list[str]:
    a = [*GGS, "character", "master", "--prompt", prompt, "--out", str(out), "--name", name, "--count", str(count),
         "--seed", str(seed), "--stature", str(stature), "--pose", pose]
    _opt(a, "--project", project)
    _opt(a, "--body", body)
    _opt(a, "--style", style)
    _opt(a, "--model", model)
    _opt(a, "--no-refine", not refine)
    return a


def retarget(bvh: Path, out: Path, profile: str = "cmu_bvh", fps: int = 24) -> list[str]:
    return [*GGS, "mocap", "retarget", str(bvh), "--profile", profile, "--fps", str(fps), "--out", str(out)]


def motion_make(prompt: str, out: Path, *, seconds: float = 3.0, seed: int = 0, count: int = 1,
                loop: bool = False, in_place: bool = False) -> list[str]:
    a = [*GGS, "motion", "make", "--prompt", prompt, "--seconds", f"{seconds:g}", "--seed", str(seed),
         "--count", str(count), "--out", str(out)]
    if loop:
        a.append("--loop")
    if in_place:
        a.append("--in-place")
    return a


def motion_add(refs: list[str], project: Path) -> list[str]:
    return [*GGS, "motion", "add", *refs, "--project", str(project)]


def clip_loop(clip: Path, out: Path) -> list[str]:
    return [*GGS, "clip", "loop", str(clip), "--out", str(out)]


def clip_keys(keys: Path, out: Path, character: Path | None = None) -> list[str]:
    a = [*GGS, "clip", "keys", str(keys), "--out", str(out)]
    _opt(a, "--character", character)
    return a


def clip_layer(base: Path, over: Path, out: Path, bones: str = "upper") -> list[str]:
    return [*GGS, "clip", "layer", str(base), "--over", str(over), "--bones", bones, "--out", str(out)]


def item_make(image: Path, out: Path, name: str, length: float, hold: str = "one_hand") -> list[str]:
    return [*GGS, "item", "make", str(image), "--out", str(out), "--name", name, "--length", str(length),
            "--hold", hold]


def export(clip: Path, character: Path, fmt: str, out: Path, *, height: int = 256, items: list[Path] = (),
           name: str | None = None, matte: str | None = None) -> list[str]:
    a = [*GGS, "export", str(clip), "--character", str(character), "--format", fmt, "--out", str(out),
         "--height", str(height)]
    for i in items:
        a += ["--item", str(i)]
    _opt(a, "--name", name)
    _opt(a, "--matte", matte)
    return a


def dataset_import(src: Path, out: Path, rights: bool) -> list[str]:
    a = [*GGS, "style", "dataset", "import", str(src), "--out", str(out)]
    _opt(a, "--i-have-the-rights", rights)
    return a


def caption(dataset: Path, trigger: str) -> list[str]:
    return [*GGS, "style", "caption", str(dataset), "--trigger", trigger]


def train(dataset: Path, name: str, out: Path, *, project: Path | None = None, model: str | None = None,
          steps: int | None = None) -> list[str]:
    a = [*GGS, "style", "train", str(dataset), "--name", name, "--out", str(out)]
    _opt(a, "--project", project)
    _opt(a, "--model", model)
    _opt(a, "--steps", steps)
    return a


def style_install(package: Path, project: Path | None) -> list[str]:
    a = [*GGS, "style", "install", str(package)]
    _opt(a, "--project", project)
    return a


def asset_new(project: Path, asset_class: str, name: str, describe: str = "", size: str = "medium",
              style: str | None = None) -> list[str]:
    a = [*GGS, "asset", "new", name, "--class", asset_class, "--project", str(project), "--describe", describe,
         "--size", size]
    _opt(a, "--style", style)
    return a


def asset_gen(project: Path, item: str, action: str, text: str = "", *, count: int = 3, seed: int | None = None,
              model: str | None = None, version: int | None = None) -> list[str]:
    a = [*GGS, "asset", "gen", item, "--project", str(project), "--action", action, "--count", str(count)]
    _opt(a, "--text", text)
    _opt(a, "--seed", seed)
    _opt(a, "--model", model)
    _opt(a, "--version", version)
    return a


def asset_accept(project: Path, item: str) -> list[str]:
    return [*GGS, "asset", "accept", item, "--project", str(project)]


def character_slice(project: Path, item: str, template: str = "side_simple", layout: str = "auto",
                    extras: list[str] | None = None, recut: bool = False) -> list[str]:
    a = [*GGS, "character", "slice", item, "--project", str(project), "--template", template, "--layout", layout]
    if extras is not None:
        a += ["--extras", ",".join(extras) or "none"]
    if recut:
        a.append("--recut")
    return a


def export_lead_in(cmd: list[str], seconds: float) -> list[str]:
    return [*cmd, "--lead-in", str(seconds)] if seconds else cmd


def character_warp(project: Path, item: str, detail: str = "normal") -> list[str]:
    return [*GGS, "character", "warp", item, "--project", str(project), "--detail", detail]


def character_frames(project: Path, item: str, clip: Path, frames: int, name: str | None = None) -> list[str]:
    a = [*GGS, "character", "frames", item, "--project", str(project), "--clip", str(clip), "--frames", str(frames)]
    _opt(a, "--name", name)
    return a


def export_many(clips: list[Path], character: Path, fmt: str, out: Path, *, height: int = 256,
                matte: str | None = None) -> list[str]:
    """``ggs export`` with several clips (one engine package, or one file per clip in a folder) or none."""
    a = [*GGS, "export", *[str(c) for c in clips], "--character", str(character), "--format", fmt, "--out", str(out),
         "--height", str(height)]
    _opt(a, "--matte", matte)
    return a


def asset_export(project: Path, item: str, size: str | None = None, version: int | None = None) -> list[str]:
    a = [*GGS, "asset", "export", item, "--project", str(project)]
    _opt(a, "--size", size)
    _opt(a, "--version", version)
    return a


def doctor() -> list[str]:
    return [*GGS, "doctor"]


def models_check() -> list[str]:
    return [*GGS, "models", "check"]


def models_add_dir(path: Path) -> list[str]:
    return [*GGS, "models", "add-dir", str(path)]


def models_fetch(pack: str) -> list[str]:
    return [*GGS, "models", "fetch", pack]


# ---------------------------------------------------------------- reading results back

@dataclass
class ResultFile:
    path: Path
    passed: bool
    detail: str


def read_generate_results(out: Path) -> list[ResultFile]:
    """The assets a ``generate`` run wrote (from its JSON sidecar), kept failures included."""
    side = Path(out).with_suffix(".json")
    if not side.exists():
        return []
    meta = json.loads(side.read_text(encoding="utf-8"))
    rows = []
    for a in meta.get("assets", []):
        if a.get("file"):
            rows.append(ResultFile(side.parent / a["file"], bool(a["passed"]),
                                   "QA passed" if a["passed"] else "; ".join(a.get("failures", []))))
    return rows


def read_character_results(out: Path) -> list[dict]:
    """Per seed of a ``character master`` run: master, validation picture, verdict, character.yaml."""
    rows = []
    for seed in sorted(Path(out).glob("seed_*")):
        val = seed / "character" / "validation.json"
        info = json.loads(val.read_text(encoding="utf-8")) if val.exists() else {}
        fails = [f"{c['name']}: {c['detail']}" for c in info.get("checks", []) if not c["passed"]]
        rows.append({"seed": seed.name, "master": seed / "master.png", "validation": seed / "character" /
                     "validation.png", "parts": seed / "character" / "parts.png",
                     "character": seed / "character" / "character.yaml", "passed": bool(info.get("passed")),
                     "score": info.get("score"), "failures": fails})
    return rows
