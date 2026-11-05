"""The user's saved poses and body shapes: per project (``<project>/.ggs/library``) or for every project
(``~/.ggs/library``). Built-in presets (``ggs.rig.poses``, ``ggs.rig.body``) are always listed first."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from ggs.paths import shared_dir


@dataclass
class Saved:
    name: str
    data: dict  # pose: {"view": ..., "angles": {...}}; body: BodyShape fields
    scope: str  # "builtin" | "project" | "global"


def _dirs(project: Path | None) -> list[tuple[str, Path]]:
    out = []
    if project is not None:
        out.append(("project", Path(project) / ".ggs" / "library"))
    out.append(("global", shared_dir() / "library"))
    return out


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def _check_name(name: str) -> str:
    name = name.strip()
    if not name or len(name) > 40 or not re.match(r"^[\w][\w \-'.]*$", name):
        raise ValueError("a name needs 1-40 letters, digits, spaces or - ' .")
    return name


# ---------------------------------------------------------------- poses

def list_poses(view: str, project: Path | None = None) -> list[Saved]:
    from ggs.rig.poses import LABELS, POSES

    out = [Saved(LABELS.get(pid, pid), {"view": view, "angles": dict(a), "id": pid}, "builtin")
           for pid, a in POSES[view].items()]
    for scope, d in _dirs(project):
        for name, item in sorted(_read(d / "poses.yaml").items()):
            if item.get("view") == view:
                out.append(Saved(name, item, scope))
    return out


def save_pose(name: str, view: str, angles: dict, project: Path | None = None, scope: str = "project") -> Saved:
    name = _check_name(name)
    d = dict(_dirs(project if scope == "project" else None))[scope]
    data = _read(d / "poses.yaml")
    data[name] = {"view": view, "angles": {k: round(float(v), 2) for k, v in angles.items()}}
    _write(d / "poses.yaml", data)
    return Saved(name, data[name], scope)


def delete_pose(name: str, project: Path | None = None, scope: str = "project") -> None:
    d = dict(_dirs(project if scope == "project" else None))[scope]
    data = _read(d / "poses.yaml")
    data.pop(name, None)
    _write(d / "poses.yaml", data)


# ---------------------------------------------------------------- body shapes

def list_bodies(project: Path | None = None) -> list[Saved]:
    from ggs.rig.body import PRESETS

    out = [Saved(b.name.capitalize(), b.model_dump(), "builtin") for b in PRESETS.values()]
    for scope, d in _dirs(project):
        for name, item in sorted(_read(d / "bodies.yaml").items()):
            out.append(Saved(name, item, scope))
    return out


def save_body(name: str, body: dict, project: Path | None = None, scope: str = "project") -> Saved:
    from ggs.rig.body import BodyShape

    name = _check_name(name)
    shape = BodyShape(**{**body, "name": name}).model_dump()  # validates the ratios
    d = dict(_dirs(project if scope == "project" else None))[scope]
    data = _read(d / "bodies.yaml")
    data[name] = shape
    _write(d / "bodies.yaml", data)
    return Saved(name, shape, scope)


def delete_body(name: str, project: Path | None = None, scope: str = "project") -> None:
    d = dict(_dirs(project if scope == "project" else None))[scope]
    data = _read(d / "bodies.yaml")
    data.pop(name, None)
    _write(d / "bodies.yaml", data)
