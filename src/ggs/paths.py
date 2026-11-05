"""Locations of repo data files and the shared per-user folder."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Where recipes/, rigs/, clips/ and models/*.yaml live.

    ``GGS_ROOT`` if set; else the repo root of a source checkout; else the installed app's folder, which holds
    the bundled Python in ``python/`` next to the data folders (see tools/build_app.py).
    """
    env = os.environ.get("GGS_ROOT")
    if env:
        return Path(env)
    src = Path(__file__).resolve().parents[2]
    if (src / "recipes").is_dir():
        return src
    return Path(sys.prefix).resolve().parent


def rigs_dir() -> Path:
    return repo_root() / "rigs"


def registry_path() -> Path:
    return repo_root() / "models" / "registry.yaml"


def shared_dir() -> Path:
    """Per-user folder for downloaded models and adapters (not inside any project)."""
    return Path.home() / ".ggs"
