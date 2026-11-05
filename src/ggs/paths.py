"""Locations of repo data files and the shared per-user folder."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Repo root when running from a source checkout (editable install)."""
    return Path(__file__).resolve().parents[2]


def rigs_dir() -> Path:
    return repo_root() / "rigs"


def registry_path() -> Path:
    return repo_root() / "models" / "registry.yaml"


def shared_dir() -> Path:
    """Per-user folder for downloaded models and adapters (not inside any project)."""
    return Path.home() / ".ggs"
