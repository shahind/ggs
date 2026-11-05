"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _drawing_cache(tmp_path_factory, monkeypatch):
    """Items move earlier drawings to ~/.ggs/cache/drawings: keep tests out of the real one."""
    from ggs import items

    root = tmp_path_factory.mktemp("drawings")
    monkeypatch.setattr(items, "cache_dir", lambda d: root / items.hashlib.sha1(str(d).encode()).hexdigest()[:16])


@pytest.fixture(autouse=True)
def _no_recycle_bin(monkeypatch):
    """Deleting in tests removes the temp files for good instead of filling the real Recycle Bin."""
    import shutil
    from pathlib import Path

    from ggs import trash

    def gone(path, permanent_ok=False):
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    monkeypatch.setattr(trash, "move_to_trash", gone)
