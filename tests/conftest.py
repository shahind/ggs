"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _drawing_cache(tmp_path_factory, monkeypatch):
    """Items move earlier drawings to ~/.ggs/cache/drawings: keep tests out of the real one."""
    from ggs import items

    root = tmp_path_factory.mktemp("drawings")
    monkeypatch.setattr(items, "cache_dir", lambda d: root / items.hashlib.sha1(str(d).encode()).hexdigest()[:16])
