import pytest

from ggs.hardware import profile_for
from ggs.models import load_registry
from ggs.models.registry import compatible_loras


def test_registry_loads_and_loras_have_bases():
    reg = load_registry()
    assert "flux2-klein-4b" in reg
    for e in reg.values():
        if e.kind == "lora":
            assert e.base in reg


def test_non_commercial_entries_are_never_bundle_eligible():
    reg = load_registry()
    for e in reg.values():
        if e.commercial in ("no", "verify"):
            assert not e.bundle_eligible
    assert reg["qwen-image-2.1"].commercial == "no"
    assert reg["flux2-klein-9b"].commercial == "no"


def test_compatible_loras():
    reg = load_registry()
    ids = {e.id for e in compatible_loras(reg, "flux2-klein-4b")}
    assert "lora-pixel-art-klein4b" in ids


@pytest.mark.parametrize(
    "vram,expected",
    [(None, "no-gpu"), (4, "minimal"), (7.9, "lite"), (8, "lite"), (11.9, "lite"), (12, "standard"), (16, "standard")],
)
def test_profiles(vram, expected):
    assert profile_for(vram) == expected
