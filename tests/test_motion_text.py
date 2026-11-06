"""Motion prompts are tidied exactly as Kimodo tidies them before it asks for their text embeddings."""

import pytest


@pytest.mark.parametrize("prompt", ["walking on a large ball trying to keep balance",
                                    "Walking on a large ball, trying to keep balance!", "A person waves. then jumps",
                                    "  -go  ", "someone go forward", "jumping)"])
def test_prompt_sentences_match_kimodos_sanitiser(prompt):
    kimodo = pytest.importorskip("kimodo.sanitize")
    from ggs.pipeline.motion import split_prompt

    for t in split_prompt(prompt):
        assert kimodo.sanitize_text(t) == t


def test_cached_text_matches_loosely():
    import torch

    from ggs.pipeline.motion import MotionError, _CachedText

    enc = _CachedText({"walking on a large ball.": torch.ones(4096)}, "cpu")
    feat, _ = enc(["Walking on a large ball."])
    assert feat.shape == (1, 1, 4096)
    with pytest.raises(MotionError):
        enc(["Swimming."])
