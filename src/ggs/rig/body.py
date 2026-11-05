"""Body shapes: per-character proportions for the pose guide (and so for the master, the cut and the animation).

A body shape scales the rig's rest lengths (head, torso, arms, legs) and the guide's part widths, then rescales
everything so the character is one standing height tall again: proportions change, not the unit. How tall a
character is in the game is ``stature`` in character.yaml (a multiplier on the export height).

The shape is always chosen explicitly (``--body chibi`` or ``--body head=1.6,legs=0.7``); it is never guessed
from the prompt. Its words are added to the master prompt so the generator follows the guide's proportions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ggs.rig.spec import RigSpec


class BodyShape(BaseModel):
    name: str = "custom"
    head: float = Field(1.0, gt=0.3, le=3.0)  # head length and skull size
    torso: float = Field(1.0, gt=0.3, le=2.0)
    arms: float = Field(1.0, gt=0.3, le=2.0)
    legs: float = Field(1.0, gt=0.3, le=2.0)
    width: float = Field(1.0, gt=0.3, le=2.5)  # thickness of every part
    # one segment on its own (None = the arms / legs factor above), so a long forearm on a short upper arm works
    upper_arm: float | None = Field(None, gt=0.3, le=2.0)
    forearm: float | None = Field(None, gt=0.3, le=2.0)
    thigh: float | None = Field(None, gt=0.3, le=2.0)
    shin: float | None = Field(None, gt=0.3, le=2.0)
    words: str = ""  # added to the master prompt

    def factor(self, segment: str) -> float:
        """Length factor of a body segment: head, torso, upper_arm, forearm, hand, thigh or shin."""
        if segment in ("head", "torso"):
            return getattr(self, segment)
        if segment in ("upper_arm", "forearm", "thigh", "shin"):
            own = getattr(self, segment)
            if own is not None:
                return own
        return self.arms if segment in ("upper_arm", "forearm", "hand") else self.legs

    @property
    def arm_mean(self) -> float:
        return (self.factor("upper_arm") + self.factor("forearm")) / 2

    @property
    def leg_mean(self) -> float:
        return (self.factor("thigh") + self.factor("shin")) / 2

    def describe(self) -> str:
        if self.words:
            return self.words
        out = []
        for value, big, small in ((self.head, "a large head", "a small head"),
                                  (self.leg_mean, "long legs", "short legs"),
                                  (self.arm_mean, "long arms", "short arms"), (self.width, "a stocky, broad build",
                                                                           "a slim build")):
            if value >= 1.15:
                out.append(big)
            elif value <= 0.87:
                out.append(small)
        return ", ".join(out)


PRESETS: dict[str, BodyShape] = {
    "standard": BodyShape(name="standard"),
    "heroic": BodyShape(name="heroic", head=0.85, torso=1.05, arms=1.05, legs=1.12, width=1.05,
                        words="heroic proportions: small head, long legs, broad shoulders"),
    "chibi": BodyShape(name="chibi", head=2.1, torso=0.75, arms=0.7, legs=0.55, width=1.25,
                       words="chibi proportions: a very large head, a small body, short arms and short legs"),
    "stocky": BodyShape(name="stocky", head=1.1, torso=0.95, arms=0.9, legs=0.78, width=1.4,
                        words="a short, stocky, broad build with short legs"),
    "lanky": BodyShape(name="lanky", head=0.9, torso=1.05, arms=1.15, legs=1.22, width=0.78,
                       words="a tall, thin, lanky build with long arms and long legs"),
    "child": BodyShape(name="child", head=1.45, torso=0.9, arms=0.9, legs=0.85, width=0.95,
                       words="child proportions: a large head and short limbs"),
}

_SEGMENTS = ("head", "torso", "upper_arm", "forearm", "hand", "thigh", "shin")
RATIOS = ("head", "torso", "arms", "legs", "width", "upper_arm", "forearm", "thigh", "shin")


def parse_body(text: str | None) -> BodyShape:
    """A preset name, or comma-separated ratios like 'head=1.6,legs=0.7,width=1.2' (others stay 1)."""
    if not text:
        return PRESETS["standard"]
    t = text.strip().lower()
    if t in PRESETS:
        return PRESETS[t]
    if "=" not in t:
        raise ValueError(f"unknown body shape {text!r}; presets: {sorted(PRESETS)}, or ratios like head=1.5,legs=0.8")
    values = {}
    for item in t.split(","):
        key, _, val = item.partition("=")
        key = key.strip()
        if key not in RATIOS:
            raise ValueError(f"body ratio {key!r} unknown; use {', '.join(RATIOS)}")
        try:
            values[key] = float(val)
        except ValueError as exc:
            raise ValueError(f"body ratio {item!r} is not a number") from exc
    return BodyShape(**values)


def _base(bone: str) -> str:
    for suffix in ("_near", "_far", "_l", "_r"):
        if bone.endswith(suffix):
            return bone[: -len(suffix)]
    return bone


def _segment(bone: str) -> str | None:
    base = _base(bone)
    return base if base in _SEGMENTS else None


def standing_height(lengths: dict[str, float], head_scale: float = 1.0) -> float:
    """Ground to the top of the skull, standing straight (feet and skull extents as the programmer art draws
    them): leg + pelvis + torso + head, plus the foot's thickness and the skull above the head bone's end."""
    side = "near" if "thigh_near" in lengths else "l"
    return (lengths[f"thigh_{side}"] + lengths[f"shin_{side}"] + lengths["pelvis"] + lengths["torso"]
            + lengths["head"] + 0.026 + 0.026 * head_scale)


def body_lengths(rig: RigSpec, body: BodyShape) -> dict[str, float]:
    """The rig's rest lengths reshaped by ``body`` and rescaled to one standing height."""
    rest = {b.name: b.rest_length for b in rig.bones}
    out = {}
    for name, length in rest.items():
        g = _segment(name)
        out[name] = length * (body.factor(g) if g else 1.0)
        if _base(name) == "foot":
            out[name] = length * (1.0 + body.leg_mean) / 2  # feet follow the legs halfway
    k = standing_height(rest) / standing_height(out, body.head)
    return {n: round(v * k, 5) for n, v in out.items()}


def body_scale(rig: RigSpec, body: BodyShape) -> float:
    """Factor the lengths were rescaled by (widths and the skull scale with it, so the drawing stays in step)."""
    rest = {b.name: b.rest_length for b in rig.bones}
    raw = {n: rest[n] * (body.factor(g) if (g := _segment(n)) else 1.0) for n in rest}
    for n in raw:
        if _base(n) == "foot":
            raw[n] = rest[n] * (1.0 + body.leg_mean) / 2
    return standing_height(rest) / standing_height(raw, body.head)
