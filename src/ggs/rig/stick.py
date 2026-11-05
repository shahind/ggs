"""Stick-figure pose references: what the drawing model sees instead of the puppet.

Given the puppet itself, the model copied its shapes (a round head, rounded-box hands and feet, even the shirt
colour). Given only lines for the bones and dots for the joints, it draws the character's own shapes in the
pose. The head is a line from the neck to the top of the skull with a short tick for the face (a circle would
come back as a round head).
"""

from __future__ import annotations

import math

LINE = (96, 96, 96)
FAR = (175, 175, 175)  # side view: the far arm and leg are lighter and drawn first (behind)


def _nearest_on(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return ax + t * dx, ay + t * dy


def draw_stick(draw, rig, joints: dict, *, scale: float = 1.0, offset=(0.0, 0.0), width: float = 8.0,
               colour=LINE, facing: str = "right") -> None:
    """Draw the figure on a PIL ImageDraw. ``joints``: bone -> ((x0, y0), (x1, y1)) pixels; drawn at
    ``offset + scale * p``. ``facing``: right (side view) or front."""
    w = max(1.0, width)

    def P(p):
        return offset[0] + p[0] * scale, offset[1] + p[1] * scale

    def dot(p, r):
        x, y = P(p)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)

    def dot_c(p, r, c):
        x, y = P(p)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)

    parents = {b.name: b.parent for b in rig.bones}
    bones = sorted(rig.bones, key=lambda b: not b.name.endswith("_far"))  # far limbs first, behind the rest
    for b in bones:
        c = FAR if b.name.endswith("_far") and colour == LINE else colour
        s, e = joints[b.name]
        par = parents[b.name]
        if par is not None:  # shoulders and hips: from the parent's line to where the limb starts
            ps, pe = joints[par]
            q = _nearest_on(s, ps, pe)
            if math.dist(q, s) > 0.5:
                draw.line([P(q), P(s)], fill=c, width=round(w))
        if b.name == "head":
            continue
        draw.line([P(s), P(e)], fill=c, width=round(w))
        dot_c(s, w * 0.85, c)
        dot_c(e, w * 0.85, c)
    s, e = joints["head"]
    draw.line([P(s), P(e)], fill=colour, width=round(w))
    dot(e, w * 0.85)
    # the face: a tick across the head at eye height, pointing the way the character looks
    hx, hy = e[0] - s[0], e[1] - s[1]
    length = math.hypot(hx, hy) or 1.0
    ux, uy = hx / length, hy / length
    eye = (s[0] + hx * 0.55, s[1] + hy * 0.55)
    if facing == "front":
        nx, ny = -uy, ux
        a = (eye[0] - nx * length * 0.28, eye[1] - ny * length * 0.28)
        b = (eye[0] + nx * length * 0.28, eye[1] + ny * length * 0.28)
    else:  # side view facing right: the tick points forward (screen right of the head line)
        nx, ny = -uy, ux
        if nx < 0:
            nx, ny = -nx, -ny
        a, b = eye, (eye[0] + nx * length * 0.35, eye[1] + ny * length * 0.35)
    draw.line([P(a), P(b)], fill=colour, width=round(w))
    dot(b, w * 0.85)


def stick_image(rig, joints: dict, size: tuple[int, int], *, width: float | None = None, facing: str = "right",
                background=(255, 255, 255)):
    """The stick figure on a plain background, same canvas and joint pixels as the guide (RGB)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, background)
    draw_stick(ImageDraw.Draw(img), rig, joints, width=width or max(3.0, size[1] / 128), facing=facing)
    return img


# OpenPose (COCO-18) keypoints: what SDXL's pose adapter was trained on. 0 nose, 1 neck, 2-4 right arm,
# 5-7 left arm, 8-10 right leg, 11-13 left leg, 14/15 eyes, 16/17 ears (the character's own right / left).
_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
          (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
_COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
            (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
            (170, 0, 255), (255, 0, 255), (255, 0, 85), (255, 0, 85)]


def openpose_points(joints: dict, view: str) -> list[tuple[float, float] | None]:
    """The 18 OpenPose keypoints from rig joints. Side view: the near side is the character's right."""
    r, left = ("near", "far") if view == "side" else ("l", "r")  # front: _l is the picture's left = their right
    s, e = joints["head"]
    hx, hy = e[0] - s[0], e[1] - s[1]
    hl = math.hypot(hx, hy) or 1.0
    ux, uy = hx / hl, hy / hl
    nx, ny = -uy, ux  # across the head
    if view == "side" and nx < 0:
        nx, ny = -nx, -ny

    def at(t, side):
        return s[0] + hx * t + nx * hl * side, s[1] + hy * t + ny * hl * side

    pts: list = [None] * 18
    pts[1] = s
    if view == "side":
        pts[0] = at(0.4, 0.42)
        pts[14] = at(0.55, 0.3)
        pts[16] = at(0.5, -0.05)
    else:
        pts[0] = at(0.4, 0.0)
        pts[14], pts[15] = at(0.55, 0.16), at(0.55, -0.16)
        pts[16], pts[17] = at(0.5, 0.36), at(0.5, -0.36)
    for side, (sh, el, wr, hip, kn, an) in ((r, (2, 3, 4, 8, 9, 10)), (left, (5, 6, 7, 11, 12, 13))):
        pts[sh] = joints[f"upper_arm_{side}"][0]
        pts[el] = joints[f"upper_arm_{side}"][1]
        pts[wr] = joints[f"forearm_{side}"][1]
        pts[hip] = joints[f"thigh_{side}"][0]
        pts[kn] = joints[f"thigh_{side}"][1]
        pts[an] = joints[f"shin_{side}"][1]
    return pts


def openpose_image(joints: dict, size: tuple[int, int], view: str):
    """An OpenPose-style skeleton (coloured limbs and keypoints on black), as the SDXL pose adapter expects."""
    from PIL import Image, ImageDraw

    pts = openpose_points(joints, view)
    w = max(4.0, size[1] / 128)
    img = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, (a, b) in enumerate(_LIMBS):
        if pts[a] is None or pts[b] is None:
            continue
        (x0, y0), (x1, y1) = pts[a], pts[b]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        length = math.hypot(x1 - x0, y1 - y0) / 2
        ang = math.atan2(y1 - y0, x1 - x0)
        ca, sa = math.cos(ang), math.sin(ang)
        poly = [(mx + ca * length * math.cos(t) - sa * w * math.sin(t),  # an ellipse along the limb
                 my + sa * length * math.cos(t) + ca * w * math.sin(t)) for t in (k * math.pi / 18 for k in range(36))]
        d.polygon(poly, fill=tuple(int(c * 0.6) for c in _COLOURS[i]))
    for i, p in enumerate(pts):
        if p is not None:
            d.ellipse([p[0] - w, p[1] - w, p[0] + w, p[1] + w], fill=_COLOURS[i])
    return img
