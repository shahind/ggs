"""Texture packing helpers: power-of-two sizes, grid and shelf packing, edge extrusion."""

from __future__ import annotations

MAX_TEXTURE = 8192


class PackError(ValueError):
    pass


def next_pot(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def grid_layout(n: int, w: int, h: int, padding: int, pot: bool = True) -> tuple[int, int, int, int]:
    """Uniform cells of (w + 2*padding) x (h + 2*padding). Returns (cols, rows, sheet_w, sheet_h),
    choosing the column count with the smallest sheet area (then the squarest sheet)."""
    cw, ch = w + 2 * padding, h + 2 * padding
    best = None
    for cols in range(1, n + 1):
        rows = -(-n // cols)
        sw, sh = cols * cw, rows * ch
        if pot:
            sw, sh = next_pot(sw), next_pot(sh)
        key = (sw * sh, abs(sw - sh))
        if best is None or key < best[0]:
            best = (key, cols, rows, sw, sh)
    _, cols, rows, sw, sh = best
    if max(sw, sh) > MAX_TEXTURE:
        raise PackError(f"sheet would be {sw}x{sh}; above {MAX_TEXTURE}. Use a smaller height or fewer frames")
    return cols, rows, sw, sh


def shelf_pack(sizes: list[tuple[int, int]], padding: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Pack rectangles (tallest first) into the smallest power-of-two sheet found. Returns (sheet size,
    top-left of each rectangle's content, in input order)."""
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], -sizes[i][0]))
    widest = max(w for w, _ in sizes) + 2 * padding
    area = sum((w + 2 * padding) * (h + 2 * padding) for w, h in sizes)
    width = next_pot(max(widest, int(area**0.5)))
    while width <= MAX_TEXTURE:
        pos: list[tuple[int, int]] = [(0, 0)] * len(sizes)
        x = y = shelf_h = 0
        for i in order:
            w, h = sizes[i][0] + 2 * padding, sizes[i][1] + 2 * padding
            if x + w > width:
                x, y, shelf_h = 0, y + shelf_h, 0
            pos[i] = (x + padding, y + padding)
            x += w
            shelf_h = max(shelf_h, h)
        height = next_pot(y + shelf_h)
        if height <= width:
            return (width, height), pos
        width *= 2
    raise PackError(f"parts do not fit in a {MAX_TEXTURE} px sheet")


def paste_extruded(sheet, img, xy: tuple[int, int], extrude: int) -> None:
    """Paste ``img`` at ``xy`` and copy its border pixels ``extrude`` px outward (against texture bleeding)."""
    x, y = xy
    w, h = img.size
    sheet.paste(img, (x, y))
    if extrude <= 0:
        return
    for i in range(1, extrude + 1):
        sheet.paste(img.crop((0, 0, w, 1)), (x, y - i))
        sheet.paste(img.crop((0, h - 1, w, h)), (x, y + h - 1 + i))
    col_l = sheet.crop((x, y - extrude, x + 1, y + h + extrude))
    col_r = sheet.crop((x + w - 1, y - extrude, x + w, y + h + extrude))
    for i in range(1, extrude + 1):
        sheet.paste(col_l, (x - i, y - extrude))
        sheet.paste(col_r, (x + w - 1 + i, y - extrude))
