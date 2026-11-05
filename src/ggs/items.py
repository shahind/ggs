"""Project items (v2): named assets the user creates, draws and exports later.

    <project>/items/<class>/<name>/item.json          what it is, its settings and its drawing history
    <project>/items/<class>/<name>/versions/vNNN.png  the drawing on screen (+ vNNN_raw.png, sidecars)
    <project>/items/<class>/<name>/design/            a character's accepted design (``accept``)

The history keeps every drawing's *recipe* (description, change, seed, pose and body, model, style), not its
image: only the drawing on screen is kept in the project. Earlier drawings go to a small cache outside the project
(``~/.ggs/cache/drawings``); going back to one restores it from there or, when the cache no longer has it, draws
it again from its recipe (a change is redrawn on top of its redrawn parent).

Actions (the app's buttons; also ``ggs asset gen``):
  new         a new drawing from the text (it becomes the description), a new seed
  refine      a change ("make it blue"): klein edits the drawing on screen (same seed); models without image
              editing draw description + change again with the same seed
  regenerate  the description again with a new seed
  variations  several new seeds at once (CLI only)
  goto        show history entry ``version`` again (restored from the cache, or redrawn)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SIZES = {  # the app's size tiles -> longest side of the exported image (None = as generated)
    "small": 128, "medium": 256, "large": 512, "full": None,
}
CACHE_PER_ITEM = 40  # drawings kept in the cache per item (the oldest go first)


class ChatTurn(BaseModel):  # kept so v1 item files still load; the app no longer writes a chat
    role: Literal["user", "app"]
    text: str
    versions: list[int] = Field(default_factory=list)
    time: str = Field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))


class CharacterSettings(BaseModel):
    """How a character is drawn: view, body shape and pose (``ggs.rig.poses``), and the output kind."""
    view: Literal["side", "front"] = "front"
    body_preset: str = "standard"  # ggs.rig.body.PRESETS name, used when ``body`` is empty
    body: dict = Field(default_factory=dict)  # BodyShape fields (the editor's sliders)
    pose: str | dict = "a_pose"  # preset name or bone angles
    mode: Literal["single"] = "single"


class Version(BaseModel):
    """One drawing in the history: everything needed to draw it again."""
    n: int
    file: str  # where its image lives while it is on screen (relative to the item folder); "" = nothing drawn
    prompt: str  # what the model was asked: the description (new / regenerate) or the change (refine)
    action: Literal["new", "refine", "regenerate", "variations"]
    seed: int
    parent: int | None = None  # the drawing a change edited
    passed: bool = True
    failures: list[str] = Field(default_factory=list)
    size: tuple[int, int] | None = None
    description: str = ""  # the description it was drawn from (for a change: its parent's)
    character: CharacterSettings | None = None  # pose and body it was drawn in (characters)
    model: str | None = None
    style: str | None = None
    style_weight: float | None = None


class ItemSettings(BaseModel):
    size: str = "medium"  # export size tile (SIZES)
    style: str | None = None
    style_weight: float | None = None
    model: str | None = None  # None = the project's base model
    gen_size: tuple[int, int] | None = None  # Advanced: generation resolution
    character: CharacterSettings | None = None  # characters only


class ItemRecord(BaseModel):
    format: Literal["ggs.item_record"] = "ggs.item_record"
    version: Literal[1] = 1
    name: str
    asset_class: str
    description: str = ""
    settings: ItemSettings = Field(default_factory=ItemSettings)
    chat: list[ChatTurn] = Field(default_factory=list)
    versions: list[Version] = Field(default_factory=list)
    current: int | None = None
    accepted: int | None = None  # the history entry saved as the design (characters)
    created: str = Field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))

    def get_version(self, n: int | None = None) -> Version | None:
        n = self.current if n is None else n
        return next((v for v in self.versions if v.n == n), None)

    def design_text(self, n: int | None = None) -> list[str]:
        """The description and every change on the way to drawing ``n`` (oldest first)."""
        out, v = [], self.get_version(n)
        while v is not None:
            if v.action == "refine":
                out.append(v.prompt)
                v = self.get_version(v.parent) if v.parent is not None else None
            else:
                out.append(v.description or v.prompt)
                break
        return out[::-1]


class ItemError(ValueError):
    pass


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:40] or "item"


def items_dir(project: Path) -> Path:
    return Path(project) / "items"


def item_dir(project: Path, asset_class: str, name: str) -> Path:
    return items_dir(project) / asset_class / slug(name)


def save(project: Path, rec: ItemRecord) -> Path:
    d = item_dir(project, rec.asset_class, rec.name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "item.json").write_text(rec.model_dump_json(indent=1), encoding="utf-8")
    return d


def load(project: Path, ref: str) -> tuple[ItemRecord, Path]:
    """``ref``: '<class>/<name>', a bare name (if unique) or an item folder."""
    p = Path(ref)
    candidates = []
    if (p / "item.json").exists():
        candidates = [p]
    else:
        root = items_dir(project)
        if "/" in ref:
            cls, name = ref.split("/", 1)
            candidates = [root / cls / slug(name)]
        else:
            candidates = [d for d in root.glob(f"*/{slug(ref)}") if (d / "item.json").exists()]
    candidates = [c for c in candidates if (c / "item.json").exists()]
    if not candidates:
        raise ItemError(f"no item {ref!r} in {project}")
    if len(candidates) > 1:
        raise ItemError(f"{ref!r} is ambiguous: {[f'{c.parent.name}/{c.name}' for c in candidates]}")
    d = candidates[0]
    try:
        return ItemRecord.model_validate_json((d / "item.json").read_text(encoding="utf-8")), d
    except Exception as exc:
        raise ItemError(f"{d / 'item.json'}: {exc}") from exc


def list_items(project: Path) -> list[tuple[ItemRecord, Path]]:
    out = []
    for f in sorted(items_dir(project).glob("*/*/item.json")):
        try:
            out.append((ItemRecord.model_validate_json(f.read_text(encoding="utf-8")), f.parent))
        except Exception:  # noqa: BLE001 - a broken item is skipped, not fatal for the list
            continue
    return out


def create(project: Path, asset_class: str, name: str, description: str = "",
           settings: ItemSettings | None = None) -> tuple[ItemRecord, Path]:
    from ggs.recipes import available_recipes

    if asset_class not in available_recipes() and asset_class != "character":
        raise ItemError(f"unknown asset type {asset_class!r}")
    settings = settings or ItemSettings()
    if asset_class == "character" and settings.character is None:
        settings.character = CharacterSettings()
    if not name.strip():
        raise ItemError("an item needs a name")
    if (item_dir(project, asset_class, name) / "item.json").exists():
        raise ItemError(f"there is already a {asset_class} called {name!r}")
    rec = ItemRecord(name=name.strip(), asset_class=asset_class, description=description.strip(),
                     settings=settings or ItemSettings())
    return rec, save(project, rec)


# ---------------------------------------------------------------- the drawing cache (outside the project)

def cache_dir(d: Path) -> Path:
    from ggs.paths import shared_dir

    key = hashlib.sha1(str(Path(d).resolve()).lower().encode()).hexdigest()[:16]
    return shared_dir() / "cache" / "drawings" / key


def _files(folder: Path, n: int) -> list[Path]:
    return sorted(folder.glob(f"v{n:03d}*")) if folder.is_dir() else []


def _stash(d: Path, n: int) -> None:
    """Move drawing ``n``'s files from the project to the cache."""
    files = _files(d / "versions", n)
    if not files:
        return
    c = cache_dir(d)
    c.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), str(c / f.name))


def _restore(d: Path, n: int) -> bool:
    """Bring drawing ``n`` back from the cache (True when its image was there)."""
    files = _files(cache_dir(d), n)
    if not any(f.suffix == ".png" and f.stem == f"v{n:03d}" for f in files):
        return False
    (d / "versions").mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), str(d / "versions" / f.name))
    return True


def _tidy(rec: ItemRecord, d: Path, keep: set[int] = frozenset()) -> None:
    """Only the drawing on screen stays in the project; the others go to the cache, which keeps the newest."""
    vdir = d / "versions"
    if vdir.is_dir():
        on_disk = {int(m.group(1)) for f in vdir.iterdir() if (m := re.match(r"v(\d{3})", f.name))}
        for n in on_disk - {rec.current} - set(keep):
            _stash(d, n)
        for f in vdir.glob("*_batch.json"):
            f.unlink()
    c = cache_dir(d)
    if c.is_dir():
        groups: dict[int, list[Path]] = {}
        for f in c.iterdir():
            m = re.match(r"v(\d{3})", f.name)
            if m:
                groups.setdefault(int(m.group(1)), []).append(f)
        for n in sorted(groups)[:-CACHE_PER_ITEM]:
            for f in groups[n]:
                f.unlink()


# ---------------------------------------------------------------- drawing

EDIT_TEMPLATE = ("{instruction}. Keep everything else the same: the same object, design, colours where not "
                 "mentioned and drawing style, centered on a plain flat pure white background.")


def generate(project: Path, ref: str, *, action: str, text: str = "", count: int = 1, seed: int | None = None,
             profile: str = "lite", enforce_vram: bool = True, model: str | None = None,
             progress=None, version: int | None = None) -> tuple[ItemRecord, list[Version]]:
    """Run one action on an item and record the new drawings (``goto``: show ``version`` again)."""
    from ggs.generate import choose_model
    from ggs.recipes import load_recipe

    rec, d = load(project, ref)
    project = Path(project)
    if action == "goto":
        if version is None or rec.get_version(version) is None:
            raise ItemError(f"there is no drawing {version!r} in {rec.name!r}")
        v = rec.get_version(version)
        if not _restore(d, v.n) and not (d / v.file).exists():
            _start_progress(rec, v.model or choose_model(load_recipe("sprite"), None, project))
            _draw(project, rec, d, [v], profile=profile, enforce_vram=enforce_vram)
        rec.current = v.n
        _tidy(rec, d)
        save(project, rec)
        return rec, []

    s = rec.settings
    recipe = load_recipe("sprite" if rec.asset_class == "character" else rec.asset_class)
    model = choose_model(recipe, model or s.model, project)
    cur = rec.get_version()
    parent = None
    if action == "new":
        if text.strip():
            rec.description = text.strip()
        if not rec.description:
            raise ItemError("describe it first")
        prompt, description = rec.description, rec.description
        seeds = [seed if seed is not None else random.randrange(1_000_000)]
    elif action == "refine":
        if cur is None:
            raise ItemError("there is nothing to refine yet: generate a first version")
        if not text.strip():
            raise ItemError("say what to change")
        prompt, description, parent = text.strip(), cur.description or rec.description, cur.n
        seeds = [cur.seed if seed is None else seed]
    elif action in ("regenerate", "variations"):
        if not rec.description:
            raise ItemError("describe it first")
        prompt = description = rec.description
        n = 1 if action == "regenerate" else max(1, count)
        base = random.randrange(1_000_000) if seed is None else seed
        seeds = [base + i for i in range(n)]
    else:
        raise ItemError(f"unknown action {action!r}")
    start = max((v.n for v in rec.versions), default=0) + 1
    snap = (s.character or CharacterSettings()).model_copy(deep=True) if rec.asset_class == "character" else None
    new = [Version(n=start + i, file="", prompt=prompt, action=action, seed=sd, parent=parent,
                   description=description, character=snap, model=model, style=s.style,
                   style_weight=s.style_weight) for i, sd in enumerate(seeds)]
    _start_progress(rec, model)
    if progress:
        progress("drawing")
    _draw(project, rec, d, new, profile=profile, enforce_vram=enforce_vram)
    rec.versions += new
    ok = [v for v in new if v.passed and v.file]
    if ok:
        rec.current = ok[0].n
    elif new and new[0].file:
        rec.current = new[0].n
    _tidy(rec, d)
    save(project, rec)
    return rec, new


def _start_progress(rec: ItemRecord, model: str) -> None:
    from ggs.pipeline import progress

    kind = "character" if rec.asset_class == "character" else rec.asset_class
    plan = ["encode", "denoise", "decode", "matte"] if model == "sdxl" else ["encode", "denoise", "matte"]
    progress.start(f"{kind}:{model}", plan)


def _ensure(project: Path, rec: ItemRecord, d: Path, n: int, **kw) -> Path:
    """Drawing ``n``'s raw image on disk (from the project, the cache, or drawn again): for a change on top."""
    raw = d / "versions" / f"v{n:03d}_raw.png"
    if raw.exists() or _restore(d, n) and raw.exists():
        return raw
    v = rec.get_version(n)
    if v is None:
        raise ItemError(f"drawing {n} is not in the history")
    _draw(project, rec, d, [v], **kw)
    if not raw.exists():
        raise ItemError(f"drawing {n} could not be made again")
    return raw


def _draw(project: Path, rec: ItemRecord, d: Path, vs: list[Version], *, profile: str,
          enforce_vram: bool) -> None:
    """Draw ``vs`` (same recipe, one seed each) into versions/vNNN*; fills in file, passed, failures, size."""
    from PIL import Image

    first = vs[0]
    edit_raw = None
    if first.action == "refine" and first.parent is not None:
        edit_raw = Image.open(_ensure(project, rec, d, first.parent, profile=profile,
                                      enforce_vram=enforce_vram)).convert("RGB")
    if rec.asset_class == "character":
        _draw_character(project, d, vs, edit_raw, profile=profile, enforce_vram=enforce_vram)
    else:
        _draw_asset(project, rec, d, vs, edit_raw, profile=profile, enforce_vram=enforce_vram)


def _draw_asset(project: Path, rec: ItemRecord, d: Path, vs: list[Version], edit_raw, *, profile: str,
                enforce_vram: bool) -> None:
    from ggs.generate import run_recipe
    from ggs.recipes import load_recipe

    recipe = load_recipe(rec.asset_class)
    first = vs[0]
    model = first.model
    reference, prompt_desc = None, first.description or rec.description
    if first.action == "refine":
        if recipe.kind == "sprite" and model != "sdxl" and edit_raw is not None:
            reference = [edit_raw]
            prompt_desc = EDIT_TEMPLATE.format(instruction=first.prompt.rstrip("."))
        else:  # no image editing: draw again with the change added, same seed
            prompt_desc = f"{prompt_desc}, {first.prompt}"
    vdir = d / "versions"
    vdir.mkdir(parents=True, exist_ok=True)
    out = vdir / f"v{first.n:03d}.png"
    results, _ = run_recipe(recipe, prompt_desc, out, model=model, seeds=[v.seed for v in vs], profile=profile,
                            keep_failed=True, enforce_vram=enforce_vram, size=rec.settings.gen_size, debug=True,
                            style_ref=first.style, style_weight=first.style_weight, project=Path(project),
                            reference=reference, prompt_is_final=reference is not None)
    for i, (v, r) in enumerate(zip(vs, results)):
        # run_recipe names several seeds out_01, out_02 ... and failed ones *_QA_FAILED: rename all to vNNN
        old_stem = out.stem if len(results) == 1 else f"{out.stem}_{i + 1:02d}"
        new_stem = f"v{v.n:03d}"
        for f in sorted(vdir.glob(f"{old_stem}*")):
            rest = f.name[len(old_stem):]
            if rest == ".json":  # the batch sidecar
                continue
            target = vdir / (new_stem + rest.replace("_QA_FAILED", ""))
            if f != target:
                f.replace(target)
        v.file = f"versions/{new_stem}.png" if r.path is not None else ""
        v.passed, v.failures, v.size = r.passed, list(r.failures), r.size
    side = out.with_suffix(".json")
    if side.exists():
        side.unlink()


def character_guide(cs: CharacterSettings, size: int = 1024):
    """The pose guide for a character's settings: (rig, guide, body)."""
    from ggs.rig.body import PRESETS, BodyShape
    from ggs.rig.make import guide_puppet
    from ggs.rig.master import render_guide
    from ggs.rig.poses import pose_angles, rig_for

    rig = rig_for(cs.view)
    body = BodyShape(**cs.body) if cs.body else PRESETS.get(cs.body_preset, PRESETS["standard"])
    puppet, lengths = guide_puppet(rig, body)
    guide = render_guide(rig, puppet, lengths, size=size, angles=pose_angles(cs.view, cs.pose))
    return rig, guide, body


def limbs_missed(on, guide, rig, radius: int = 8, need: float = 0.6) -> list[str]:
    """Arm and leg bones whose length is not covered by the drawing (sampled from 30% to 100% of each bone)."""
    import numpy as np

    h, w = on.shape
    out = []
    for bone in rig.bones:
        base = bone.name.rsplit("_", 1)[0]
        if base not in ("upper_arm", "forearm", "thigh", "shin"):
            continue
        (x0, y0), (x1, y1) = guide.spec.joints[bone.name]
        hits = 0
        ts = np.linspace(0.3, 1.0, 6)
        for t in ts:
            x, y = round(x0 + t * (x1 - x0)), round(y0 + t * (y1 - y0))
            ya, yb, xa, xb = max(0, y - radius), min(h, y + radius + 1), max(0, x - radius), min(w, x + radius + 1)
            hits += bool(on[ya:yb, xa:xb].any()) if ya < yb and xa < xb else 0
        if hits / len(ts) < need:
            out.append(bone.name.replace("_", " "))
    return out


def skeleton_json(rig, guide, cs: CharacterSettings, image: str, size: tuple[int, int], x0: float, y0: float):
    return {
        "format": "ggs.skeleton", "version": 1, "rig": rig.id, "view": cs.view,
        "image": image, "size": [int(size[0]), int(size[1])],
        "px_per_height": round(guide.spec.height_px, 3),
        "bones": [{"name": b.name, "parent": b.parent,
                   "start": [round(guide.spec.joints[b.name][0][0] - x0, 1),
                             round(guide.spec.joints[b.name][0][1] - y0, 1)],
                   "end": [round(guide.spec.joints[b.name][1][0] - x0, 1),
                           round(guide.spec.joints[b.name][1][1] - y0, 1)]} for b in rig.bones],
    }


def _draw_character(project: Path, d: Path, vs: list[Version], edit_raw, *, profile: str,
                    enforce_vram: bool) -> None:
    from PIL import Image

    from ggs.pipeline.post import trim
    from ggs.rig.cutter import _grow
    from ggs.rig.make import draw_character

    first = vs[0]
    cs = first.character or CharacterSettings()
    lora, weight, style = None, 1.0, ""
    if first.style:
        from ggs.style.package import StyleError, resolve_style

        meta, lora = resolve_style(first.style, project)
        if first.model != "stub" and meta.base_model != first.model:
            raise StyleError(f"style {meta.name!r} was trained on {meta.base_model!r}; it cannot load on "
                             f"{first.model!r}")
        style = f"{meta.trigger}, "
        weight = first.style_weight or meta.recommended_weight
    rig, guide, body = character_guide(cs)
    drawn = draw_character(guide, first.description, view=cs.view, body=body, seeds=[v.seed for v in vs],
                           model=first.model, profile=profile, lora=lora, lora_scale=weight, style=style,
                           edit_of=edit_raw, instruction=first.prompt if first.action == "refine" else "",
                           enforce_vram=enforce_vram, rig=rig)
    vdir = d / "versions"
    vdir.mkdir(parents=True, exist_ok=True)
    guide_mask = guide.labels >= 0
    grow = max(2, round(0.04 * guide.image.size[0]))
    for v, (rgba, raw, _sd) in zip(vs, drawn):
        stem = f"v{v.n:03d}"
        on = rgba[..., 3] >= 128
        precision = float((on & _grow(guide_mask, grow)).sum() / max(on.sum(), 1))
        missed = limbs_missed(on, guide, rig, radius=grow)
        failures = []
        if not on.any():
            failures.append("coverage: nothing was drawn")
        elif precision < 0.6:
            failures.append(f"pose: the drawing does not sit on the pose ({precision:.2f})")
        elif missed:
            failures.append(f"pose: these limbs are not where the pose puts them: {', '.join(missed)}")
        raw.save(vdir / f"{stem}_raw.png")
        guide.image.save(vdir / f"{stem}_guide.png")
        file, size = "", None
        if on.any():
            out, box = trim(rgba, 4)
            Image.fromarray(out, "RGBA").save(vdir / f"{stem}.png")
            sk = skeleton_json(rig, guide, cs, f"{stem}.png", (out.shape[1], out.shape[0]), box[0], box[1])
            (vdir / f"{stem}_skeleton.json").write_text(json.dumps(sk, indent=1), encoding="utf-8")
            file, size = f"versions/{stem}.png", (int(out.shape[1]), int(out.shape[0]))
        v.file, v.passed, v.failures, v.size = file, not failures, failures, size


# ---------------------------------------------------------------- accepting a design

def design_dir(d: Path) -> Path:
    return Path(d) / "design"


def accept(project: Path, ref: str) -> tuple[ItemRecord, Path]:
    """Save the drawing on screen as the item's design (``design/design.png`` + raw, guide, skeleton,
    design.json with its recipe). Animating and the sliced version start from the design."""
    rec, d = load(project, ref)
    v = rec.get_version()
    if v is None or not v.file or not (d / v.file).exists():
        raise ItemError(f"{rec.name!r} has no drawing to keep yet")
    out = design_dir(d)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    stem = f"v{v.n:03d}"
    for f in _files(d / "versions", v.n):
        target = out / ("design" + f.name[len(stem):])
        if f.name.endswith("_skeleton.json"):
            sk = json.loads(f.read_text(encoding="utf-8"))
            sk["image"] = "design.png"
            target.write_text(json.dumps(sk, indent=1), encoding="utf-8")
        else:
            shutil.copy2(f, target)
    info = {"format": "ggs.design", "version": 1, "item": rec.name, "drawing": v.model_dump(),
            "text": rec.design_text(v.n), "accepted": dt.datetime.now().isoformat(timespec="seconds")}
    (out / "design.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    rec.accepted = v.n
    save(project, rec)
    return rec, out


# ---------------------------------------------------------------- messages and export

def summary(new: list[Version]) -> str:
    good = [v for v in new if v.passed]
    if len(new) == 1:
        v = new[0]
        return "Here it is." if v.passed else "I drew it, but it did not pass the quality check: " + friendly(
            v.failures)
    return f"{len(good)} of {len(new)} passed the quality check."


FRIENDLY = [
    (r"halo", "a light rim is left around the edges"),
    (r"leftover_background|background-coloured", "some background is left on the object"),
    (r"cut_off|touches|reaches the frame edge", "it touches the edge of the picture"),
    (r"islands|specks", "there are stray specks around it"),
    (r"coverage|covers", "it is too small or too large in the picture"),
    (r"background_flat|spread", "the background behind it was not plain"),
    (r"seam", "the edges do not repeat smoothly"),
    (r"border", "the frame could not be found"),
    (r"centre not flat", "the middle is too busy to stretch"),
    (r"pose:", "the drawing did not follow the pose"),
]


def friendly(failures: list[str]) -> str:
    """QA failure codes -> a short plain sentence."""
    said = []
    for f in failures:
        for pat, words in FRIENDLY:
            if re.search(pat, f) and words not in said:
                said.append(words)
                break
        else:
            said.append(f.split(":")[0].replace("_", " "))
    return "; ".join(said) + ". Try Regenerate, or describe it a little differently." if said else ""


def export_item(project: Path, ref: str, out: Path | None = None, version: int | None = None,
                size: str | None = None) -> Path:
    """Copy the item's image to ``<project>/exports/<class>/<name>.png`` at its export size: a character's
    accepted design when there is one, else the drawing on screen (``version``: that one, if on screen)."""
    from PIL import Image

    rec, d = load(project, ref)
    design = design_dir(d) / "design.png"
    if version is None and rec.accepted is not None and design.exists():
        src, n = design, rec.accepted
        v = rec.get_version(n)
    else:
        v = rec.get_version(version)
        if v is None or not v.file or not (d / v.file).exists():
            raise ItemError(f"{rec.name!r} has no drawing to export")
        src, n = d / v.file, v.n
    img = Image.open(src)
    side = SIZES.get(size or rec.settings.size)
    if side and max(img.size) > side:
        f = side / max(img.size)
        img = img.convert("RGBa").resize((max(1, round(img.width * f)), max(1, round(img.height * f))),
                                         Image.LANCZOS).convert("RGBA")
    out = Path(out) if out else Path(project) / "exports" / rec.asset_class / f"{slug(rec.name)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    for extra in ("_hover.png", "_pressed.png", "_slices.json", "_far.png", "_skeleton.json"):  # companions
        comp = src.with_name(src.stem + extra)
        if comp.exists():
            dst = out.with_name(out.stem + extra)
            if extra.endswith(".png") and side:
                Image.open(comp).convert("RGBA").resize(img.size, Image.LANCZOS).save(dst)
            elif extra == "_skeleton.json":
                sk = json.loads(comp.read_text(encoding="utf-8"))
                k = img.size[0] / max(1, sk["size"][0])
                sk["image"], sk["size"] = out.name, list(img.size)
                sk["px_per_height"] = round(sk["px_per_height"] * k, 3)
                for b in sk["bones"]:
                    b["start"] = [round(c * k, 1) for c in b["start"]]
                    b["end"] = [round(c * k, 1) for c in b["end"]]
                dst.write_text(json.dumps(sk, indent=1), encoding="utf-8")
            else:
                dst.write_bytes(comp.read_bytes())
    meta = {"item": rec.name, "class": rec.asset_class, "version": n, "prompt": v.prompt if v else "",
            "seed": v.seed if v else None}
    out.with_suffix(".export.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out
