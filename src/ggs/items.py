"""Project items (v2): named assets the user creates, talks to and exports later.

    <project>/items/<class>/<name>/item.json          what it is, its settings, the chat, its versions
    <project>/items/<class>/<name>/versions/v001.png  every generation (with v001.json sidecar and v001_raw.png)

Chat actions (the app's buttons; also ``ggs asset gen``):
  new         first drawing from the description
  refine      an instruction ("make it blue"): klein edits the current version (same seed); models without image
              editing redraw from description + instruction with the same seed
  regenerate  the latest description, a new seed
  variations  several new seeds at once
Seeds are kept in the versions, so a version can be redone exactly; the app shows them only in Advanced.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SIZES = {  # the app's size tiles -> longest side of the exported image (None = as generated)
    "small": 128, "medium": 256, "large": 512, "full": None,
}


class ChatTurn(BaseModel):
    role: Literal["user", "app"]
    text: str
    versions: list[int] = Field(default_factory=list)
    time: str = Field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))


class Version(BaseModel):
    n: int
    file: str  # relative to the item folder; empty when nothing passed QA and nothing was kept
    prompt: str  # what the model was asked (description, or the refine instruction)
    action: Literal["new", "refine", "regenerate", "variations"]
    seed: int
    parent: int | None = None  # the version a refine edited
    passed: bool = True
    failures: list[str] = Field(default_factory=list)
    size: tuple[int, int] | None = None


class CharacterSettings(BaseModel):
    """How a character is drawn: view, body shape and pose (``ggs.rig.poses``), and the output kind."""
    view: Literal["side", "front"] = "front"
    body_preset: str = "standard"  # ggs.rig.body.PRESETS name, used when ``body`` is empty
    body: dict = Field(default_factory=dict)  # BodyShape fields (the editor's sliders)
    pose: str | dict = "a_pose"  # preset name or bone angles
    mode: Literal["single"] = "single"  # single image; sliced sheets and frame-by-frame come next


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
    created: str = Field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))

    def get_version(self, n: int | None = None) -> Version | None:
        n = self.current if n is None else n
        return next((v for v in self.versions if v.n == n), None)


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


EDIT_TEMPLATE = ("{instruction}. Keep everything else the same: the same object, design, colours where not "
                 "mentioned and drawing style, centered on a plain flat pure white background.")


def generate(project: Path, ref: str, *, action: str, text: str = "", count: int = 1, seed: int | None = None,
             profile: str = "lite", enforce_vram: bool = True, model: str | None = None,
             progress=None) -> tuple[ItemRecord, list[Version]]:
    """Run one chat action on an item and record the new versions."""
    from PIL import Image

    from ggs.generate import choose_model, run_recipe
    from ggs.recipes import load_recipe

    rec, d = load(project, ref)
    if rec.asset_class == "character":
        new = _generate_character(Path(project), rec, d, action=action, text=text, count=count, seed=seed,
                                  profile=profile, enforce_vram=enforce_vram, model=model)
        return _record(project, rec, action, text, new), new
    recipe = load_recipe(rec.asset_class)
    s = rec.settings
    model = choose_model(recipe, model or s.model, Path(project))
    cur = rec.get_version()
    reference, prompt_desc, parent = None, rec.description, None
    if action == "new":
        if text.strip():
            rec.description = text.strip()
        prompt_desc = rec.description
        if not prompt_desc:
            raise ItemError("describe the item first")
        seeds = [seed if seed is not None else random.randrange(1_000_000)]
    elif action == "refine":
        if cur is None:
            raise ItemError("there is nothing to refine yet: generate a first version")
        if not text.strip():
            raise ItemError("say what to change")
        seeds = [cur.seed if seed is None else seed]
        raw = d / "versions" / f"v{cur.n:03d}_raw.png"
        if recipe.kind == "sprite" and model != "sdxl" and raw.exists():
            reference = [Image.open(raw).convert("RGB")]
            prompt_desc = EDIT_TEMPLATE.format(instruction=text.strip().rstrip("."))
        else:  # no image editing: redraw with the instruction added, same seed
            prompt_desc = f"{rec.description}, {text.strip()}"
        parent = cur.n
    elif action in ("regenerate", "variations"):
        if not rec.description:
            raise ItemError("describe the item first")
        n = 1 if action == "regenerate" else max(1, count)
        base = random.randrange(1_000_000) if seed is None else seed
        seeds = [base + i for i in range(n)]
    else:
        raise ItemError(f"unknown action {action!r}")

    start = max((v.n for v in rec.versions), default=0) + 1
    vdir = d / "versions"
    vdir.mkdir(parents=True, exist_ok=True)
    out = vdir / f"v{start:03d}.png"
    if progress:
        progress("drawing")
    results, _ = run_recipe(recipe, prompt_desc, out, model=model, seeds=seeds, profile=profile,
                            keep_failed=True, enforce_vram=enforce_vram, size=s.gen_size, debug=True,
                            style_ref=s.style, style_weight=s.style_weight, project=Path(project),
                            reference=reference, prompt_is_final=reference is not None)
    new = []
    for i, r in enumerate(results):
        n = start + i
        # run_recipe names several seeds out_01, out_02 ... and failed ones *_QA_FAILED: rename all to vNNN
        old_stem = out.stem if len(results) == 1 else f"{out.stem}_{i + 1:02d}"
        new_stem = f"v{n:03d}"
        for f in sorted(vdir.glob(f"{old_stem}*")):
            rest = f.name[len(old_stem):]
            if rest == ".json":  # the batch sidecar is moved below
                continue
            target = vdir / (new_stem + rest.replace("_QA_FAILED", ""))
            if f != target:
                f.replace(target)
        produced = vdir / f"{new_stem}.png" if r.path is not None else None
        v = Version(n=n, file=f"versions/{new_stem}.png" if produced is not None else "",
                    prompt=text.strip() if action == "refine" else prompt_desc, action=action, seed=r.seed,
                    parent=parent, passed=r.passed, failures=r.failures, size=r.size)
        new.append(v)
    side = out.with_suffix(".json")
    if side.exists():
        side.replace(vdir / f"v{start:03d}_batch.json")
    return _record(project, rec, action, text, new), new


def _record(project: Path, rec: ItemRecord, action: str, text: str, new: list[Version]) -> ItemRecord:
    """Add new versions to the item, pick the current one, write the chat turns, save."""
    rec.versions += new
    ok = [v for v in new if v.passed and v.file]
    if ok:
        rec.current = ok[0].n
    elif new and new[0].file:
        rec.current = new[0].n
    user_text = {"new": text or rec.description, "refine": text, "regenerate": "(regenerate)",
                 "variations": f"({len(new)} variations)"}[action]
    rec.chat.append(ChatTurn(role="user", text=user_text))
    rec.chat.append(ChatTurn(role="app", text=summary(new), versions=[v.n for v in new]))
    save(project, rec)
    return rec


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


def _generate_character(project: Path, rec: ItemRecord, d: Path, *, action: str, text: str, count: int,
                        seed: int | None, profile: str, enforce_vram: bool, model: str | None) -> list[Version]:
    from PIL import Image

    from ggs.generate import choose_model
    from ggs.pipeline.post import trim
    from ggs.recipes import load_recipe
    from ggs.rig.cutter import _grow
    from ggs.rig.make import draw_character

    cs = rec.settings.character or CharacterSettings()
    model = choose_model(load_recipe("sprite"), model or rec.settings.model, project)
    cur = rec.get_version()
    edit_of, parent, instruction = None, None, ""
    if action == "new":
        if text.strip():
            rec.description = text.strip()
        if not rec.description:
            raise ItemError("describe the character first")
        seeds = [seed if seed is not None else random.randrange(1_000_000)]
    elif action == "refine":
        if cur is None:
            raise ItemError("there is nothing to refine yet: generate a first version")
        if not text.strip():
            raise ItemError("say what to change")
        edit_of = Image.open(d / f"versions/v{cur.n:03d}_raw.png").convert("RGB")
        seeds, parent, instruction = [cur.seed if seed is None else seed], cur.n, text
    elif action in ("regenerate", "variations"):
        if not rec.description:
            raise ItemError("describe the character first")
        base = random.randrange(1_000_000) if seed is None else seed
        seeds = [base + i for i in range(1 if action == "regenerate" else max(1, count))]
    else:
        raise ItemError(f"unknown action {action!r}")
    lora, weight, style = None, 1.0, ""
    if rec.settings.style:
        from ggs.style.package import resolve_style

        meta, lora = resolve_style(rec.settings.style, project)
        style = f"{meta.trigger}, "
        weight = rec.settings.style_weight or meta.recommended_weight
    rig, guide, body = character_guide(cs)
    drawn = draw_character(guide, rec.description, view=cs.view, body=body, seeds=seeds, model=model,
                           profile=profile, lora=lora, lora_scale=weight, style=style, edit_of=edit_of,
                           instruction=instruction, enforce_vram=enforce_vram)
    vdir = d / "versions"
    vdir.mkdir(parents=True, exist_ok=True)
    start = max((v.n for v in rec.versions), default=0) + 1
    guide_mask = guide.labels >= 0
    grow = max(2, round(0.04 * guide.image.size[0]))
    new = []
    for i, (rgba, raw, sd) in enumerate(drawn):
        n = start + i
        stem = f"v{n:03d}"
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
            x0, y0 = box[0], box[1]
            skeleton = {
                "format": "ggs.skeleton", "version": 1, "rig": rig.id, "view": cs.view,
                "image": f"{stem}.png", "size": [int(out.shape[1]), int(out.shape[0])],
                "px_per_height": round(guide.spec.height_px, 3),
                "bones": [{"name": b.name, "parent": b.parent,
                           "start": [round(guide.spec.joints[b.name][0][0] - x0, 1),
                                     round(guide.spec.joints[b.name][0][1] - y0, 1)],
                           "end": [round(guide.spec.joints[b.name][1][0] - x0, 1),
                                   round(guide.spec.joints[b.name][1][1] - y0, 1)]} for b in rig.bones],
            }
            (vdir / f"{stem}_skeleton.json").write_text(json.dumps(skeleton, indent=1), encoding="utf-8")
            file, size = f"versions/{stem}.png", (int(out.shape[1]), int(out.shape[0]))
        new.append(Version(n=n, file=file, prompt=text.strip() if action == "refine" else rec.description,
                           action=action, seed=sd, parent=parent, passed=not failures, failures=failures,
                           size=size))
    return new


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
    """Copy a version to ``<project>/exports/<class>/<name>.png`` at the item's export size."""
    from PIL import Image

    rec, d = load(project, ref)
    v = rec.get_version(version)
    if v is None or not v.file:
        raise ItemError(f"{rec.name!r} has no version to export")
    img = Image.open(d / v.file)
    side = SIZES.get(size or rec.settings.size)
    if side and max(img.size) > side:
        f = side / max(img.size)
        img = img.convert("RGBa").resize((max(1, round(img.width * f)), max(1, round(img.height * f))),
                                         Image.LANCZOS).convert("RGBA")
    out = Path(out) if out else Path(project) / "exports" / rec.asset_class / f"{slug(rec.name)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    for extra in ("_hover.png", "_pressed.png", "_slices.json", "_far.png", "_skeleton.json"):  # companions
        src = (d / v.file).with_name((d / v.file).stem + extra)
        if src.exists():
            dst = out.with_name(out.stem + extra)
            if extra.endswith(".png") and side:
                Image.open(src).convert("RGBA").resize(img.size, Image.LANCZOS).save(dst)
            else:
                dst.write_bytes(src.read_bytes())
    meta = {"item": rec.name, "class": rec.asset_class, "version": v.n, "prompt": v.prompt, "seed": v.seed}
    out.with_suffix(".export.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out
