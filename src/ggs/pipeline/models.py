"""Model folders (~/.ggs/config.yaml) and resolution of pipeline files (models/pipelines.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from ggs.paths import repo_root, shared_dir

PROFILES = ("standard", "lite")


class UserConfig(BaseModel):
    model_dirs: list[str] = Field(default_factory=list)  # extra folders, searched after ~/.ggs/models


def config_path() -> Path:
    return shared_dir() / "config.yaml"


def load_user_config(path: Path | None = None) -> UserConfig:
    path = path or config_path()
    if not path.exists():
        return UserConfig()
    try:
        return UserConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc


def save_user_config(cfg: UserConfig, path: Path | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")
    return path


def model_dirs(cfg: UserConfig | None = None) -> list[Path]:
    cfg = cfg or load_user_config()
    # last: the app's own models folder, where the installer puts the bundled LoRAs (downloads.yaml `bundled`)
    dirs = [shared_dir() / "models", *(Path(d).expanduser() for d in cfg.model_dirs), repo_root() / "models"]
    seen, out = set(), []
    for d in dirs:
        key = str(d.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


LOADERS = ("flux2_klein", "sdxl")


class TrainingSpec(BaseModel):
    """Style-LoRA training defaults for a base model (CLI flags override them)."""
    resolution: int = 512
    max_resolution: dict[str, int] = Field(default_factory=lambda: {"lite": 768, "standard": 1024})
    steps: int = Field(1500, ge=1)
    lr: float = Field(1e-4, gt=0)
    rank: int = Field(16, ge=1, le=256)


class PipelineSpec(BaseModel):
    registry: str
    loader: str
    configs: list[str]
    text_encoder: list[str]
    transformer: dict[str, list[str]]  # the denoiser per profile: a transformer (FLUX) or a UNet (SDXL)
    vae: list[str]
    extras: dict[str, list[str]] = Field(default_factory=dict)  # optional files, used when present
    defaults: dict[str, float | str] = Field(default_factory=dict)
    training: TrainingSpec | None = None  # None = this base cannot train styles

    @field_validator("loader")
    @classmethod
    def _known_loader(cls, v: str) -> str:
        if v not in LOADERS:
            raise ValueError(f"unknown loader {v!r}; known: {LOADERS}")
        return v


class MattingSpec(BaseModel):
    registry: str
    path: list[str]


class MotionSpec(BaseModel):
    """Text-to-motion: the motion model, and the text encoder (base LLM + LLM2Vec adapters) it reads prompts with."""

    registry: str
    model: list[str]
    text_encoder: list[str]
    mntp: list[str]
    supervised: list[str]


class FramesSpec(BaseModel):
    """Pose-driven frames. AnimateDiff: SD 1.5 base, motion adapter, OpenPose ControlNet, IP-Adapter folder.
    One-to-All: ``model`` (its checkpoint), ``base`` (Wan2.1 1.3B: VAE, tokenizer, configs), ``text_encoder``
    (umT5-xxl file)."""

    registry: str
    base: list[str] = Field(default_factory=list)
    motion: list[str] = Field(default_factory=list)
    controlnet: list[str] = Field(default_factory=list)
    ip_adapter: list[str] = Field(default_factory=list)
    model: list[str] = Field(default_factory=list)
    text_encoder: list[str] = Field(default_factory=list)


class PipelinesFile(BaseModel):
    pipelines: dict[str, PipelineSpec]
    matting: dict[str, MattingSpec] = Field(default_factory=dict)
    captioning: dict[str, MattingSpec] = Field(default_factory=dict)  # same shape: registry id + paths
    pose: dict[str, MattingSpec] = Field(default_factory=dict)  # joint detectors (same shape)
    frames: dict[str, FramesSpec] = Field(default_factory=dict)  # pose-driven frame models (AnimateDiff)
    motion: dict[str, MotionSpec] = Field(default_factory=dict)


def load_pipelines(path: Path | None = None) -> PipelinesFile:
    path = path or (repo_root() / "models" / "pipelines.yaml")
    try:
        return PipelinesFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError(f"{path}: {exc}") from exc


class MissingModelError(FileNotFoundError):
    pass


WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pth", ".ckpt")


def _usable(p: Path, need_weights: bool) -> bool:
    if p.is_file():
        return True
    if not p.is_dir():
        return False
    return not need_weights or any(f.suffix in WEIGHT_SUFFIXES for f in p.rglob("*") if f.is_file())


def resolve(candidates: list[str], dirs: list[Path], what: str, need_weights: bool = True) -> Path:
    """First usable ``dir / candidate`` (dirs outer loop, so ~/.ggs/models wins over shared folders).
    A folder only counts when it holds weight files, unless ``need_weights`` is False (config folders)."""
    for d in dirs:
        for c in candidates:
            p = d / Path(c)
            if _usable(p, need_weights):
                return p
    tried = ", ".join(candidates)
    raise MissingModelError(f"{what}: none of [{tried}] found in {[str(d) for d in dirs]}. "
                            "Download it into ~/.ggs/models or add a folder with `ggs models add-dir`.")


def resolve_pipeline(name: str, profile: str, dirs: list[Path] | None = None,
                     spec_file: PipelinesFile | None = None) -> dict[str, Path]:
    spec_file = spec_file or load_pipelines()
    if name not in spec_file.pipelines:
        raise ValueError(f"unknown pipeline {name!r}; known: {sorted(spec_file.pipelines)}")
    if profile not in PROFILES:
        raise ValueError(f"profile {profile!r} cannot run generation (needs one of {PROFILES})")
    spec = spec_file.pipelines[name]
    dirs = dirs or model_dirs()
    tr = spec.transformer.get(profile) or spec.transformer.get("lite") or []
    files = {
        "configs": resolve(spec.configs, dirs, f"{name} configs", need_weights=False),
        "text_encoder": resolve(spec.text_encoder, dirs, f"{name} text encoder"),
        "transformer": resolve(tr, dirs, f"{name} denoiser ({profile})"),
        "vae": resolve(spec.vae, dirs, f"{name} vae"),
    }
    for key, candidates in spec.extras.items():  # optional: left out when not on disk
        try:
            files[key] = resolve(candidates, dirs, f"{name} {key}")
        except MissingModelError:
            pass
    return files


def resolve_captioner(name: str, dirs: list[Path] | None = None, spec_file: PipelinesFile | None = None) -> Path:
    spec_file = spec_file or load_pipelines()
    if name not in spec_file.captioning:
        raise ValueError(f"unknown captioner {name!r}; known: {sorted(spec_file.captioning)}")
    return resolve(spec_file.captioning[name].path, dirs or model_dirs(), f"captioner {name}")


def resolve_matting(name: str, dirs: list[Path] | None = None, spec_file: PipelinesFile | None = None) -> Path:
    spec_file = spec_file or load_pipelines()
    if name not in spec_file.matting:
        raise ValueError(f"unknown matting model {name!r}; known: {sorted(spec_file.matting)}")
    return resolve(spec_file.matting[name].path, dirs or model_dirs(), f"matting {name}")


def resolve_pose(name: str, dirs: list[Path] | None = None, spec_file: PipelinesFile | None = None) -> Path:
    spec_file = spec_file or load_pipelines()
    if name not in spec_file.pose:
        raise ValueError(f"unknown pose model {name!r}; known: {sorted(spec_file.pose)}")
    return resolve(spec_file.pose[name].path, dirs or model_dirs(), f"pose model {name}")


def resolve_frames(name: str, dirs: list[Path] | None = None, spec_file: PipelinesFile | None = None
                   ) -> dict[str, Path]:
    spec_file = spec_file or load_pipelines()
    if name not in spec_file.frames:
        raise ValueError(f"unknown frames model {name!r}; known: {sorted(spec_file.frames)}")
    s, dirs = spec_file.frames[name], dirs or model_dirs()
    keys = ("base", "motion", "controlnet", "ip_adapter", "model", "text_encoder")
    return {k: resolve(getattr(s, k), dirs, f"{name} {k}", need_weights=(k != "base" or not s.model))
            for k in keys if getattr(s, k)}


def _motion_spec(name: str, spec_file: PipelinesFile | None) -> MotionSpec:
    spec_file = spec_file or load_pipelines()
    if name not in spec_file.motion:
        raise ValueError(f"unknown motion model {name!r}; known: {sorted(spec_file.motion)}")
    return spec_file.motion[name]


def resolve_motion(name: str, dirs: list[Path] | None = None, spec_file: PipelinesFile | None = None) -> Path:
    """Folder of the motion model (config.yaml + model.safetensors)."""
    return resolve(_motion_spec(name, spec_file).model, dirs or model_dirs(), f"motion model {name}")


def resolve_motion_text(name: str, dirs: list[Path] | None = None,
                        spec_file: PipelinesFile | None = None) -> dict[str, Path]:
    """Folders of the motion model's text encoder: base LLM, LLM2Vec MNTP adapter, supervised adapter."""
    spec, dirs = _motion_spec(name, spec_file), dirs or model_dirs()
    return {"base": resolve(spec.text_encoder, dirs, f"{name} text encoder"),
            "mntp": resolve(spec.mntp, dirs, f"{name} text encoder adapter (mntp)"),
            "supervised": resolve(spec.supervised, dirs, f"{name} text encoder adapter (supervised)")}
