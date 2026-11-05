"""Model registry with license gating."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, model_validator

from ggs.paths import registry_path

Kind = Literal["base", "component", "lora", "motion", "segmentation", "matting", "captioning", "dataset"]


class ModelEntry(BaseModel):
    id: str
    kind: Kind
    name: str
    license: str
    commercial: Literal["yes", "no", "verify"]
    min_vram_gb: Optional[float] = None
    base: Optional[str] = None  # for LoRAs: id of the base model they were trained on
    source: str
    notes: str = ""

    @property
    def bundle_eligible(self) -> bool:
        """Only confirmed-commercial entries may be bundled or enabled by default."""
        return self.commercial == "yes"


def load_registry(path: str | Path | None = None) -> dict[str, ModelEntry]:
    path = Path(path) if path else registry_path()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = [ModelEntry.model_validate(e) for e in data["entries"]]
    ids = [e.id for e in entries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate model ids")
    by_id = {e.id: e for e in entries}
    for e in entries:
        if e.kind == "lora":
            if not e.base:
                raise ValueError(f"{path}: LoRA {e.id!r} must name its base model")
            if e.base not in by_id:
                raise ValueError(f"{path}: LoRA {e.id!r} references unknown base {e.base!r}")
    return by_id


def compatible_loras(registry: dict[str, ModelEntry], base_id: str) -> list[ModelEntry]:
    return [e for e in registry.values() if e.kind == "lora" and e.base == base_id]
