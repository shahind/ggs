"""Pick the generator class for a pipeline's loader (models/pipelines.yaml). Heavy imports stay lazy."""

from __future__ import annotations

from pathlib import Path

from ggs.pipeline.models import PipelineSpec


def make_generator(spec: PipelineSpec, files: dict[str, Path]):
    d = spec.defaults
    if spec.loader == "flux2_klein":
        from ggs.pipeline.flux2 import Flux2KleinGenerator

        return Flux2KleinGenerator(files, steps=int(d.get("steps", 4)), guidance=float(d.get("guidance", 1.0)))
    if spec.loader == "sdxl":
        from ggs.pipeline.sdxl import SdxlGenerator

        return SdxlGenerator(files, steps=int(d.get("steps", 30)), guidance=float(d.get("guidance", 6.0)),
                             speed_steps=int(d.get("speed_steps", 8)), negative=str(d.get("negative", "")))
    raise ValueError(f"loader {spec.loader!r} is not implemented")
