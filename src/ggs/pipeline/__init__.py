"""Generation core: model file resolution, VRAM-staged generators, matting and post-processing.

Modules that import torch/diffusers (flux2, matting) need the ``gpu`` extra and are imported lazily.
"""
