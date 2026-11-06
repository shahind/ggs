"""See-through (SIGGRAPH 2026), vendored layer-decomposition inference.

Source: https://github.com/shitagaki-lab/see-through at commit df019de5129d6c4b406587a14c3501669441a783, Apache License 2.0 (LICENSE in this
folder). Copied: ``common/modules/layerdiffuse`` (diffusers_kdiffusion_sdxl.py, vae.py, layerdiff3d.py,
transformer3d.py) and, in helpers.py, the functions they use from ``common/utils`` (torch_utils.py, cv.py).
Changes: imports made package-relative. The weights (24yearsold/seethroughv0.0.2_layerdiff3d_nf4) are
openrail++ with use restrictions (see docs/LICENSE_REVIEW.md). GGS's wrapper is ``ggs.pipeline.layers``.
"""
