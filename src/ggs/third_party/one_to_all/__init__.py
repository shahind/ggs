"""One-to-All Animation (CVPR 2026), vendored inference code for the 1.3B model.

Source: https://github.com/ssj9596/One-to-All-Animation at commit b3c9d886c93c43e767b995e93968708ee2d410cb,
Apache License 2.0 (LICENSE in this folder). Only the files the 1.3B inference path needs are copied:
``video-generation/opensora/model_variants/wanx_diffusers_src`` (transformer.py is
``transformer_wan_refextractor_2d_controlnet_last_prefix.py``), ``controlnet_modules/controlnet.py``,
``vae_variants/hunyuan_vae_src/unet_causal_3d_blocks.py`` and ``sample/pipeline_wanx_vhuman_tokenreplace.py``
(pipeline.py). Changes: imports made package-relative; ftfy made optional in pipeline.py. GGS's wrapper is
``ggs.pipeline.one_to_all``.
"""
