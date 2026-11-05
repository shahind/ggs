---
name: gpu-test-protocol
description: How to write and run GPU tests safely on the owner's Windows machine (16GB primary, 8GB lite profile): VRAM budgets, one-model-at-a-time checks, marking, and what to report. Use for any code that loads a diffusion, motion, or segmentation model.
---

# GPU test protocol

- Mark every test that needs CUDA with `@pytest.mark.gpu`; they are skipped by default. Run them with
  `python -m pytest -m gpu` on the owner's machine only.
- First run `python -m ggs doctor` and record the profile. Skip tests whose model needs more VRAM than the profile offers.
- **Budget test:** record `torch.cuda.max_memory_allocated()` after each stage and assert it stays under the profile
  budget (standard: 14GB, lite: 7GB). Call `torch.cuda.reset_peak_memory_stats()` between stages.
- **Unload test:** after a stage finishes, `torch.cuda.memory_allocated()` should drop near zero before the next model loads.
- Use tiny settings (few steps, 512px, batch 1) for smoke tests; keep full-quality runs out of CI.
- Fix seeds; save outputs under `outputs/test/` (git-ignored) so the owner can look at them.
- Windows notes: install the CUDA build of torch from the official index before `pip install -e ".[gpu]"`;
  use pathlib; avoid multiprocessing DataLoader workers > 0 unless tested.
- Report: GPU name, VRAM, profile, peak VRAM per stage, wall time, and a path to the output images.
