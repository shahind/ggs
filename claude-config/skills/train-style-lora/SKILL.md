---
name: train-style-lora
description: Procedure for turning a user's reference images into a portable style LoRA (dataset check, rights acknowledgement, captioning, training config, sample review, packaging). Use when implementing or running style training.
---

# Train a style LoRA

1. **Rights gate.** Show the warning that the user is responsible for the rights to the training images and require
   the acknowledgement checkbox. Store the acknowledgement timestamp in the package metadata.
2. **Dataset check.** Aim for 30-60 images. Reject images that are too small (< 512px short side), near-duplicates, or
   obviously inconsistent in style. Show the user a contact sheet and let them remove images.
3. **Bootstrap option.** If the user has few or messy references, generate a candidate board from them, let the user
   approve/reject, and train on the approved set.
4. **Captions.** Auto-caption with a VLM describing content only (not style); add the trigger word to every caption.
5. **Config by profile.** `standard`: 768px+, train locally. `lite`: 512px, offload text encoder and layers, warn that
   quality is lower, or export a dataset bundle for cloud training. Check the base model's current training guide for
   step counts and learning rates (style LoRAs are typically in the low thousands of steps).
6. **Samples.** Render samples every few hundred steps with fixed prompts and seeds; pick the best checkpoint with the user.
7. **Package.** `<name>.ggsstyle` = safetensors + `style.yaml` (base model id, trigger word, dataset hash, rights
   acknowledgement, sample images). Loading checks the base model id matches.
8. Tests: CPU-only test for packaging/metadata validation; `@pytest.mark.gpu` smoke training for ~50 steps.
