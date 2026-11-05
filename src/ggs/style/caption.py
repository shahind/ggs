"""Auto-captioning with a small vision-language model (Qwen3-VL-2B-Instruct; needs the ``gpu`` extra).

Captions describe only the content (subject, pose, colours, composition) and never the art style: the style
is what the LoRA learns and the trigger word stands for (BFL's klein training guidance).
"""

from __future__ import annotations

import re
from pathlib import Path

from ggs.pipeline.vram import Stages

INSTRUCTION = (
    "Write one caption for this image for training an image generator. Describe only what is shown: the main "
    "subject, its pose or view, notable parts, colours and the background. Do NOT mention the art style, medium, "
    "rendering, line work, shading technique, or words like illustration, drawing, painting, cartoon, sketch, "
    "watercolor or ink. One or two plain sentences, no lists."
)
STYLE_WORDS = re.compile(
    r"\b(illustrat\w*|drawing|drawn|painting|painted|cartoon\w*|sketch\w*|watercolou?r\w*|ink|inked|artwork|"
    r"art style|style|rendered|rendering|digital art|storybook|cel[- ]shaded|line ?work|brush\w*)\b", re.IGNORECASE)


def strip_style_words(caption: str) -> str:
    """Remove sentences that talk about the medium/style (the model does not always obey the instruction)."""
    parts = [s.strip() for s in caption.replace("\n", " ").split(".") if s.strip()]
    kept = [s for s in parts if not STYLE_WORDS.search(s)]
    return (". ".join(kept or parts[:1]) + ".").strip()


def with_trigger(trigger: str, caption: str) -> str:
    caption = caption.strip()
    return caption if caption.startswith(trigger) else f"{trigger}, {caption}"


class QwenVLCaptioner:
    name = "qwen3-vl-2b"

    def __init__(self, path: Path, max_new_tokens: int = 96):
        self.path = path
        self.max_new_tokens = max_new_tokens

    def caption(self, images: list, stages: Stages) -> list[str]:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        out = []
        with stages.stage("caption"):
            processor = AutoProcessor.from_pretrained(self.path)
            model = AutoModelForImageTextToText.from_pretrained(self.path, dtype=torch.bfloat16).to("cuda").eval()
            for img in images:
                messages = [{"role": "user", "content": [{"type": "image", "image": img.convert("RGB")},
                                                         {"type": "text", "text": INSTRUCTION}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                text = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                out.append(strip_style_words(text))
            del model, processor, inputs, ids
        return out
