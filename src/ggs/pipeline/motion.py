"""Text to motion with NVIDIA Kimodo (SOMA skeleton): prompt -> 3D motion -> BVH, then the rig's ``soma`` mocap
profile retargets it to a 2D clip like any other motion capture.

Stages (hard rule 2, one heavy model on the GPU at a time):
  ``motion_text``  LLM2Vec (Llama-3-8B-Instruct + McGill's MNTP and supervised adapters, exactly as Kimodo builds it)
              kept in RAM in bf16 and streamed onto the GPU block by block (~1 GB peak). One 4096-number
              embedding per sentence, cached on disk, so a prompt is read only once. Embeddings shipped in
              ``motions/embeds`` (the starter library's prompts) need no text encoder at all.
  ``motion``  Kimodo (282 M parameters, < 3 GB), every seed in one load.
Kimodo's foot-contact clean-up needs a compiled C++ extension, so it is off; the 2D retarget does its own
grounding. A prompt with several sentences makes one motion whose parts follow each other ("A person walks
forward. The person jumps."), like Kimodo's own command line.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from ggs.paths import repo_root, shared_dir
from ggs.pipeline.vram import Stages

MODEL = "kimodo-soma"
FPS = 30
MAX_SECONDS = 10.0
MIN_SECONDS = 1.0
STEPS = 100
ENCODER_KEY = "llm2vec-llama3-8b-instruct-mntp-supervised|bf16"
LLAMA3_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # LLM2Vec picks its prompt template by this name


class MotionError(ValueError):
    pass


def tidy_text(text: str) -> str:
    """A sentence as Kimodo sanitises it before asking its text encoder (single spaces, capital first letter,
    final punctuation), so the embedding GGS computes is found under the same text."""
    t = " ".join(text.split())
    while t and not t[0].isalnum():
        t = t[1:]
    while t and not t[-1].isalnum() and t[-1] not in ".!?\"])'":
        t = t[:-1]
    if not t:
        return t
    t = t.capitalize()
    return t if t[-1] in ".!?" else t + "."


def split_prompt(prompt: str) -> list[str]:
    """Sentences of a prompt (split after '.', '!' or '?'), each tidied the way Kimodo will see it."""
    import re

    parts = re.split(r"(?<=[.!?])\s+|\n+", prompt.strip())
    texts = [tidy_text(t) for t in parts if t.strip(" .!?")]
    texts = [t for t in texts if t]
    if not texts:
        raise MotionError("the motion prompt is empty")
    return texts


def frames_for(texts: list[str], seconds: float | list[float]) -> list[int]:
    """Frame count of every sentence; one ``seconds`` value is shared by all of them."""
    secs = [float(seconds)] * len(texts) if isinstance(seconds, (int, float)) else [float(s) for s in seconds]
    if len(secs) != len(texts):
        raise MotionError(f"{len(texts)} sentences but {len(secs)} durations")
    total = sum(secs)
    if min(secs) < MIN_SECONDS or total > MAX_SECONDS:
        raise MotionError(f"each part needs at least {MIN_SECONDS:g} s and the whole motion at most "
                          f"{MAX_SECONDS:g} s (got {total:g} s)")
    return [round(s * FPS) for s in secs]


def embed_name(text: str) -> str:
    return hashlib.sha256(f"{ENCODER_KEY}|{text}".encode()).hexdigest()[:24] + ".pt"


def embed_dirs() -> list[Path]:
    """Where embeddings are looked up: the user cache (written), then the shipped library's."""
    return [shared_dir() / "cache" / "motion_embeds", repo_root() / "motions" / "embeds"]


def find_embed(text: str, dirs: list[Path] | None = None) -> Path | None:
    for d in dirs or embed_dirs():
        p = d / embed_name(text)
        if p.is_file():
            return p
    return None


def _load_encoder(files: dict[str, Path]):
    """LLM2Vec as Kimodo builds it: bidirectional Llama, MNTP adapter merged, supervised adapter on top."""
    import torch
    from kimodo.model.llm2vec.llm2vec import LLM2Vec
    from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
    from peft import PeftModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(files["mntp"]))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = LlamaBiModel.from_pretrained(str(files["base"]), dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(files["mntp"])).merge_and_unload()
    model = PeftModel.from_pretrained(model, str(files["supervised"]))
    model.config._name_or_path = LLAMA3_NAME
    model.get_base_model().config._name_or_path = LLAMA3_NAME
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return LLM2Vec(model=model, tokenizer=tok)


def encode_texts(texts: list[str], stages: Stages, files: dict[str, Path] | None = None) -> dict[str, object]:
    """Embedding ([4096] float32 tensor) of every text. The text encoder loads only if some text is new."""
    import torch

    found = {t: find_embed(t) for t in texts}
    todo = [t for t, p in found.items() if p is None]
    out = {t: torch.load(p, map_location="cpu", weights_only=True) for t, p in found.items() if p is not None}
    if not todo:
        return out
    if files is None:
        from ggs.pipeline.models import resolve_motion_text

        files = resolve_motion_text(MODEL)
    from diffusers.hooks import apply_group_offloading

    cache = embed_dirs()[0]
    cache.mkdir(parents=True, exist_ok=True)
    import transformers

    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    with stages.stage("motion_text"):
        enc = _load_encoder(files)
        cuda = torch.cuda.is_available()
        if cuda:
            apply_group_offloading(enc.model, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                   offload_type="leaf_level", use_stream=False)
        # LLM2Vec.encode moves the whole model to the device first; the offload hooks do that block by block
        enc.to = _stays(enc)
        with torch.no_grad():
            for t in todo:
                emb = enc.encode([t], batch_size=1, show_progress_bar=False, device="cuda" if cuda else "cpu")
                emb = emb[0].float().cpu()
                torch.save(emb, cache / embed_name(t))
                out[t] = emb
        del enc
    return out


def _stays(module):
    def to(*a, **k):
        return module

    return to


class _CachedText:
    """Stands in for Kimodo's text encoder: hands back embeddings computed (and unloaded) beforehand."""

    def __init__(self, embeds: dict[str, object], device: str):
        self.embeds, self.device = embeds, device

    def __call__(self, texts):
        import torch

        texts = [texts] if isinstance(texts, str) else list(texts)
        rows = [self._lookup(t) if t.strip() else torch.zeros(4096) for t in texts]
        feat = torch.stack(rows)[:, None].to(self.device)
        return feat, [1] * len(texts)

    def _lookup(self, text: str):
        """The embedding for ``text``, matched loosely (Kimodo may re-case or re-punctuate a sentence)."""
        if text in self.embeds:
            return self.embeds[text]

        def key(t: str) -> str:
            return " ".join("".join(c.lower() if c.isalnum() else " " for c in t).split())

        found = {key(k): v for k, v in self.embeds.items()}
        if key(text) in found:
            return found[key(text)]
        raise MotionError(f"no text embedding for {text!r} (have: {list(self.embeds)})")

    def to(self, *a, **k):
        return self

    def eval(self):
        return self


@dataclass
class Motion:
    """One generated take: SOMA 77-joint positions and rotations at 30 fps (metres, y up, +z forward)."""

    seed: int
    posed_joints: object      # [T, 77, 3]
    global_rot_mats: object   # [T, 77, 3, 3]
    foot_contacts: object     # [T, 4]

    @property
    def frames(self) -> int:
        return int(self.posed_joints.shape[0])


def generate(prompt: str, seconds: float | list[float], seeds: list[int], stages: Stages,
             steps: int = STEPS, model_dir: Path | None = None) -> list[Motion]:
    """One motion per seed for ``prompt`` (one Kimodo load for all seeds)."""
    import torch

    texts = split_prompt(prompt)
    num_frames = frames_for(texts, seconds)
    embeds = encode_texts(texts, stages)
    if model_dir is None:
        from ggs.pipeline.models import resolve_motion

        model_dir = resolve_motion(MODEL)
    from kimodo.model.load_model import load_model
    from kimodo.tools import seed_everything

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out: list[Motion] = []
    with stages.stage("motion"):
        os.environ["CHECKPOINT_DIR"] = str(Path(model_dir).parent)
        model = load_model(Path(model_dir).name, device=device, text_encoder=_CachedText(embeds, device))
        sample = model._generate  # the multi-sentence path does not pass progress_bar on: give it ours
        for n, seed in enumerate(seeds):
            bar = _Progress(n, len(seeds))
            model._generate = lambda *a, _s=sample, _b=bar, **k: _s(*a, **{**k, "progress_bar": _b})
            seed_everything(seed)
            with torch.no_grad():
                r = model(texts, num_frames, num_denoising_steps=steps, num_samples=1, multi_prompt=True,
                          post_processing=False, return_numpy=True, progress_bar=bar)
            out.append(Motion(seed, r["posed_joints"][0], r["global_rot_mats"][0], r["foot_contacts"][0]))
        out_skeleton = model.output_skeleton.to("cpu")
        del model._generate, sample  # break the model -> hook -> model loop so the weights can be freed
        del model
    for m in out:
        m.skeleton = out_skeleton  # type: ignore[attr-defined]
    return out


class _Progress:
    """Kimodo's progress-bar hook: ``@progress N`` lines (0-100 over all takes) for the app instead of tqdm."""

    def __init__(self, take: int, takes: int):
        self.take, self.takes, self.last = take, takes, -1

    def __call__(self, it, *a, **k):
        items = list(it)
        for i, x in enumerate(items):
            pct = int(100 * (self.take + (i + 1) / max(1, len(items))) / self.takes)
            if pct // 5 != self.last // 5:
                self.last = pct
                print(f"@progress {pct}", flush=True)
            yield x


def save_bvh(motion: Motion, path: Path) -> Path:
    """Write the take as a somaskel77 BVH (standard T-pose rest, centimetres) for ``ggs mocap retarget``."""
    import torch
    from kimodo.exports.bvh import save_motion_bvh
    from kimodo.skeleton import global_rots_to_local_rots

    skel = motion.skeleton  # type: ignore[attr-defined]
    pos = torch.as_tensor(motion.posed_joints, dtype=torch.float32)
    rot = torch.as_tensor(motion.global_rot_mats, dtype=torch.float32)
    local = global_rots_to_local_rots(rot, skel)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_motion_bvh(path, local, pos[:, skel.root_idx, :], skeleton=skel, fps=FPS, standard_tpose=True)
    return path
