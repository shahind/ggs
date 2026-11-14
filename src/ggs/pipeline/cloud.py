"""Cloud base models: Google's Nano Banana family through the Gemini API, with the user's own API key.

A cloud model is used exactly like a local generator (``render`` jobs of (prompt, seed[, reference images])), so
every recipe, the character designer, refine and the asset kinds work with it unchanged. Differences:

- nothing is downloaded and no VRAM is used for drawing; matting (BiRefNet) and QA still run on this computer, so
  transparency is real (hard rule 5): the model draws on white, BiRefNet cuts it out;
- results are not bit-for-bit reproducible: the seed is sent, but Google does not promise the same image back;
- trained styles (LoRAs) cannot load on it: style words and reference images can.

The API key comes from ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` or from ``ggs cloud key set`` (stored in
~/.ggs/google_api_key, encrypted for this Windows user with DPAPI). It is sent only to Google, in a header.
Requests use the REST ``models/{model}:generateContent`` call (stdlib only, no SDK to bundle).
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ggs.pipeline.models import CloudSpec, PipelinesFile, load_pipelines
from ggs.pipeline.vram import Stages

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MODEL_INFO = "https://generativelanguage.googleapis.com/v1beta/models/{model}"
KEY_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
SIZE_PX = {"512": 512, "1K": 1024, "2K": 2048, "4K": 4096}
REF_MAX_PX = 1536  # reference images are sent at most this big (upload size; the models read ~1K anyway)
TIMEOUT_S = 180
RETRIES = 4  # on rate limits (429) and server errors (5xx)
WORKERS = 4  # requests in flight at once (several seeds / layers)


class CloudError(OSError):
    """Talking to the cloud model failed (no key, refused prompt, network, quota). The message is for the user."""


# ------------------------------------------------------------------ which models are cloud models


def cloud_spec(name: str | None, spec_file: PipelinesFile | None = None) -> CloudSpec | None:
    if not name:
        return None
    return (spec_file or load_pipelines()).cloud.get(name)


def is_cloud(name: str | None, spec_file: PipelinesFile | None = None) -> bool:
    return cloud_spec(name, spec_file) is not None


# ------------------------------------------------------------------ the API key


def key_file() -> Path:
    from ggs.paths import shared_dir

    return shared_dir() / "google_api_key"


def _dpapi(data: bytes, protect: bool) -> bytes:
    """Windows DPAPI: only this Windows user on this computer can decrypt the result."""
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    src, dst = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), Blob()
    crypt = ctypes.windll.crypt32
    fn = crypt.CryptProtectData if protect else crypt.CryptUnprotectData
    ok = fn(ctypes.byref(src), None, None, None, None, 0x1, ctypes.byref(dst))  # 0x1: CRYPTPROTECT_UI_FORBIDDEN
    if not ok:
        raise CloudError("could not read the saved Google API key (saved by another Windows user?); "
                         "set it again with `ggs cloud key set`")
    try:
        return ctypes.string_at(dst.pbData, dst.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(dst.pbData)


def save_key(key: str, path: Path | None = None) -> Path:
    key = key.strip()
    if not key or any(c.isspace() for c in key):
        raise CloudError("that does not look like an API key (empty or contains spaces)")
    path = path or key_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = key.encode("utf-8")
    if sys.platform == "win32":
        path.write_text("dpapi:" + base64.b64encode(_dpapi(raw, True)).decode("ascii"), encoding="ascii")
    else:
        path.write_text("plain:" + key, encoding="utf-8")
        path.chmod(0o600)
    return path


def saved_key(path: Path | None = None) -> str | None:
    path = path or key_file()
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    kind, _, body = text.partition(":")
    if kind == "dpapi":
        return _dpapi(base64.b64decode(body), False).decode("utf-8")
    if kind == "plain":
        return body
    raise CloudError(f"{path}: not a saved key; set it again with `ggs cloud key set`")


def clear_key(path: Path | None = None) -> bool:
    path = path or key_file()
    if path.exists():
        path.unlink()
        return True
    return False


def key_source() -> str | None:
    """Where the key would come from: an environment variable name, "saved", or None."""
    for env in KEY_ENV:
        if os.environ.get(env, "").strip():
            return env
    return "saved" if key_file().exists() else None


def api_key() -> str:
    for env in KEY_ENV:
        v = os.environ.get(env, "").strip()
        if v:
            return v
    k = saved_key()
    if not k:
        raise CloudError("no Google API key: get one at aistudio.google.com (Get API key), then add it in "
                         "Settings > Google AI, or run `ggs cloud key set`")
    return k


def mask(key: str) -> str:
    return f"{key[:4]}...{key[-4:]}" if len(key) > 12 else "****"


# ------------------------------------------------------------------ sizes


def _ratio(a: str) -> float:
    w, h = a.split(":")
    return float(w) / float(h)


def nearest_aspect(width: int, height: int, allowed: list[str]) -> str:
    """The allowed aspect ratio closest to width:height (compared in log space, so 2:1 is as far from 1:1 as 1:2)."""
    import math

    want = math.log(width / height)
    return min(allowed, key=lambda a: abs(math.log(_ratio(a)) - want))


def image_size(width: int, height: int, sizes: list[str]) -> str | None:
    """Smallest accepted size that covers the longer side (the largest when none does); None = do not send."""
    if not sizes:
        return None
    ordered = sorted(sizes, key=lambda s: SIZE_PX.get(s, 1024))
    side = max(width, height)
    return next((s for s in ordered if SIZE_PX.get(s, 1024) >= side), ordered[-1])


def fit(img, width: int, height: int):
    """Centre-crop to width:height, then resize to exactly width x height."""
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    want = width / height
    if abs(w / h - want) > 1e-3:
        if w / h > want:
            nw = round(h * want)
            img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = round(w / want)
            img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return img if img.size == (width, height) else img.resize((width, height), Image.LANCZOS)


def _png_b64(img) -> str:
    from PIL import Image

    img = img.convert("RGB")
    if max(img.size) > REF_MAX_PX:
        s = REF_MAX_PX / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
    b = io.BytesIO()
    img.save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode("ascii")


# ------------------------------------------------------------------ transport

Transport = Callable[[str, dict, "dict | None", float], tuple[int, dict]]


def http_post(url: str, headers: dict, body: dict | None, timeout: float) -> tuple[int, dict]:
    """POST ``body`` as JSON (GET when it is None), return (status, parsed JSON). Network failures raise
    CloudError; HTTP errors come back as their status."""
    import urllib.error
    import urllib.request

    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8") or "{}")
        except ValueError:
            return e.code, {}
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        reason = getattr(e, "reason", e)
        raise CloudError(f"could not reach Google ({reason}); check the internet connection") from e


def _api_message(status: int, data: dict) -> str:
    err = data.get("error") if isinstance(data, dict) else None
    msg = (err or {}).get("message") or f"HTTP {status}"
    if status in (401, 403) or "API key not valid" in msg:
        return f"Google refused the API key: {msg}"
    if status == 429:
        return f"Google's rate limit or quota was reached: {msg}"
    return f"Google returned an error ({status}): {msg}"


# ------------------------------------------------------------------ the generator


class GeminiGenerator:
    """Nano Banana as a GGS generator: text-to-image, and editing / following reference images."""

    supports_reference = True
    cloud = True

    def __init__(self, spec: CloudSpec, key: str | None = None, transport: Transport | None = None,
                 workers: int = WORKERS, sleep: Callable[[float], None] = time.sleep):
        self.spec = spec
        self.name = spec.api_model
        self.key = key
        self.transport = transport or http_post
        self.workers = max(1, workers)
        self.sleep = sleep
        self.send_seed = True
        self.calls = 0  # requests that returned an image (for the job's cost line)

    def generate(self, prompt: str, seeds: list[int], width: int, height: int, stages: Stages,
                 lora=None, lora_scale: float = 1.0, reference: list | None = None) -> list:
        return self.render([(prompt, s, reference) for s in seeds], width, height, stages, lora, lora_scale)

    def render(self, jobs: list[tuple], width: int, height: int, stages: Stages, lora=None,
               lora_scale: float = 1.0) -> list:
        if lora is not None:
            raise CloudError(f"{self.spec.label} runs on Google's servers and cannot load a trained style (LoRA); "
                             "use style words, or a local base model for trained styles")
        if not jobs:
            return []
        key = self.key or api_key()
        with stages.stage("cloud"):
            if len(jobs) == 1:
                return [self._one(jobs[0], width, height, key)]
            with ThreadPoolExecutor(min(self.workers, len(jobs))) as pool:
                return list(pool.map(lambda j: self._one(j, width, height, key), jobs))

    def body(self, prompt: str, seed: int, refs: list | None, width: int, height: int) -> dict:
        parts: list[dict] = [{"inline_data": {"mime_type": "image/png", "data": _png_b64(r)}}
                             for r in (refs or [])[: self.spec.max_refs]]
        parts.append({"text": prompt})
        image_config = {"aspectRatio": nearest_aspect(width, height, self.spec.aspects)}
        size = image_size(width, height, self.spec.sizes)
        if size:
            image_config["imageSize"] = size
        config: dict = {"responseModalities": ["IMAGE"], "imageConfig": image_config}
        if self.send_seed:
            config["seed"] = int(seed) % 2**31
        return {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}

    def _one(self, job: tuple, width: int, height: int, key: str):
        prompt, seed = job[0], job[1]
        refs = job[2] if len(job) > 2 else None
        url = ENDPOINT.format(model=self.spec.api_model)
        headers = {"x-goog-api-key": key}
        for attempt in range(RETRIES + 1):
            status, data = self.transport(url, headers, self.body(prompt, seed, refs, width, height), TIMEOUT_S)
            if status == 200:
                img = parse_image(data)
                self.calls += 1
                return fit(img, width, height)
            msg = _api_message(status, data)
            if status == 400 and self.send_seed and "seed" in msg.lower():
                self.send_seed = False  # a model that rejects the field: ask again without it
                continue
            if (status == 429 or status >= 500) and attempt < RETRIES:
                self.sleep(min(60.0, 2.0 * 2 ** attempt) + random.random())
                continue
            raise CloudError(msg)
        raise CloudError(f"{self.spec.label}: no image after {RETRIES + 1} tries")


def parse_image(data: dict):
    """The final image of a generateContent response (thought images are skipped)."""
    from PIL import Image

    fb = data.get("promptFeedback") or {}
    if fb.get("blockReason"):
        raise CloudError(f"Google blocked the prompt ({fb['blockReason']}); change the wording and try again")
    cands = data.get("candidates") or []
    if not cands:
        raise CloudError("Google returned no image (empty answer)")
    c = cands[0]
    image, text = None, []
    for p in (c.get("content") or {}).get("parts") or []:
        if p.get("thought"):
            continue
        blob = p.get("inlineData") or p.get("inline_data")
        if blob and str(blob.get("mimeType") or blob.get("mime_type") or "image/").startswith("image/"):
            image = blob["data"]  # the last image part is the final render
        elif p.get("text"):
            text.append(p["text"].strip())
    if image is None:
        why = c.get("finishReason") or ""
        said = f": {' '.join(text)[:300]}" if text else ""
        if why and why not in ("STOP", "MAX_TOKENS"):
            raise CloudError(f"Google did not draw the image ({why}){said}")
        raise CloudError(f"Google answered without an image{said}")
    return Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")


def cost(gen, width: int, height: int) -> dict | None:
    """What a job asked of the cloud: model, images drawn, list-price estimate (USD). None for local models."""
    if not getattr(gen, "cloud", False):
        return None
    size = image_size(width, height, gen.spec.sizes) or "1K"
    per = gen.spec.price.get(size)
    return {"model": gen.spec.api_model, "size": size, "images": gen.calls,
            "usd_estimate": None if per is None else round(per * gen.calls, 4)}


def make_cloud_generator(spec: CloudSpec) -> GeminiGenerator:
    return GeminiGenerator(spec)


def check_key(spec: CloudSpec, key: str | None = None, transport: Transport | None = None) -> str:
    """A key check that draws nothing (and costs nothing): read the model's info. Returns its display name."""
    status, data = (transport or http_post)(MODEL_INFO.format(model=spec.api_model),
                                            {"x-goog-api-key": key or api_key()}, None, 30)
    if status != 200:
        raise CloudError(_api_message(status, data))
    return data.get("displayName") or spec.api_model
