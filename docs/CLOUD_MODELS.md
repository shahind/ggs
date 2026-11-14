# Cloud base models: Google Nano Banana (2026-11-13)

Owner request: "let user insert their API key and select nano banana models to generate characters and assets"
(Google AI Studio, Flash to Pro).

## What it is
A cloud model is a **base model** like `flux2-klein-4b` or `sdxl`: pick it per project (top bar, Base model) or per
command (`--model nano-banana-pro`). Everything that draws goes through it: assets of every class, character
designs over the pose guide, Refine edits, slicing fills, background layers, seamless tile fixes. It is a
`GeminiGenerator` (`src/ggs/pipeline/cloud.py`) with the same `render(jobs)` contract as the local generators and
`supports_reference = True`, so it takes the klein path (reference images, edit instructions).

| GGS id | Google model | Sizes | Refs | List price / image |
|---|---|---|---|---|
| `nano-banana-2.1` | `gemini-nano-banana-2.1` | 1K 2K 4K | 14 | $0.034 (1K) |
| `nano-banana-2` | `gemini-3.1-flash-image` | 512 1K 2K 4K | 14 | $0.067 (1K) |
| `nano-banana-2-lite` | `gemini-3.1-flash-lite-image` | 1K | 3 | $0.034 |
| `nano-banana-pro` | `gemini-3-pro-image` | 1K 2K 4K | 14 | $0.134 (1K) |
| `nano-banana` | `gemini-2.5-flash-image` (legacy, off 2027-03-15) | 1K only | 3 | $0.039 |

The catalogue lives in `models/pipelines.yaml` (`cloud:`); a new Google model is a new entry there plus a registry
entry. Model ids and prices were read from aistudio.google.com/docs (models, image-generation, pricing) on
2026-11-13. No image model has a free tier.

## How a request is made
- REST `POST v1beta/models/{model}:generateContent`, key in the `x-goog-api-key` header (never in the URL),
  stdlib `urllib` (no SDK to bundle). Google calls this API "legacy but fully supported" next to its newer
  Interactions API; switching is one function (`GeminiGenerator.body` / `parse_image`).
- `generationConfig`: `responseModalities: ["IMAGE"]`, `imageConfig.aspectRatio` (nearest allowed ratio),
  `imageConfig.imageSize` (smallest size covering the longer side; not sent to the legacy model), `seed`
  (dropped automatically if a model rejects it). The answer is centre-cropped and resized to the recipe size.
- Reference images go first as `inline_data` PNG (at most 1536 px, at most the model's reference count).
- Thought images (`thought: true` parts) are skipped; the last image part is the result. Blocked prompts and
  text-only answers become a plain error message.
- Several seeds/layers run as parallel requests (4 at a time). 429 and 5xx are retried with backoff.

## What stays local
Matting (BiRefNet) and QA run on this computer, so transparency is real (hard rule 5): the model draws on white
and BiRefNet cuts it out. BiRefNet falls back to the CPU when there is no NVIDIA GPU, so a laptop without a GPU
can use the cloud models (`ggs generate` skips the profile check for them). Style LoRAs cannot load on a cloud
model: style words work; style training stays on the local models.

## The key
`GEMINI_API_KEY` / `GOOGLE_API_KEY`, or `ggs cloud key set` / the app (Settings > Google AI > API Key):
`~/.ggs/google_api_key`, encrypted with Windows DPAPI for the current user. `ggs cloud key check` reads the
model's info (draws nothing, costs nothing). With a key saved, first-run setup no longer requires the 10 GB local
drawing pack (the 0.85 GB cut-out pack still is).

## Determinism (hard rule 1)
The CLI still takes an explicit class and never asks an LLM to interpret the request; the same command does the
same thing whoever calls it. The image itself is not bit-for-bit reproducible: Google does not promise the same
image for the same seed. Each result's JSON records the model and a `cloud` block (images, size, list-price
estimate).
