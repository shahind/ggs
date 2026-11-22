# GGS — Game Graphic Studio

Local AI tool (with optional Google cloud models) for making 2D game graphics without graphic skills. See `CLAUDE.md` for the project brief,
`docs/ROADMAP.md` for milestones and `docs/KICKOFF.md` for how to start a Claude Code session here.

## Install

**Windows 10/11 (64-bit):** download `GGS-Setup-<version>.exe` from
[Releases](https://github.com/shahind/ggs/releases) and run it. It is small: during setup it downloads Python, the
libraries (the CUDA build of PyTorch when there is an NVIDIA GPU), the motion library and, if ticked, the AI models
(about 11 GB). No admin rights needed. "GGS Repair" in the Start menu re-runs that step.

**macOS (Apple Silicon, macOS 14 or newer):** download `GGS-<version>-macos-arm64.dmg`, drag GGS to Applications,
then right-click GGS > Open the first time (the app is not signed; on macOS 15 use System Settings > Privacy &
Security > Open Anyway). The first start sets everything up in a Terminal window. A Mac has no NVIDIA GPU, so it
draws with Google's Nano Banana models (your own API key, Settings > Google AI); cutting out, slicing, warping,
the animation library and exporting run on the Mac. See `docs/CLOUD_MODELS.md`.

**From source** (any of the above, or to develop): clone the repo and run the starter. It fetches
[uv](https://github.com/astral-sh/uv), which installs Python 3.13 and the libraries at the tested versions
(`requirements/constraints.txt`) into `.venv`, and downloads the motion library. Run it again to repair or update.

```powershell
git clone https://github.com/shahind/ggs.git; cd ggs
.\setup.cmd                      # Windows;  macOS / Linux:  ./setup.sh
.\setup.cmd --models required    # also download the AI models now (else the app offers them on first start)
.venv\Scripts\python -m ggs ui    # the desktop app;  macOS:  .venv/bin/python -m ggs ui
.venv\Scripts\python -m pytest
```

Large files are not in git: the AI models come from Hugging Face (`models/downloads.yaml`, `ggs models fetch`),
the motion library from a GitHub release (`ggs motion fetch`). Building the installers: `tools/stage_app.py` and
`tools/installer_web.iss`, or the `installers` GitHub Actions workflow (push a `v*` tag to publish a release). The
full offline Windows installer (~5 GB, everything bundled) is still `tools/build_app.py` + `tools/installer.iss`.

`.\setup-claude.ps1` installs the Claude Code agents, skills and settings into `.claude\`.

Retarget a mocap clip onto the side-view rig (M1):

```powershell
python -m ggs mocap inspect data\mocap\cmu\02_01_walk.bvh
python -m ggs mocap retarget data\mocap\cmu\02_01_walk.bvh --profile cmu_bvh --fps 24 --out walk.json
python -m ggs mocap preview walk.json --out walk.gif
```

Make a looping walk cycle and export it with placeholder art (M2, see `docs/EXPORT_FORMATS.md`):

```powershell
python -m ggs clip loop walk.json --out cycle.json
python -m ggs character programmer-art --out chars\pa
python -m ggs export cycle.json --character chars\pa\character.yaml --format apng --out walk.png
```

## Desktop app (v1)
```powershell
pip install -e ".[dev,image,gpu,ui]"     # see GPU setup below for the CUDA build of torch
python -m ggs ui                          # or: ggs-studio
```
Pages: **Project** (create/open, base model), **Generate** (asset type, style, prompt -> results with QA badges),
**Characters** (prompt, body shape, stature -> master, part sheet, test poses), **Animate** (character, motion from
a BVH, upper-body holds/swings, held item, format: APNG/WebP/GIF/sheet/rig/Godot/Unity/Defold), **Styles**
(import with the rights checkbox, captions, training, install) and **System** (doctor, model check, model folders).
Every button runs the same `ggs` command you could type; the log at the bottom shows it and its output, and jobs
run one at a time so each gets the whole GPU.

## GPU setup (image generation, M3)

Install the CUDA build of PyTorch **first**, then the `gpu` extra. Don't use `pip install -U` on the gpu
extra: it can replace CUDA torch with a CPU-only build from PyPI.

```powershell
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,image,gpu]"
python -m ggs models add-dir "$env:LOCALAPPDATA\Comfy-Desktop\ComfyUI-Shared\models"   # reuse ComfyUI weights
python -m ggs models check
python -m ggs generate --class sprite --prompt "a red health potion" --style-words "flat cartoon game art, bold outlines" --count 4 --out potion.png
```

Files that `models check` reports missing go into `%USERPROFILE%\.ggs\models` (see `models/pipelines.yaml`). Each
generated sprite is a trimmed RGBA PNG with a `.json` sidecar (prompt, seed, QA results, stage timings and VRAM).
