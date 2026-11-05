"""Turn job output into plain sentences for the user (the raw text stays available under "details")."""

from __future__ import annotations

import re

STAGES = {  # @stage <name> markers (GGS_PROGRESS=1) -> what the user sees
    "encode": "Reading your description...",
    "denoise": "Drawing...",
    "decode": "Drawing...",
    "matte": "Cutting it out...",
    "latents": "Preparing your images...",
    "train": "Training the style (this takes a while)...",
    "control": "Reading the pose...",
    "caption": "Describing your images...",
    "download_drawing": "Downloading the drawing model...",
    "download_cutout": "Downloading the cut-out model...",
    "download_sdxl": "Downloading SDXL...",
    "download_captioner": "Downloading the image describer...",
    "download_motion": "Downloading the motion model...",
    "download_motion_text": "Downloading the motion text reader...",
    "motion_text": "Reading your description (about half a minute)...",
    "motion": "Animating...",
}

RULES = [
    (r"motion model kimodo-soma: none of|ggs models fetch motion\b",
     "The motion model is not downloaded yet. Press 'Download motion models...' (1.1 GB)."),
    (r"kimodo-soma text encoder|text encoder adapter",
     ("Describing new motions needs the motion text reader (15 GB download, and 16 GB of computer memory). "
      "Without it you can still make new takes of the Library's starter motions.")),
    (r"each part needs at least",
     "Pick a length between 1 and 10 seconds (all sentences together)."),
    (r"captioner .*none of|qwen3-vl-2b.*not found|Qwen3-VL.*missing",
     "Training a style needs the image describer (4 GB). Open Settings > Models and download it."),
    (r"needs FLUX.2 klein|cannot follow reference images|frame-by-frame sprites need",
     "Drawn frames need the FLUX.2 klein drawing model. Switch the project's base model to FLUX.2 klein."),
    (r"no design yet",
     "Press 'Use this design' on the character first."),
    (r"MissingModelError|none of \[.*\] found|weights missing|not found in \[",
     ("Some AI model files are missing. Open Settings > Models and add the folder that holds them "
     "(for example your ComfyUI models folder).")),
    (r"VramBudgetError|out of memory|CUDA out of memory|peaked at .* GB",
     ("Your graphics card ran out of memory. Close other programs that use the GPU (games, browsers with "
     "many tabs) and try again, or choose a smaller size in Advanced.")),
    (r"no CUDA|Torch not compiled with CUDA|CUDA is not available|profile '(minimal|no-gpu)'",
     "No supported NVIDIA graphics card was found. GGS needs an NVIDIA GPU with at least 8 GB of memory."),
    (r"Install the GPU stack|No module named 'torch'|No module named 'diffusers'",
     "The AI runtime is not installed. Run the GGS setup again and choose 'Install AI runtime'."),
    (r"resource already mapped|CUDA error",
     "The graphics driver hiccuped. Please try again; if it keeps happening, restart the app."),
    (r"nothing to refine",
     "There is nothing to change yet: describe the item and press Send first."),
    (r"describe the item first|say what to change",
     "Type what you want first."),
    (r"already a .* called",
     "That name is already used in this project. Pick another name."),
    (r"was trained on .* cannot load on",
     "This style was made for a different base model. Switch the base model at the top, or pick another style."),
    (r"is not a GGS project",
     "That folder is not a GGS project."),
]


def friendly_error(raw: str) -> str:
    for pat, text in RULES:
        if re.search(pat, raw, re.IGNORECASE):
            return text
    m = re.search(r"ERROR: (.+)", raw)
    if m:
        return "Something went wrong: " + m.group(1).strip().rstrip(".") + "."
    return "Something went wrong. Open 'details' to see what happened."


def stage_text(line: str) -> str | None:
    m = re.match(r"@stage (\w+)", line.strip())
    return STAGES.get(m.group(1), "Working...") if m else None


# ---------------------------------------------------------------- pop-ups (Qt imported lazily: the rules above
# are unit-tested without a display)

def show_message(parent, text: str, title: str = "GGS") -> None:
    """A plain information pop-up."""
    from PySide6.QtWidgets import QMessageBox

    QMessageBox.information(parent, title, text)


def show_error(parent, raw: str, title: str = "Something went wrong", text: str | None = None) -> None:
    """An error pop-up: the plain sentence on top, the job's raw output under "Show Details..."."""
    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(text or friendly_error(raw))
    if raw.strip():
        box.setDetailedText(raw.strip()[-6000:])
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()


def show_saved(parent, title: str, path, text: str | None = None) -> None:
    """'Saved to ...' with a button that opens the folder."""
    from pathlib import Path

    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle(title)
    box.setText(text or (f"Saved to\n{path}" if path else "Done."))
    open_btn = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole) if path else None
    box.addButton(QMessageBox.StandardButton.Ok)
    box.exec()
    if open_btn is not None and box.clickedButton() is open_btn:
        p = Path(path)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p if p.is_dir() else p.parent)))
