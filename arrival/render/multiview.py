"""Human preview of synchronized views; labels never enter the training videos."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from arrival.export.lerobot import write_video


def write_paired_preview(stem: Path, ego: np.ndarray, exo: np.ndarray, fps: int) -> None:
    """Write <stem>.mp4 with the ego and exo views side by side, each frame labelled with its index and time, and
    <stem>.png of the middle frame. For people only: the labels never enter the training videos."""
    if ego.shape != exo.shape or ego.ndim != 4 or len(ego) == 0:
        raise ValueError("paired preview needs equally shaped, nonempty video arrays")
    n, h, w, _ = ego.shape

    def frame(i):
        image = Image.new("RGB", (w * 2, h + 32), "#111820")
        image.paste(Image.fromarray(ego[i]), (0, 32))
        image.paste(Image.fromarray(exo[i]), (w, 32))
        draw = ImageDraw.Draw(image)
        draw.text((4, 1), "EGO", fill="white")
        draw.text((w + 4, 1), "EXO", fill="white")
        draw.text((4, 16), f"{i:04d} | {i / fps:.2f}s", fill="#b7d7ee")
        return np.asarray(image)

    stem.parent.mkdir(parents=True, exist_ok=True)
    write_video(stem.parent / (stem.name + ".mp4"), (frame(i) for i in range(n)), fps)
    Image.fromarray(frame(n // 2)).save(stem.parent / (stem.name + ".png"))
