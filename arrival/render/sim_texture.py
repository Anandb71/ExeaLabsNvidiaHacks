"""sim_texture: 64x64 schematic RGB frames drawn from episode cartesian traces. Not a splat."""

from __future__ import annotations

import numpy as np

SIZE = 64
BACKGROUND = {"dim": (28, 32, 38), "neutral": (70, 78, 86), "bright": (210, 206, 196)}
COUNTER = (150, 140, 130)
OBJECT = (180, 90, 70)
EEF = (240, 240, 240)
APPEARANCE = "sim_texture"

_yy, _xx = np.mgrid[0:SIZE, 0:SIZE]


def _project(p, room_center_x: float) -> tuple[int, int]:
    px = 32 + (p[0] - room_center_x) * 8
    py = 40 - (p[2] - 0.9) * 40
    return int(round(min(max(px, 0), SIZE - 1))), int(round(min(max(py, 0), SIZE - 1)))


def render_frame(frame: dict, lighting: str, room_center_x: float = 2.0) -> np.ndarray:
    """One schematic SIZE x SIZE frame: the lighting's background, the counter as a band, the object as a disc and the
    end effector as a small square, projected from the frame's cartesian positions."""
    img = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
    img[:] = BACKGROUND[lighting]
    img[SIZE - 20 :, :] = COUNTER
    ox, oy = _project(frame["object_position"], room_center_x)
    img[(_xx - ox) ** 2 + (_yy - oy) ** 2 <= 16] = OBJECT
    ex, ey = _project(frame["eef"], room_center_x)
    img[max(ey - 1, 0) : ey + 2, max(ex - 1, 0) : ex + 2] = EEF
    return img


def render_episode(frames: list[dict], lighting: str, room_center_x: float = 2.0) -> np.ndarray:
    return np.stack([render_frame(f, lighting, room_center_x) for f in frames])


def room_center_x(twin: dict) -> float:
    xs = [p[0] for w in twin.get("walls", []) for p in (w["start"], w["end"])]
    return (min(xs) + max(xs)) / 2 if xs else 2.0
